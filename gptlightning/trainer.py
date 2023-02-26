import os
from functools import partial
from typing import Type, Union

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import (barrier, destroy_process_group,
                               init_process_group)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from gptlightning.metrics import Metrics


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        traindata: Dataset,
        valdata: Dataset,
        optimizer: Union[Type[nn.Module], partial(nn.Module)],
        lr_scheduler: Union[Type[nn.Module], partial(nn.Module)],
        criterion: nn.Module,
        max_epochs: int,
        callbacks: list[nn.Module] = None,
        metrics: Metrics = None,
        log_every_n_steps: int = 50,
        limit_val_batches: int = None,
        limit_train_batches: int = None,
    ) -> None:
        self.traindata = traindata
        self.valdata = valdata

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion

        self.max_epochs = max_epochs
        self.callbacks = callbacks
        self.metrics = metrics

        self.log_every_n_steps = log_every_n_steps
        self.limit_val_batches = limit_val_batches
        self.limit_train_batches = limit_train_batches

        # set up gpu for ddp training
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = model.to(self.gpu_id)
        self.model = DistributedDataParallel(model, device_ids=[self.gpu_id])

        # track train/val step, epoch for logging
        self.trainstep = 0
        self.valstep = 0
        self.epoch = 0

        # track loss (other metrics are handled by torchmetrics)
        self.trainloss = []
        self.valloss = []

        self.configure_optimizers()

    def setup_dataloaders(self, *args, **kwargs):
        self.trainloader = DataLoader(dataset=self.traindata, sampler=DistributedSampler(self.traindata), *args, **kwargs)
        self.valloader = DataLoader(dataset=self.valdata, sampler=DistributedSampler(self.valdata), *args, **kwargs)

    def configure_optimizers(self):
        self.optimizer = self.optimizer([p for p in self.model.parameters() if p.requires_grad])
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.lr_scheduler(self.optimizer)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        data, targets = batch
        data = data.to(self.gpu_id)
        targets = targets.to(self.gpu_id)

        self.optimizer.zero_grad()
        logits = self.model(data)

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = self.criterion(logits, targets)
        loss.backward()

        # if self.trainer.global_step < self.warmup_iter:
        #     lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_iter)
        #     for pg in optimizer.param_groups:
        #         pg["lr"] = lr_scale * self.learning_rate

        self.optimizer.step()

        self.trainloss.append(loss.cpu().item())

        # only run callbacks on main process
        if self.gpu_id == 0:
            if self.metrics is not None:
                self.metrics.compute_step(phase="train", preds=logits, targets=targets, log_every_n_steps=self.log_every_n_steps)
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_train_batch_end(self, batch, logits, batch_idx)

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        data, targets = batch
        data = data.to(self.gpu_id)
        targets = targets.to(self.gpu_id)

        logits = self.model(data)
        loss = self.criterion(logits, targets)
        self.valloss.append(loss.cpu().item())

        if self.gpu_id == 0:
            if self.metrics is not None:
                self.metrics.compute_step(
                    phase="validation", preds=logits, targets=targets, log_every_n_steps=self.log_every_n_steps
                )
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_validation_batch_end(self, batch, logits, batch_idx)

    def training_epoch(self):
        assert hasattr(self, "trainloader"), "Must run setup_dataloaders before training epoch"

        for i, batch in enumerate(tqdm(self.trainloader)):
            if self.limit_train_batches is not None and i > self.limit_train_batches:
                break

            self.training_step(batch=batch, batch_idx=i)
            self.trainstep += 1

        if self.gpu_id == 0:
            if self.metrics is not None:
                self.metrics.compute_epoch("train")
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_training_epoch_end(self)

    def validation_epoch(self):
        assert hasattr(self, "valloader"), "Must run setup_dataloaders before training epoch"

        for i, batch in enumerate(tqdm(self.valloader)):
            if self.limit_val_batches is not None and i > self.limit_val_batches:
                break
            self.training_step(batch=batch, batch_idx=i)
            self.valstep += 1

        if self.gpu_id == 0:
            if self.metrics is not None:
                self.metrics.compute_epoch("validation")
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_validation_epoch_end(self)

    def run(self):
        for epoch in tqdm(range(self.max_epochs)):
            self.epoch = epoch
            # set these for DDP
            self.trainloader.sampler.set_epoch(epoch)
            self.valloader.sampler.set_epoch(epoch)

            self.model.train()
            self.training_epoch()

            self.model.eval()
            with torch.no_grad():
                self.validation_epoch()
