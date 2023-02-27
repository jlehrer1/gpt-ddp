import os
from functools import partial
from typing import Optional, Type, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


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
        callbacks: Optional[list[nn.Module]] = None,
        log_every_n_steps: Optional[int] = 50,
        limit_val_batches: Optional[int] = None,
        limit_train_batches: Optional[int] = None,
        val_loop_every_n_steps: Optional[int] = None,
        scaler: torch.cuda.amp.GradScaler = None,
    ) -> None:
        """Class train model using distributed data parallism.
        Handles setting up dataloaders + model layers (i.e. batchnorm)
        for DDP training

        :param model: Base torch model to train
        :type model: nn.Module
        :param traindata: torch dataset to train on
        :type traindata: Dataset
        :param valdata: torch dataset to use for val loop
        :type valdata: Dataset
        :param optimizer: torch optimizer, not initialized with model params
        :type optimizer: Union[Type[nn.Module], partial
        :param lr_scheduler: torch lr scheduler, not initialized with model params
        :type lr_scheduler: Union[Type[nn.Module], partial
        :param criterion: loss definition, i.e. CrossEntropyLoss
        :type criterion: nn.Module
        :param max_epochs: number of epochs to train for
        :type max_epochs: int
        :param callbacks: List of callbacks to use, defaults to None
        :type callbacks: list[nn.Module], optional
        :param log_every_n_steps: Number of steps to log metrics at, defaults to 50
        :type log_every_n_steps: int, optional
        :param limit_val_batches: Number of val batches to use in val phase, defaults to None
        :type limit_val_batches: int, optional
        :param limit_train_batches: Number of training batches to use in train phase, defaults to None
        :type limit_train_batches: int, optional
        """
        self.traindata = traindata
        self.valdata = valdata

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion

        self.max_epochs = max_epochs
        self.callbacks = callbacks

        self.log_every_n_steps = log_every_n_steps
        self.limit_val_batches = limit_val_batches
        self.limit_train_batches = limit_train_batches
        self.val_loop_every_n_steps = val_loop_every_n_steps

        # set up gpu for ddp training
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = self.model.to(self.gpu_id)
        self.model = DistributedDataParallel(self.model, device_ids=[self.gpu_id])

        # gradient scaler
        self.scaler = scaler
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

    def __compute_forward_and_loss(self, data, targets) -> float:
        logits = self.model(data)
        B, T, C = logits.shape
        loss = self.criterion(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # TODO: Implement this as lr scheduler
        # if self.trainer.global_step < self.warmup_iter:
        #     lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_iter)
        #     for pg in optimizer.param_groups:
        #         pg["lr"] = lr_scale * self.learning_rate

        data, targets = batch
        data = data.to(self.gpu_id)
        targets = targets.to(self.gpu_id)

        self.optimizer.zero_grad()

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits, loss = self.__compute_forward_and_loss(data, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, loss = self.__compute_forward_and_loss(data, targets)
            loss.backward()
            self.optimizer.step()

        self.trainloss.append(loss.cpu().item())

        # only run callbacks on main process
        if self.gpu_id == 0:
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_train_batch_end(self, batch, logits, batch_idx)

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        data, targets = batch
        data = data.to(self.gpu_id)
        targets = targets.to(self.gpu_id)

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits, loss = self.__compute_forward_and_loss(data, targets)
        else:
            logits, loss = self.__compute_forward_and_loss(data, targets)

        self.valloss.append(loss.cpu().item())

        if self.gpu_id == 0:
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_validation_batch_end(self, batch, logits, batch_idx)

    def training_epoch(self):
        assert hasattr(self, "trainloader"), "Must run setup_dataloaders before training epoch"

        for i, batch in enumerate(tqdm(self.trainloader, desc="Training epoch")):
            if self.limit_train_batches is not None and i > self.limit_train_batches:
                break

            self.training_step(batch=batch, batch_idx=i)
            self.trainstep += 1

            if self.val_loop_every_n_steps is not None and i % self.val_loop_every_n_steps == 0:
                self.validation_epoch()

        if self.gpu_id == 0:
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_train_epoch_end(self)

        self.trainloss = []

    def validation_epoch(self):
        assert hasattr(self, "valloader"), "Must run setup_dataloaders before training epoch"

        for i, batch in enumerate(tqdm(self.valloader, desc="Validation epoch")):
            if self.limit_val_batches is not None and i > self.limit_val_batches:
                break
            self.validation_step(batch=batch, batch_idx=i)
            self.valstep += 1

        if self.gpu_id == 0:
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback.on_validation_epoch_end(self)

        self.valloss = []

    def run(self):
        for epoch in tqdm(range(self.max_epochs), desc="Full epoch"):
            self.epoch = epoch
            # set these for DDP
            self.trainloader.sampler.set_epoch(epoch)
            self.valloader.sampler.set_epoch(epoch)

            self.model.train()
            self.training_epoch()

            self.model.eval()
            with torch.no_grad():
                self.validation_epoch()
