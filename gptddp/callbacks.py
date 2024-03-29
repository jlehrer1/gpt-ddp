import os
from copy import deepcopy
from functools import partial
from typing import Any, Optional, Type, Union

import boto3
import numpy as np
import torch
import torchmetrics as tm
from torch.optim.lr_scheduler import _LRScheduler

import wandb
from gptddp.trainer import ModelTrainer


class ModelCallback:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def silentprint(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    def on_train_batch_end(self, modeltrainer: ModelTrainer, batch: tuple[torch.Tensor], outputs: torch.Tensor, batch_idx: int):
        pass

    def on_validation_batch_end(
        self, modeltrainer: ModelTrainer, batch: tuple[torch.Tensor], outputs: torch.Tensor, batch_idx: int
    ):
        pass

    def on_train_epoch_end(self, modeltrainer: ModelTrainer):
        pass

    def on_validation_epoch_end(self, modeltrainer: ModelTrainer):
        pass


class SampleTextGenerationCallback(ModelCallback):
    def __init__(
        self,
        write_path: str = "./sample_output",
        every_n_epochs: int = 1,
        every_n_batches: int = None,
        prompt: str = None,
        new_tokens: int = 1000,
        log_wandb: bool = False,
        quiet: bool = False,
    ) -> None:
        super().__init__(quiet=quiet)
        self.write_path = write_path
        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches

        self.prompt = prompt
        self.new_tokens = new_tokens
        self.log_wandb = log_wandb

        os.makedirs(write_path, exist_ok=True)

    def _sample_output_from_prompt(self, modeltrainer: ModelTrainer, epoch: int, step: int):
        prompt = torch.zeros((1, 1), dtype=torch.long) if self.prompt is None else self.prompt
        text = modeltrainer.model.module.generate(
            prompt,
            max_new_tokens=self.new_tokens,
        )
        # just for writing purposes
        text = text.split(" ")

        with open(
            os.path.join(
                self.write_path,
                f"epoch_{epoch}_step_{step}_sample_output.txt",
            ),
            "w",
        ) as f:
            for idx, word in enumerate(text):
                # write a newline every ten words to make the output
                # more human readable
                if idx % 10 == 0:
                    f.write(f"{word} \n")
                else:
                    f.write(f"{word} ")

        if self.log_wandb:
            table = wandb.Table(columns=["epoch", "step", "text"])
            table.add_data(epoch, step, " ".join(text))
            wandb.log({f"Text Generation Prompt={'No Prompt' if self.prompt is None else self.prompt[0: 100]} (etc.)": table})

    def on_train_batch_end(
        self,
        modeltrainer: ModelTrainer,
        batch: tuple[torch.Tensor],
        outputs: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.every_n_batches == 0:
            self._sample_output_from_prompt(modeltrainer, modeltrainer.epoch, modeltrainer.trainstep)

    def on_train_epoch_end(self, modeltrainer: ModelTrainer) -> None:
        curr_epoch = modeltrainer.epoch

        if curr_epoch % self.every_n_epochs == 0:
            # generate writing sample just from "empty" prompt
            self._sample_output_from_prompt(modeltrainer, curr_epoch, modeltrainer.trainstep)


class UploadCheckpointToS3(ModelCallback):
    """Custom PyTorch callback for uploading model checkpoints to a s3_resource bucket using a boto3
    resource object."""
    def __init__(
        self,
        path: str,
        desc: str,
        s3_resource: boto3.resource,
        bucket: str,
        upload_prefix: int = "model_checkpoints",
        n_epochs: int = 10,
        n_steps: int = None,
        quiet: bool = False,
    ) -> None:
        """
        Callback for uploading model checkpoints to s3_resource bucket.

        :param path: Local path to folder where model checkpoints are saved
        :param desc: Description of checkpoint that is appended to checkpoint file name on save
        :param s3_resource: boto3 resource object for s3_resource
        :param bucket: Name of s3_resource bucket to upload checkpoints to
        :param upload_prefix: Path in bucket/ to upload model checkpoints to, defaults to model_checkpoints
        :param n_epochs: Save checkpoint every n epochs
        :param n_steps: Save checkpoint every n steps
        :param quiet: Whether to print intermediate messages to stdout
        """
        super().__init__(quiet=quiet)
        self.path = path
        self.desc = desc

        self.s3_resource = s3_resource
        self.bucket = bucket
        self.upload_prefix = upload_prefix

        self.n_epochs = n_epochs
        self.n_steps = n_steps

        os.makedirs(self.path, exist_ok=True)

    def _save_and_upload_checkpoint(self, modeltrainer: ModelTrainer, epoch: int, step: int) -> None:
        checkpoint = f"checkpoint-{epoch}-step-{step}-desc-{self.desc}.ckpt"
        checkpoint_path = os.path.join(self.path, checkpoint)

        model = modeltrainer.model
        optimizer = modeltrainer.optimizer

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": np.mean(modeltrainer.valloss),
            },
            checkpoint_path,
        )

        self.silentprint(f"Uploading checkpoint at epoch {epoch} and step {step}")
        try:
            self.s3_resource.Bucket(self.bucket).upload_file(
                Filename=checkpoint_path,
                Key=os.path.join(self.upload_prefix, checkpoint_path.split("/")[-1]),
            )
        except Exception as e:
            self.silentprint(f"Error when uploading on epoch {epoch}")
            self.silentprint(e)

    def on_train_batch_end(
        self,
        modeltrainer: ModelTrainer,
        batch: tuple[torch.Tensor],
        outputs: torch.Tensor,
        batch_idx: int,
    ) -> None:
        epoch = modeltrainer.epoch
        step = modeltrainer.trainstep

        if self.n_steps is not None and step % self.n_steps == 0:
            self._save_and_upload_checkpoint(modeltrainer, epoch, step)

    def on_train_epoch_end(self, modeltrainer: ModelTrainer):
        epoch = modeltrainer.epoch
        step = modeltrainer.trainstep

        if epoch % self.n_epochs == 0:  # Save every ten epochs
            self._save_and_upload_checkpoint(modeltrainer, epoch, step)


class WandbMetricsCallback(ModelCallback):
    """A wrapper class for logging multiple metrics from
    torchmetrics"""

    def __init__(self, metrics: dict[str, tm.Metric], phases: list[str], project: str, name: str, quiet: bool = False) -> None:
        """A class for logging metrics

        :param metrics: A dictionary of {name: Metric class} to log
        :type metrics: dict[str, tm.Metric]
        :param phases: A list of phases to log for, will be shown in wandb this way
        :type phases: list[str]
        """
        super().__init__(quiet=quiet)
        self.gpu_id = int(os.environ["LOCAL_RANK"])

        if self.gpu_id == 0:
            wandb.init(project=project, name=name)

        self.metrics = metrics
        self.phases = phases

        self.phase_metrics = None
        self.step_metrics_container = None
        self.epoch_metrics_container = None

        try:
            self.phase_metrics = {
                phase: {name: deepcopy(metric).to(self.gpu_id) for name, metric in self.metrics.items()} for phase in self.phases
            }
            self.step_metrics_container = {phase: {name: 0 for name in self.metrics} for phase in self.phases}
            self.epoch_metrics_container = {phase: {name: [] for name in self.metrics} for phase in self.phases}
        except RuntimeError as r:
            if "invalid device ordinal" in str(r).lower():
                print(
                    f"Error. The {self.__class__.__name__} class expected more GPUs than are actually available. In order to not stop your job, we'll continue to log loss only."
                )
        except Exception as ex:
            name, args = type(ex).__name__, ex.args
            print(f"Found {name} error in {self.__class__.__name__} with args {args}. Trying to continue.")

    def compute_step(self, phase: str, preds: torch.Tensor, targets: torch.Tensor) -> Optional[dict[str, Any]]:
        """Calculates all metrics and logs them to wandb

        :param phase: model phase, i.e. train or val
        :type phase: str
        :param preds: tensor of model predictions (logits)
        :type preds: torch.Tensor
        :param targets: tensor of ground truth labels
        :type targets: torch.Tensor
        :return: dictionary containing the metric name for the given phase and its value
        :rtype: Optional[dict[str, Any]]
        """
        assert phase in self.phases, f"Phase {phase} not in phases set at initiliazation"
        curr_step_metrics = {}

        if self.phase_metrics is not None:
            # calculate metric no matter what to update internal state
            for metric in self.metrics:
                try:
                    r = self.phase_metrics[phase][metric](preds, targets).item()
                    self.step_metrics_container[phase][metric] = r
                except AttributeError:
                    self.silentprint(f"Metric {metric} did not return a float, not logging and continuing...")

            curr_step_metrics = {f"{phase}_{metric}": self.step_metrics_container[phase][metric] for metric in self.metrics}

        return curr_step_metrics

    def compute_epoch(self, phase: str) -> dict[str, float]:
        """Computes the epoch-level metrics given the metric state across all batches from the current phase

        :param phase: model phase to calculate metrics for, i.e train or val
        :type phase: str
        :return: dictionary of results
        :rtype: dict[str, float]
        """
        # store epoch metrics for "best metric" logging
        curr_epoch_metrics = {}
        if self.phase_metrics is not None:
            for metric in self.metrics:
                try:
                    r = self.phase_metrics[phase][metric].compute().item()
                    self.epoch_metrics_container[phase][metric].append(r)
                except AttributeError:  # if .item() is not valid since metric doesnt return a 0dim tensor
                    self.silentprint(f"Metric {metric} did not return a float, not logging and continuing...")

            # reset the metric classes for next epoch
            for metric in self.metrics:
                self.phase_metrics[phase][metric].reset()

            curr_epoch_metrics = {f"{phase}_{metric}": self.epoch_metrics_container[phase][metric][-1] for metric in self.metrics}

        return curr_epoch_metrics

    def finish(self, phase: str) -> dict[str, float]:
        summary_metrics = {}
        for metric in self.metrics:
            best_func = np.argmax if self.phase_metrics[phase][metric].higher_is_better else np.argmin

            summary_metrics[f"{metric}_best_epoch_{phase}"] = best_func(self.epoch_metrics_container[phase][metric])

            best_func = np.maximum if self.phase_metrics[phase][metric].higher_is_better else np.minimum
            summary_metrics[f"{metric}_best_value_{phase}"] = best_func(self.epoch_metrics_container[phase][metric])

        return summary_metrics

    # all of these Callbacks methods should only get called on the main process by the ModelTrainer
    # so it should be safe to use wandb.log without checking it it's initialized
    # but initialization I'm not sure, that's why we have the conditional
    def on_train_batch_end(self, modeltrainer: ModelTrainer, batch: tuple[torch.Tensor], outputs: torch.Tensor, batch_idx: int):
        _, targets = batch
        if modeltrainer.trainstep % modeltrainer.log_every_n_steps == 0:
            metric_results = self.compute_step(phase="train", preds=outputs, targets=targets)
            metric_results["train_loss"] = modeltrainer.trainloss[-1]
            metric_results["train_step"] = modeltrainer.trainstep
            metric_results["global_step"] = modeltrainer.trainstep + modeltrainer.valstep
            wandb.log(metric_results)

    def on_validation_batch_end(
        self, modeltrainer: ModelTrainer, batch: tuple[torch.Tensor], outputs: torch.Tensor, batch_idx: int
    ):
        if modeltrainer.valstep % modeltrainer.log_every_n_steps == 0:
            _, targets = batch

            metric_results = self.compute_step(phase="validation", preds=outputs, targets=targets)
            metric_results["validation_loss"] = modeltrainer.valloss[-1]
            metric_results["validation_step"] = modeltrainer.valstep
            metric_results["global_step"] = modeltrainer.trainstep + modeltrainer.valstep
            wandb.log(metric_results)

    def on_train_epoch_end(self, modeltrainer: ModelTrainer):
        metric_results = self.compute_epoch("train")
        metric_results["train_loss"] = np.mean(modeltrainer.trainloss)
        metric_results["epoch"] = modeltrainer.epoch
        wandb.log(metric_results)

    def on_validation_epoch_end(self, modeltrainer: ModelTrainer):
        metric_results = self.compute_epoch("validation")
        metric_results["validation_loss"] = np.mean(modeltrainer.valloss)
        metric_results["epoch"] = modeltrainer.epoch
        wandb.log(metric_results)


class WarmupAndSlowDecayScheduler(_LRScheduler):
    # Write a learning rate scheduler that uses the warmup and slow decay
    # scheduler from the original GPT paper
    # https://arxiv.org/pdf/2002.04745.pdf
    def __init__(
        self,
        optimizer: Union[Type[torch.optim.Optimizer], partial(torch.optim.Optimizer)],
        init_lr: float,
        peak_lr: float,
        final_lr: float,
        final_lr_scale: float,
        warmup_steps: int,
        decay_steps: int,
    ) -> None:
        """LR scheduler

        optimizer (Optimizer): Optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        final_lr (float): Final learning rate.
        final_lr_scale (float): Final learning rate scale
        warmup_steps (int): Warmup the learning rate linearly for the first N updates
        decay_steps (int): Steps in decay stages
        """
        self.optimizer = optimizer
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -np.log(final_lr_scale) / self.decay_steps

        self.init_lr = init_lr
        self.update_steps = 0

    def _decide_stage(self) -> tuple[int, Optional[int]]:
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        if self.warmup_steps <= self.update_steps < self.warmup_steps + self.decay_steps:
            return 1, self.update_steps - self.warmup_steps

        return 2, None

    def step(self, val_loss: Optional[torch.FloatTensor] = None) -> float:
        self.update_steps += 1
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.update_steps * self.warmup_rate
        elif stage == 1:
            self.lr = self.peak_lr * np.exp(-self.decay_factor * steps_in_stage)
        elif stage == 2:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        for g in self.optimizer.param_groups:
            g["lr"] = self.lr

        return self.lr
