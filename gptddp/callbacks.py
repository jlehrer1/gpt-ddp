import os
from copy import deepcopy
from typing import Any, Optional

import boto3
import numpy as np
import torch
import torchmetrics as tm

import wandb
from gptddp.trainer import ModelTrainer


class ModelCallback:
    def __init__(self) -> None:
        pass

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
    ) -> None:
        self.write_path = write_path
        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches

        self.prompt = prompt
        self.new_tokens = new_tokens
        self.log_wandb = log_wandb

        os.makedirs(write_path, exist_ok=True)

    def _sample_output_from_prompt(self, modeltrainer: ModelTrainer, epoch: int, step: int):
        prompt = torch.zeros((1, 1), dtype=torch.long) if self.prompt is None else self.prompt
        text = modeltrainer.model.generate(
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
            self._sample_output_from_prompt(modeltrainer, modeltrainer.epoch, modeltrainer.traintep)

    def on_train_epoch_end(self, modeltrainer: ModelTrainer) -> None:
        curr_epoch = modeltrainer.epoch

        if curr_epoch % self.every_n_epochs == 0:
            # generate writing sample just from "empty" prompt
            self._sample_output_from_prompt(modeltrainer, curr_epoch, modeltrainer.trainstep)


class UploadCheckpointToS3(ModelCallback):
    """Custom PyTorch callback for uploading model checkpoints to a s3_resource bucket using a boto3
    resource object.

    Parameters:
    path: Local path to folder where model checkpoints are saved
    desc: Description of checkpoint that is appended to checkpoint file name on save
    upload_prefix: Path in bucket/ to upload model checkpoints to, defaults to model_checkpoints
    """

    def __init__(
        self,
        path: str,
        desc: str,
        s3_resource: boto3.resource,
        bucket: str,
        upload_prefix: int = "model_checkpoints",
        n_epochs: int = 10,
        n_steps: int = None,
    ) -> None:
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

        print(f"Uploading checkpoint at epoch {epoch} and step {step}")
        try:
            self.s3_resource.Bucket(self.bucket).upload_file(
                Filename=checkpoint_path,
                Key=os.path.join(self.upload_prefix, checkpoint_path.split("/")[-1]),
            )
        except Exception as e:
            print(f"Error when uploading on epoch {epoch}")
            print(e)

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

    def __init__(self, metrics: dict[str, tm.Metric], phases: list[str], project: str, name: str) -> None:
        """A class for logging metrics

        :param metrics: A dictionary of {name: Metric class} to log
        :type metrics: dict[str, tm.Metric]
        :param phases: A list of phases to log for, will be shown in wandb this way
        :type phases: list[str]
        """
        super().__init__()
        self.metrics = metrics
        self.phases = phases

        self.phase_metrics = {phase: {name: deepcopy(metric) for name, metric in self.metrics.items()} for phase in self.phases}

        self.step_metrics_container = {phase: {name: 0 for name in self.metrics} for phase in self.phases}

        self.epoch_metrics_container = {phase: {name: [] for name in self.metrics} for phase in self.phases}

        self.gpu_id = int(os.environ["LOCAL_RANK"])

        if self.gpu_id == 0:
            wandb.init(project=project, name=name)

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
        curr_step_metrics = None

        # calculate metric no matter what to update internal state
        for metric in self.metrics:
            try:
                r = self.phase_metrics[phase][metric](preds, targets).item()
                self.step_metrics_container[phase][metric] = r
            except AttributeError:
                print(f"Metric {metric} did not return a float, not logging and continuing...")

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
        for metric in self.metrics:
            try:
                r = self.phase_metrics[phase][metric].compute().item()
                self.epoch_metrics_container[phase][metric].append(r)
            except AttributeError:  # if .item() is not valid since metric doesnt return a 0dim tensor
                print(f"Metric {metric} did not return a float, not logging and continuing...")

        # reset the metric classes for next epoch
        for metric in self.metrics:
            self.epoch_metrics_container[phase][metric].reset()

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
