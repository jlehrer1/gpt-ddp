import os
from typing import *

import boto3
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

import wandb


class SampleTextGenerationCallback(Callback):
    def __init__(
        self,
        write_path: str = "./sample_output",
        every_n_epochs: int = 1,
        every_n_batches: int = None,
        prompt: str = None,
        new_tokens: int = 1000,
        log_wandb: bool = False,
    ) -> None:
        super().__init__()
        self.write_path = write_path
        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches

        self.prompt = prompt
        self.new_tokens = new_tokens
        self.log_wandb = log_wandb

        os.makedirs(write_path, exist_ok=True)

    def _sample_output_from_prompt(self, pl_module: pl.LightningModule, epoch: int, step: int):
        prompt = torch.zeros((1, 1), dtype=torch.long) if self.prompt is None else self.prompt
        text = pl_module.base_model.generate(
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
            wandb.log({f"Text Generation Prompt={'No Prompt' if self.prompt is None else self.prompt[0: 100]}...": table})

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        curr_epoch = pl_module.current_epoch
        step = pl_module.global_step

        if curr_epoch % self.every_n_epochs == 0:
            # generate writing sample just from "empty" prompt
            self._sample_output_from_prompt(pl_module=pl_module, epoch=curr_epoch, step=step)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: torch.Tensor,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        if batch_idx % self.every_n_batches == 0:
            epoch = pl_module.current_epoch
            self._sample_output_from_prompt(pl_module=pl_module, epoch=epoch, step=batch_idx)


class UploadCheckpointToS3(Callback):
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
        super().__init__()
        self.path = path
        self.desc = desc

        self.s3_resource = s3_resource
        self.bucket = bucket
        self.upload_prefix = upload_prefix

        self.n_epochs = n_epochs
        self.n_steps = n_steps

        os.makedirs(self.path, exist_ok=True)

    def _save_and_upload_checkpoint(self, trainer: pl.Trainer, epoch: int, step: int) -> None:
        checkpoint = f"checkpoint-{epoch}-step-{step}-desc-{self.desc}.ckpt"
        checkpoint_path = os.path.join(self.path, checkpoint)

        trainer.save_checkpoint(checkpoint_path)

        print(f"Uploading checkpoint at epoch {epoch}")

        try:
            self.s3_resource.Bucket(self.bucket).upload_file(
                Filename=checkpoint_path,
                Key=os.path.join(self.upload_prefix, checkpoint_path.split("/")[-1]),
            )
        except Exception as e:
            print(f"Error when uploading on epoch {epoch}")
            print(e)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: torch.Tensor, batch: Any, batch_idx: int
    ) -> None:
        epoch = trainer.current_epoch
        step = trainer.global_step

        if self.n_steps is not None and step % self.n_steps == 0:
            self._save_and_upload_checkpoint(trainer, epoch, step)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        step = trainer.global_step

        if epoch % self.n_epochs == 0:  # Save every ten epochs
            self._save_and_upload_checkpoint(trainer, epoch, step)
