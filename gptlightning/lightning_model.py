from functools import partial
from typing import Type, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GPTModel


class GPT(pl.LightningModule):
    def __init__(
        self,
        optimizer: Union[Type[nn.Module], partial],
        scheduler: Union[Type[nn.Module], partial] = None,
        vocab_size: int = 50304,
        n_blocks: int = 6,
        n_heads: int = 4,
        n_embd: int = 64,
        context_length: int = 64,
        dropout: float = 0.0,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        self.base_model = GPTModel(
            vocab_size=vocab_size,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_embd=n_embd,
            context_length=context_length,
            dropout=dropout,
            tokenizer=tokenizer,
        )

    def forward(self, x):
        return self.base_model(x)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = y.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optims = {}
        optimizer = self.optimizer(self.parameters())
        optims["optimizer"] = optimizer

        if self.scheduler is not None:
            scheduler = scheduler(optimizer)
            optims["scheduler"] = scheduler

        return optims
