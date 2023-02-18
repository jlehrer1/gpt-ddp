import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Union
from functools import partial

from .model import GPTModel

class GPT(pl.LightningModule):
    def __init__(
        self, 
        optimizer: Union[Type[nn.Module], partial],
        scheduler: Union[Type[nn.Module], partial] = None,
        vocab_size: int = 50304,
        n_blocks: int = 6, 
        n_heads: int = 4, 
        n_embd: int = 128, 
        block_size: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler 
        
        self.base_model = GPTModel(
            vocab_size=vocab_size,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )

    def forward(self, x):
        return self.base_model(x)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = y.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return loss 

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)

        return loss

    def configure_optimizers(self):
        optims = {}
        optimizer = self.optimizer(self.parameters())
        optims["optimizer"] = optimizer

        if self.scheduler is not None:
            scheduler = scheduler(optimizer)
            optims["scheduler"] = scheduler

        return optims


        

    
    