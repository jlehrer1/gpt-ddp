from functools import partial
from typing import Type, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GPTModel

device = "cuda" if torch.cuda.is_available() else "cpu"


class GPT(pl.LightningModule):
    def __init__(
        self,
        optimizer: Union[Type[nn.Module], partial(nn.Module)],
        scheduler: Union[Type[nn.Module], partial(nn.Module)] = None,
        vocab_size: int = 50304,
        n_blocks: int = 6,
        n_heads: int = 4,
        n_embd: int = 64,
        context_length: int = 64,
        dropout: float = 0.0,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        optims = {}
        optimizer = self.optimizer(self.parameters())
        optims["optimizer"] = optimizer

        if self.scheduler is not None:
            scheduler = scheduler(optimizer)
            optims["scheduler"] = scheduler

        return optims

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, sample_tokens: bool = False):
        if not torch.is_tensor(prompt):
            try:
                # cast to tensor and make a batch dim
                prompt = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
            except AttributeError:
                raise RuntimeError(
                    f"Prompt input is not tokenized and tokenizer was not provided to {self.__class__.__name__}. Either provide integer input or provide tokenizer to model initialization."
                )

        prompt = prompt.to(device)

        # Move model to eval() mode if needed
        # and cache state to set it back after generating tokens
        was_training = False
        if self.training:
            was_training = True
            self.eval()

        for _ in range(max_new_tokens):
            prompt_cond = prompt[:, -self.base_model.context_length :]
            logits = self(prompt_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] / temperature  # becomes (1, context_length, C) -> (1, C)
            probs = F.softmax(logits, dim=-1)

            if sample_tokens:
                prompt_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
            else:
                _, prompt_next = torch.topk(probs, k=1, dim=-1)

            prompt = torch.cat((prompt, prompt_next), dim=1)  # (1, T+1)

        if self.tokenizer is not None:
            # remove the batch dim for decoding
            prompt = self.tokenizer.decode(prompt.cpu().squeeze())

        if was_training:
            self.train()

        return prompt
