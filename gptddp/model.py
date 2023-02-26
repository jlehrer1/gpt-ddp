from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        n_embd: int,
        head_size: int,
        context_length: int,
    ) -> None:
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # dont register as attribute / in gradient graph
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(context_length, context_length)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # dont want to transpose along batch dim
        affinity = q @ k.transpose(-2, -1) * C ** (-0.5)

        # masking out "future values" in between each matmul
        # makes this a decoder block
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        affinity = F.softmax(affinity, dim=-1)

        v = self.value(x)
        x = affinity @ v
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_embd: int,
        head_size: int,
        context_length: int,
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList([SelfAttentionHead(n_embd, head_size, context_length) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # concat along channel dim of (B, T, C) Tensors
        x = torch.concat(
            [head(x) for head in self.heads],
            dim=-1,
        )
        x = self.projection(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_embd: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        head_size = n_embd // n_heads
        self.attention = MultiHeadAttention(
            n_heads=n_heads,
            n_embd=n_embd,
            head_size=head_size,
            context_length=context_length,
        )

        # feedforward section
        # multiplier of 4 is recommended from attention is all you need paper
        # for inner matrix :shrug:
        self.ff = nn.ModuleDict(
            {
                "linear": nn.Linear(n_embd, 4 * n_embd),
                "projection": nn.Linear(4 * n_embd, n_embd),  # just calling it this for init
                "activation": nn.GELU(),
                "dropout": nn.Dropout(dropout),
            }
        )

        # Layernorm for training stability
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # wrapper for running moduledict
        self.mlp = lambda x: self.ff.dropout(self.ff.activation(self.ff.projection(self.ff.linear(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = x + self.attention(x)
        x = self.ln2(x)
        x = x + self.mlp(x)

        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_embd: int,
        context_length: int,
        dropout: float,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    n_embd=n_embd,
                    n_heads=n_heads,
                    context_length=context_length,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.tokenizer = tokenizer

        self.apply(self.standard_initialize)
        self.specialized_initialize()

    def standard_initialize(self, module: Union[nn.Linear, nn.Embedding, nn.LayerNorm]) -> None:
        """Initializes all linear, embedding and layernorm layers with values given by GPT-1 and GPT-2"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def specialized_initialize(self):
        """Initializes the projection layers. These layers are Linear layers at the end of a multiheaded attention block
        and at the end of the feedforward portion of a transformer decoder block. Practically speaking, since the forward of the
        decoder block is

        x = self.ln1(x)
        x = x + self.attention(x)
        x = self.ln2(x)
        x = x + self.ff(x)

        This guarantees that the contribution of the attention and FF layers is small at the beginning of training.
        This has been shown to improve convergence in gpt-2
        """
        for name, param in self.named_parameters():
            if name.endswith("projection.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / np.sqrt(2 * self.n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T = x.shape

        tok_emb = self.token_embedding(x)
        # maybe dont need unsqueeze but I don't know broadcasting
        pos_emb = self.positional_embedding(torch.arange(T).unsqueeze(0).to(x.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(
        self, prompt: Union[torch.Tensor, str], max_new_tokens: int, temperature: float = 0.8, sample_tokens: bool = False
    ):
        if not torch.is_tensor(prompt):
            try:
                # cast to tensor and make a batch dim
                prompt = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
            except AttributeError:
                raise RuntimeError(
                    f"""Prompt input is not tokenized and tokenizer was not provided to 
                    {self.__class__.__name__}. 
                    Either provide integer input or provide tokenizer to model initialization."""
                )

        # model params should all be on same device
        # with regular training or ddp, so this trick can get the right device
        # to move our prompt to (cuda:i for i=0,..,N gpu's)
        curr_device = next(self.parameters()).device
        prompt = prompt.to(curr_device)

        # Move model to eval() mode if needed
        # and cache state to set it back after generating tokens
        was_training = False
        if self.training:
            was_training = True
            self.eval()

        for _ in range(max_new_tokens):
            prompt_cond = prompt[:, -self.context_length :]
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
