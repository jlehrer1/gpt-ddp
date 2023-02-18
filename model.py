import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # dont want to transpose along batch dim
        affinity = q @ k.transpose(-2, -1) * C ** (-0.5)

        # masking out "future values" in between each matmul
        # makes this a decoder block
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        affinity = F.softmax(affinity)

        x = affinity @ self.value(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()

        self.heads = nn.ModuleList([SelfAttentionHead(n_embd, head_size, block_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.concat(self.heads(x), dim=-1)
        x = self.projection(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_heads: int, n_embd: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.head_size = n_embd // n_heads
        self.attention = MultiHeadAttention(n_heads, n_embd, self.head_size, block_size)

        # feedforward section
        # multiplier is recommended from attention is all you need paper
        # for inner matrix
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

        # Layernorm for training stability
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(nn.ln1(x))
        x = x + self.ff(self.ln2(x))
        x = self.relu(x)

        return x


class GPTModel(nn.Module):
    def __init__(
        self, vocab_size: int, n_blocks: int, n_heads: int, n_embd: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            DecoderBlock(n_embd=n_embd, n_heads=n_heads, block_size=block_size, dropout=dropout)
            for _ in range(n_blocks)
        )
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
