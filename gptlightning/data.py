import numpy as np
import torch
import transformers
from torch.utils.data import Dataset


class AutoRegressiveTextSampler(Dataset):
    def __init__(
        self, 
        text: list[str], 
        context_length: int,
        tokenizer: transformers.PreTrainedTokenizer,
        padding: int = 2,
    ):
        self.text = text
        self.context_length = context_length 
        self.tokenizer = tokenizer 
        self.padding = padding

    def __getitem__(self, idx):
        data = " ".join(self.text[idx : idx + self.context_length + self.padding])
        encoded = self.tokenizer.encode(data)

        X = encoded[0 : self.context_length]
        Y = encoded[1 : self.context_length + 1]
        
        assert len(X) == len(Y)
        return torch.tensor(X), torch.tensor(Y)

    def __len__(self):
        return len(self.text) - self.context_length
