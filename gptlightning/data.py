from torch.utils.data import Dataset
import transformers
import numpy as np
import torch


class AutoRegressiveTextSampler(Dataset):
    def __init__(
        self,
        text: list[str],
        context_length: int,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        self.text = text
        self.context_length = context_length
        self.tokenizer = tokenizer

        self.num_samples = len(text)

    def __getitem__(self, idx):
        rand_idx = np.random.randint(self.num_samples - self.context_length)
        r = rand_idx + self.context_length

        data = "".join(self.text[rand_idx : r + 2])
        encoded = self.tokenizer(data)["input_ids"][: self.context_length + 1]

        X = encoded[0 : self.context_length]
        Y = encoded[1 : self.context_length + 1]

        return torch.tensor(X), torch.tensor(Y)

    def __len__(self):
        return len(self.text)
