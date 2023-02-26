import os
from typing import Collection, Union

import pytorch_lightning as pl
import torch
import transformers
from torch.utils.data import DataLoader, Dataset


class AutoRegressiveTextSampler(Dataset):
    def __init__(
        self,
        text: Union[Collection[str], Collection[int]],
        context_length: int,
        tokenizer: transformers.PreTrainedTokenizer = None,
        padding: int = 2,
    ):
        """Creates a dataset object for generating pairs of text
        for training autoregressive lanuage models. The __getitem__
        returns a pair of (data, label), where the data is a sequence of text
        (encoded) of length=sequence_length, and the label is the sequence of data
        starting at the same position, but shifted right by one word. This way, we
        can generate samples for our model to predict the next word.

        :param text: Either a list of strings, such as ['my', 'name', 'is'...]
        or an already tokenized list of strings, such as [15, 143, 1, ...]
        :type text: Union[list[str], list[int]]
        :param context_length: Integer representing the sequence length to generate
        :type context_length: int
        :param tokenizer: A class with a .encode method to convert a string into a list
        of integers
        :type tokenizer: transformers.PreTrainedTokenizer, optional
        :param padding: Number of extra words to tokenizer (since sometimes the tokenizer
        slightly compresses the length of the sequence, and we need the length
        of both sequences to be the same), defaults to 2
        :type padding: int, optional
        """

        assert len(text) > context_length, "Number of words must be greater than the context length"
        self.text = text
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.padding = padding

        if isinstance(text[0], str) and tokenizer is None:
            raise RuntimeError(
                """If providing list of text, such as ['my', 'name', 'is', ...] 
                a tokenizer object with .encode() must be provided to convert the text into sequences of integers."""
            )

    def __getitem__(self, idx):
        if self.tokenizer is not None:
            data = " ".join(self.text[idx : idx + self.context_length + self.padding])
            encoded = self.tokenizer.encode(data)
        else:
            data = self.text[idx : idx + self.context_length + self.padding]

        X = encoded[0 : self.context_length]
        Y = encoded[1 : self.context_length + 1]

        # a bit hacky, but saves us if tokenized input isn't long enough
        # we might repeated samples occasionally, it's alright
        if len(X) != self.context_length or len(Y) != self.context_length:
            return self(idx + 1)

        return torch.tensor(X), torch.tensor(Y)

    def __len__(self):
        return len(self.text) - self.context_length


class TextSequenceModule(pl.LightningDataModule):
    """Wrapper around generating trainloader/valloader
    This exists so we can use the auto batch size finder from Lightning
    which requires a datamodule with the batch_size parameter"""

    def __init__(
        self,
        train_dataset: AutoRegressiveTextSampler,
        val_dataset: AutoRegressiveTextSampler,
        batch_size: int,
        num_workers: int = None,
    ) -> None:
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
