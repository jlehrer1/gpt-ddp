import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .data import AutoRegressiveTextSampler
from transformers import AutoTokenizer
from .lightning_model import GPT
from torch.optim import Adam
from functools import partial

context_length = 32
batch_size = 4
num_workers = 0

device = "gpu" if torch.cuda.is_available() else None

# Text file containing all text you want to train on
with open('training_data.txt') as f:
    traintext = f.read()

# Text file containing all validation data
with open('validation_data.txt') as f:
    valtext = f.read()

traintext = traintext.split(' ')
valtext = valtext.split(' ')

tokenizer = AutoTokenizer.from_pretrained('gpt2')

traindata = AutoRegressiveTextSampler(
    text=traintext,
    context_length=context_length,
    tokenizer=tokenizer,
)

valdata = AutoRegressiveTextSampler(
    text=valtext,
    context_length=context_length,
    tokenizer=tokenizer,
)

trainloader = DataLoader(traindata, batch_size=batch_size, num_workers=num_workers)
valloader = DataLoader(valdata, batch_size=batch_size, num_workers=num_workers)

model = GPT(
    optimizer=partial(
        Adam, lr=3e-4,
    ),
    vocab_size=tokenizer.vocab_size,
)

trainer = pl.Trainer(
    accelerator=device,
    devices=1 if device == "gpu" else None,
    max_epochs=10,
)

trainer.fit(trainloader, valloader)