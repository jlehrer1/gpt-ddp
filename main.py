import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from gptlightning.data import AutoRegressiveTextSampler
from transformers import AutoTokenizer
from gptlightning.lightning_model import GPT
from torch.optim import Adam
from functools import partial
from pytorch_lightning.loggers import WandbLogger

context_length = 4
batch_size = 2
num_workers = 0

device = "gpu" if torch.cuda.is_available() else None

# Text file containing all text you want to train on
with open('training_data.txt') as f:
    traintext = f.read()

# Text file containing all validation data
with open('validation_data.txt') as f:
    valtext = f.read()

print('Splitting text')
traintext = traintext[0: 1000].split(' ')
valtext = valtext[0: 1000].split(' ')

print('Generating tokenizer')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

print('Setting up datasets')
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

print('Setting up model')
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
    logger=WandbLogger(
        name="GPT First Pass",
        project="Language Modeling"
    )
)

print('Beginning training phase')
trainer.fit(model, trainloader, valloader)