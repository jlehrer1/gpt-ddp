import argparse
from functools import partial

import boto3
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from gptlightning import (GPT, AutoRegressiveTextSampler,
                          SampleTextGenerationCallback, UploadCheckpointToS3)

# set up parser for command line args
parser = argparse.ArgumentParser()

parser.add_argument("--context-length", default=64, type=int)
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--name", default="GPT Model", type=str)

args = parser.parse_args()
context_length, batch_size, num_workers, name = (
    args.context_length,
    args.batch_size,
    args.num_workers,
    args.name,
)

device = "gpu" if torch.cuda.is_available() else None

# Text file containing all text you want to train on
with open("training_data.txt") as f:
    traintext = f.read()

# Text file containing all validation data
with open("validation_data.txt") as f:
    valtext = f.read()

print("Splitting text")
traintext = traintext.split(" ")
valtext = valtext.split(" ")

print("Generating tokenizer")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("Setting up datasets")
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

trainloader = DataLoader(
    traindata,
    batch_size=batch_size,
    num_workers=num_workers,
)
valloader = DataLoader(
    valdata,
    batch_size=batch_size,
    num_workers=num_workers,
)

print("Setting up model")
model = GPT(
    optimizer=partial(
        Adam,
        lr=3e-4,
    ),
    vocab_size=tokenizer.vocab_size,
    tokenizer=tokenizer,
    context_length=context_length,
)

# set up callbacks
sample_text_generator = SampleTextGenerationCallback(
    prompt="Breaking news, Barack Obama has",
    every_n_epochs=1,
    log_wandb=True,
    new_tokens=100,
)

with open("credentials") as f:
    key, access = [line.rstrip() for line in f.readlines()]

s3 = boto3.resource(
    "s3",
    endpoint_url="https://s3-west.nrp-nautilus.io/",
    aws_access_key_id=key,
    aws_secret_access_key=access,
)

upload_callback = UploadCheckpointToS3(
    path="./checkpoints",
    desc=f"{name}-checkpoint",
    s3_resource=s3,
    bucket="braingeneersdev",
    upload_prefix="jlehrer/gpt_model/",
    n_epochs=1,
)

trainer = pl.Trainer(
    accelerator=device,
    devices=1 if device == "gpu" else None,
    max_epochs=500,
    logger=WandbLogger(name=name, project="Language Modeling"),
    callbacks=[
        sample_text_generator,
        upload_callback,
    ],
)

print("Beginning training phase")
trainer.fit(model, trainloader, valloader)
