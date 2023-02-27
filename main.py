import argparse
import gc
import os
from functools import partial

import boto3
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.optim import Adam
from transformers import AutoTokenizer

from gptddp import (AutoRegressiveTextSampler, DDPManager, GPTModel,
                    ModelTrainer, SampleTextGenerationCallback,
                    UploadCheckpointToS3, WandbMetricsCallback)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    # set up parser for command line args
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--name", default="GPT Model", type=str)

    # model hparams
    parser.add_argument("--context-length", default=32, type=int)
    parser.add_argument("--n-layers", default=4, type=int)
    parser.add_argument("--n-heads", default=2, type=int)
    parser.add_argument("--n-embd", default=32, type=int)

    # optimizer hparams
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--accumulate-batches", default=0, type=int)
    parser.add_argument("--warmup", default=4000, type=int)
    args = parser.parse_args()

    # dataloader params
    batch_size, num_workers, name = (
        args.batch_size,
        args.num_workers,
        args.name,
    )

    # model params
    context_length, n_layers, n_heads, n_embd = args.context_length, args.n_layers, args.n_heads, args.n_embd

    # optimizer params
    lr, accumulate_batches, warmup = args.lr, args.accumulate_batches, args.warmup

    # Text file containing all text you want to train on
    with open("training_data.txt") as f:
        traintext = f.read()

    # Text file containing all validation data
    with open("validation_data.txt") as f:
        valtext = f.read()

    print("Splitting text")
    traintext = traintext.split(" ")
    valtext = valtext.split(" ")

    with DDPManager():
        rank = int(os.environ["RANK"])
        num_devices = torch.cuda.device_count()

        print(f"Total number of CUDA devices is {num_devices}")
        print("Generating tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        print("Setting up model")
        model = GPTModel(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
            context_length=context_length,
            n_layers=n_layers,
            n_embd=n_embd,
            n_heads=n_heads,
            dropout=0.0,
        )

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

        print("Setting up model")
        # set up metrics
        metrics = WandbMetricsCallback(
            metrics={
                "perplexity": tm.Perplexity(),
            },
            phases=["train", "validation"],
            project="Language Modeling (multi-gpu)",
            name=f"{name}-heads-{n_heads}-blocks-{n_layers}-nembd-{n_embd}-accum-{accumulate_batches}-gpu-{num_devices}",
        )

        sample_text_generator = SampleTextGenerationCallback(
            prompt="It's a beautiful day for ",
            every_n_epochs=1,
            every_n_batches=10,
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
            desc=f"{name}-checkpoint-heads-{n_heads}-blocks-{n_layers}-nembd-{n_embd}",
            s3_resource=s3,
            bucket="braingeneersdev",
            upload_prefix="jlehrer/gpt_model/",
            n_epochs=1,
            n_steps=50000,
        )
        # fp16_scaler = torch.cuda.amp.GradScaler(enabled=True)
        trainer = ModelTrainer(
            model=model,
            traindata=traindata,
            valdata=valdata,
            optimizer=partial(Adam, lr=lr),
            lr_scheduler=None,
            criterion=nn.CrossEntropyLoss(),
            max_epochs=1,
            callbacks=[
                sample_text_generator,
                upload_callback,
                metrics,
            ],
            log_every_n_steps=50,
            limit_train_batches=None,
            limit_val_batches=1000,
            # scaler=fp16_scaler,
        )

        trainer.setup_dataloaders(batch_size=batch_size, num_workers=num_workers)
        trainer.run()
