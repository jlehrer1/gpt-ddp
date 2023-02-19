import os

import torch
from pytorch_lightning.callbacks import Callback
import wandb
import pytorch_lightning as pl

class SampleTextGenerationCallback(Callback):
    def __init__(
        self,
        context_length: int,
        write_path: str = "./sample_output",
        every_n_epochs: int = 4,
        prompt: str = None,
        new_tokens: int = 1000,
        log_wandb: bool = False,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.write_path = write_path
        self.every_n_epochs = every_n_epochs
        self.prompt = prompt
        self.new_tokens = new_tokens
        self.log_wandb = log_wandb
        
        os.makedirs(write_path, exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        curr_epoch = pl_module.current_epoch

        if curr_epoch % self.every_n_epochs == 0:
            # generate writing sample just from "empty" prompt
            prompt = torch.zeros((1, 1), dtype=torch.long) if self.prompt is not None else self.prompt
            text = pl_module.base_model.generate(
                prompt,
                max_new_tokens=self.new_tokens,
                context_length=self.context_length,
            )
            # just for writing purposes
            text = text.split(' ')

            with open(os.path.join(self.write_path, f"{curr_epoch}_sample_output.txt"), "w") as f:
                for idx, word in enumerate(text):
                    # write a newline every ten words to make the output 
                    # more human readable
                    if idx % 10 == 0:
                        f.write(f"{word} \n")
                    else:
                        f.write(f"{word} ")

            if self.log_wandb:
                table = wandb.Table(columns=["epoch", "text"])
                table.add_data(curr_epoch, ' '.join(text[0: 100]))
                wandb.log({"Text Generation (No Prompt)": table})