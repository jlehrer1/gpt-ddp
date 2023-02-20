# GPT Lightning

*A small library for training text GPT models via PyTorch Lightning.*

Contents:
`data.AutoRegressiveTextSampler`:  
- For handling data, the class is a torch dataset subclass for sampling text sequences from a list of encoded or non-encoded data. It produces pairs of text sequences, where the second is "shifted right" by one word.

- For example, the phrase "I walked to the grocery store to get bread" could be split with a context length of 8, into samples "I walked to the grocery store to get" and "walked to the grocery store to get bread". This allows us to calculate loss on correctly generating the next word. 

`metrics.Metrics`  
- A wrapper for calculating metrics across batches and epochs. Accepts a dictionary with names and `torchmetrics.Metric` subclasses, and handles updating internal states + generating dictionaries for logging, finding the best epoch for different metrics etc. 

`callbacks.SampleTextGenerationCallback`
- A PyTorch-Lightning callback for generating text samples from the model at intermediate times during training, so we can get a sense of if the text completion is good enough. Accepts an prompt to start the generation from, and parameters to control the frequency of text sampling

`callbacks.UploadCheckpointToS3`
- A PyTorch-Lightning callback for uploading model checkpoints to S3. Useful when training on remote clusters, and we want to save model checkpoints at intermediate times

`lightning_model.GPT`
- The LightningModule class. The wrapper around the "GPT" model, which in this case is simply a decoder-block only (i.e. self-attention across inputs, no cross-attention to any encoded data) transformer model with a MLP head.

To use:
Put all text you want to train on in `training_text.py`, and all text for validation in `validation_text.py`. Then run `main.py`.