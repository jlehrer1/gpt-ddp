# GPT Lightning

*A small library for training transformer models on text data via PyTorch Lightning.*

To install, run
`pip install git+https://github.com/jlehrer1/lightning-gpt`

Library Details:  
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

To use the provided `main.py`:
Put all text you want to train on in `training_text.py`, and all text for validation in `validation_text.py`. Then run `main.py`.

Model initialization scheme:
As described in the README for [minGPT initialization scheme](https://github.com/karpathy/minGPT/blob/master/README.md) which are sourced from the GPT-1 and GPT-2 papers and repos, we initialize all standard linear layers + embedding with a normal dist. with mean 0 and std 0.02. All bias vectors are initialized to 0. For the forward pass in the `DecoderBlock`, we initialize the weights of the projection Linear layer in the `MultiHeadedAttention` module to be `1/sqrt(num layers)` at initialization -- effectively scaling down the attention outputs and having MLP part of the decoder contribute most during early training. 