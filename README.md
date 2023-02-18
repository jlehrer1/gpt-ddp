# GPT-Lightning

A small library for training text GPT models (Decoder-block only models with implicit embedding) with Docker on Kubernetes clusters. 

Summary of the library:
`data.py` contains a torch dataset for generating samples for an autoregressive language model
`callbacks.py` contains a callback that generates sample text from the model and writes it to file, so we can see the progression of text generation over train time 
`lightning_model.py` contains the PyTorch-Lightning wrapper around the base GPT model for training 