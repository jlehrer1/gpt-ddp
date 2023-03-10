FROM anibali/pytorch:1.10.2-cuda11.3
USER root

WORKDIR /src

RUN sudo apt-get update
RUN sudo apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get --allow-releaseinfo-change update && \
    sudo apt-get install -y --no-install-recommends \
    curl \
    sudo \
    vim

RUN curl -L https://bit.ly/glances | /bin/bash

RUN pip install 'matplotlib<3.7' \
    seaborn \
    wandb \
    sklearn \
    boto3 \ 
    tenacity \ 
    pandas \
    plotly \
    scipy \
    torchmetrics \
    scikit-misc \
    datasets \
    transformers

COPY 'training_data.txt' .
COPY 'validation_data.txt' .
COPY main.py .
COPY gptddp gptddp

# copy credential things
COPY wandbcreds .
COPY credentials .