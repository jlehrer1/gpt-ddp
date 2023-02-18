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
    pytorch-lightning \
    wandb \
    sklearn \
    boto3 \ 
    tenacity \ 
    pandas \
    plotly \
    scipy \
    scikit-misc \
    datasets \
    transformers

COPY . .