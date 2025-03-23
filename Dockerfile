FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update -qq && apt-get upgrade -qq &&\
    apt-get install -qq man wget sudo vim tmux
RUN apt update
RUN apt install -y cudnn9

# Upgrade pip
RUN yes | pip install --upgrade pip

# Set environment variable to disable oneDNN custom operations (from agentic_ai.py)
ENV TF_ENABLE_ONEDNN_OPTS=0

# Copy requirements file first
COPY requirements.txt /tmp/

# Modify requirements file to use CUDA 12.4 instead of 12.1 to match base image
RUN sed -i 's/cu121/cu124/g' /tmp/requirements.txt

# Install from requirements file (this handles extra index URLs correctly)
RUN pip install -r /tmp/requirements.txt

RUN pip install tf-keras

# Clean up pip cache
RUN pip cache purge

# Create directories
RUN mkdir -p /home/data /home/output /home/models

# Copy Python files and configuration
COPY *.py /home/
COPY config.yaml /home/
COPY prompts.txt /home/

# Make sure data directory exists before copying
COPY data/chunks_selected.pkl /home/data/chunks_selected.pkl
COPY data/embeddings_selected.npy /home/data/embeddings_selected.npy

# Copy finetuned model
COPY models /home/models

# Set working directory
WORKDIR /home