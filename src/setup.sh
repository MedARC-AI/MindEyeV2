#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3 -m venv mindeyev2
source mindeyev2/bin/activate

pip install numpy matplotlib tqdm scikit-image jupyterlab accelerate transformers xformers webdataset einops clip pandas matplotlib ftfy regex kornia umap-learn h5py torchvision torch==2.0.1 diffusers deepspeed omegaconf pytorch-lightning==2.0.1 dalle2_pytorch

pip install git+https://github.com/openai/CLIP.git --no-deps

pip install git+https://github.com/mlfoundations/open_clip.git --no-deps


