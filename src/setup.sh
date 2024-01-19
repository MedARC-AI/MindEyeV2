#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3.11 -m venv fmri2
source fmri2/bin/activate

pip install numpy matplotlib tqdm scikit-image jupyterlab accelerate webdataset clip pandas matplotlib ftfy regex kornia umap-learn h5py torchvision torch==2.0.1 diffusers deepspeed omegaconf pytorch-lightning==2.0.1

pip install git+https://github.com/openai/CLIP.git --no-deps

pip install dalle2-pytorch && pip install open-clip-torch && pip install einops && pip install transformers