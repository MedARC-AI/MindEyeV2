#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3.11 -m venv fmri
source fmri/bin/activate

pip install numpy matplotlib jupyter jupyterlab_nvdashboard jupyterlab tqdm scikit-image accelerate webdataset pandas matplotlib einops ftfy regex kornia h5py open_clip_torch torchvision torch==2.2.0 transformers xformers torchmetrics diffusers==0.23.0 deepspeed wandb nilearn nibabel omegaconf pytorch-lightning==2.0.1 
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install dalle2-pytorch
