#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yaml file for "conda env export > environment.yaml" after running this.

set -e

conda create -n fmri python=3.10.8 -y
conda activate fmri

conda install numpy matplotlib tqdm scikit-image jupyterlab -y

pip install accelerate webdataset clip pandas matplotlib ftfy regex kornia umap-learn h5py
pip install torchvision==0.15.2 torch==2.0.1
pip install diffusers
pip install deepspeed
