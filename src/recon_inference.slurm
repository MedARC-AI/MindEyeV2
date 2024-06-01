#!/bin/bash
#SBATCH --account=topfmri
#SBATCH --partition=a80
#SBATCH --job-name=eyeinference
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     # should = number of gpus
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue

source ~/.bashrc
cd /weka/proj-fmri/paulscotti/MindEyeV2/src

jupyter nbconvert recon_inference.ipynb --to python
jupyter nbconvert enhanced_recon_inference.ipynb --to python

model_name="finetuned_subj01_40sess"

srun python recon_inference.py --model_name=${model_name} --subj=1 --n_blocks=4 --hidden_dim=1024 && python enhanced_recon_inference.py --model_name=${model_name} --subj=1