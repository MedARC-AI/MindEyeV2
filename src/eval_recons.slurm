#!/bin/bash
#SBATCH --account=topfmri
#SBATCH --partition=a80
#SBATCH --job-name=eye
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

srun python recon_inference.py --model_name=git_tong_unCLIP_150ep_4block_hid4096_bs40 --subj=1 --n_blocks=4 --hidden_dim=4096

srun python enhanced_recon_inference.py --model_name=git_tong_unCLIP_150ep_4block_hid4096_bs40 --subj=1
