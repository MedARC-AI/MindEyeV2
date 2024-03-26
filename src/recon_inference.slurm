#!/bin/bash
#SBATCH --account=topfmri
#SBATCH --partition=a40x
#SBATCH --job-name=re_MEv2
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=fmri
#SBATCH --no-requeue


model_name="re_MindEye2"
ckpt_path="/weka/proj-fmri/mihirneal/MindEyeV2/src/ablation_ckpt/re_MindEye2"

srun python recon_inference.py --model_name=${model_name} --subj=1 --n_blocks=4 --hidden_dim=1024 --ckpt_path=${ckpt_path} && python enhanced_recon_inference.py --model_name=${model_name} --subj=1
