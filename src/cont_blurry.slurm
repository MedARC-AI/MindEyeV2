#!/bin/bash
#SBATCH --account=topfmri
#SBATCH --partition=a40x
#SBATCH --job-name=cont_blur      
#SBATCH --gres=gpu:8
#SBATCH --time=80:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=80 
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=fmri
#SBATCH --no-requeue

accelerate launch Train_no_use_prior.py