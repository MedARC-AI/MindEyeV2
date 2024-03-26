#!/bin/bash
#SBATCH --account=topfmri
#SBATCH --partition=a40x
#SBATCH --job-name=MEV1        
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=80 
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=fmri
#SBATCH --no-requeue

accelerate launch MEV1.py