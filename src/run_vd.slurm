#!/bin/bash
#SBATCH --account=topfmri
#SBATCH --partition=a40x
#SBATCH --job-name=gen_vd
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     # should = number of gpus
#SBATCH --gres=gpu:1
#SBATCH --time=25:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=fmri
#SBATCH --no-requeue

cd /weka/proj-fmri/mihirneal/MindEyeV2/src
python gen_vd.py