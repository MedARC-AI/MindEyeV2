#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=g40x
#SBATCH --job-name=retrieval
#SBATCH --nodes=1
#SBATCH --exclusive              
#SBATCH --ntasks-per-node=8   # should = number of gpus
#SBATCH --time=104:00:00          # total run time limit (HH:MM:SS)
#SBATCH --comment=fmri
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out

# Set to equal gres=gpu:#
export NUM_GPUS=8
# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
cd /fsx/proj-fmri/mihirneal/MindEyeV2/src

sh scripts/run_voxelmlp.sh