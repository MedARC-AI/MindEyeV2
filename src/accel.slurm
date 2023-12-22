#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=g40x
#SBATCH --job-name=mindeye
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     # should = number of gpus
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc

source ~/.bashrc
cd /fsx/proj-fmri/YOURNAME/MindEyeV2/src
jupyter nbconvert Train_with_unCLIP.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=128
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

model_name="model_name"
echo model_name=${model_name}

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train_with_unCLIP.py --data_path=/fsx/proj-fmri/shared/mindeyev2_dataset --model_name=${model_name} --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.66 --num_epochs=120 --use_prior --prior_mult=30 --clip_scale=1. --blur_scale=100. --depth_scale=100. --no-blurry_recon --no-depth_recon --no-use_image_aug --ckpt_interval=10 --ckpt_saving --hidden_dim=1024 --seq_len=1 --subj_list=1 --num_sessions=37 --wandb_log