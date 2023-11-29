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
jupyter nbconvert Train.ipynb --to python

model_name="1gpu_l1loss_nopixcorr_MLPMixer_revised_linear"
echo model_name=${model_name}

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=128
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

export WANDB_MODE="online"
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

###########

cd /fsx/proj-fmri/YOURNAME/MindEyeV2/src

accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/fsx/proj-fmri/shared/mindeyev2_dataset --model_name=${model_name} --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.66 --num_epochs=120 --ckpt_interval=5 --no-use_image_aug --clip_scale=1. --blur_scale=100. --blurry_recon --ckpt_saving --hidden_dim=4096 --wandb_log