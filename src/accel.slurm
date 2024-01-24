#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=a40x
#SBATCH --job-name=mindeye23
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     # should = number of gpus
#SBATCH --gres=gpu:1
#SBATCH --time=31:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=fmri
#SBATCH --no-requeue

source ~/.bashrc
cd /weka/proj-fmri/conscioustahoe/MindEyeV2/src
jupyter nbconvert Train.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=16
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

model_name="mindeye_23"
echo model_name=${model_name}

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-fmri/shared/mindeyev2_dataset --model_name=mindeye_23 --no-multi_subject --subj=1 --batch_size=16 --num_sessions=10 --hidden_dim=1024 --clip_scale=1. --blurry_recon --blur_scale=.5  --seq_past=0 --seq_future=0 --use_prior --prior_scale=30 --no-visualize_prior --n_blocks=4 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=50 --no-use_image_aug --ckpt_interval=1 --ckpt_saving --no-wandb_log