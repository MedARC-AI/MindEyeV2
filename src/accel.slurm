#!/bin/bash
#SBATCH --account=topfmri
#SBATCH --partition=g40x
#SBATCH --job-name=ms
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=4     # should = number of gpus
#SBATCH --gres=gpu:4
#SBATCH --time=32:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --comment=topfmri

module load cuda/11.7 # should match torch.cuda.version

export NUM_GPUS=4 # Set to equal gres=gpu:#
export GLOBAL_BATCH_SIZE=512

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

export WANDB_DIR="/fsx/proj-fmri/paulscotti/MindEyeV2/wandb/"
export WANDB_CACHE_DIR="/fsx/home-paulscotti/.cache"
export WANDB_MODE="online"

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

###########

cd /fsx/proj-fmri/paulscotti/MindEyeV2
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT Train.py --data_path=/fsx/proj-fmri/shared/mindeyev2_dataset --model_name=test --subj=1 --batch_size=${GLOBAL_BATCH_SIZE} --n_samples_save=0 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=240 --ckpt_interval=999 --no-use_image_aug

# --wandb_log
