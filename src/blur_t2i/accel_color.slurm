#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=g40x
#SBATCH --job-name=t2i_color
#SBATCH --nodes=2
#SBATCH --exclusive              
#SBATCH --ntasks-per-node=8    # should = number of gpus
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

export WANDB_DIR="/fsx/proj-fmri/mihirneal/MEV2_exp/src/wandb/"
export WANDB_CACHE_DIR="/admin/home-mihirneal/.cache"
export WANDB_MODE="online"
export HF_DATASETS_CACHE="/scratch/huggingface"

H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo RANK = $RANK
echo WORLD_SIZE=${COUNT_NODE}

source /admin/home-mihirneal/.bashrc

sh t2i_color_train.sh