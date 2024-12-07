#!/bin/bash
#SBATCH --job-name=mindeye_prep
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgstabilityai-gpu-a100
#SBATCH --time=04:00:00   
#SBATCH -e slurms/%j.err    # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu

# # Full dataset takes ~200GB of disk space
# raw eeg data -> ~130GB
# images ->  
# preprocessed eeg data -> ~40GB

data_dir=/net/pr2/projects/plgrid/plggrai/kzrobek/MindEyeV2/data

# Download THINGSEEG2 raw data for all subjects 1-10
python download_data.py --data_path $data_dir

# Prep image path info for model training 
# this is not a resource heavy task
python image_paths.py --data_path $data_dir

# Preprocess eeg (see preprocessing methods in load_eeg.py)
python load_eeg.py $data_dir --sfreq 100 --sessionNumber 4




