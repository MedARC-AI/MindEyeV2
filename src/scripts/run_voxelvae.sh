#!/bin/bash
export DATA_DIR="/fsx/proj-fmri/shared/natural-scenes-dataset/webdataset_avg_split"
export MODEL_NAME="SDXL_VAE"
cd /fsx/proj-fmri/mihirneal/MindEyeV2/src
accelerate launch train_vox_vae.py \
    --data_path="$DATA_DIR" \
    --model_name="$MODEL_NAME" \
    --output_dir="/fsx/proj-fmri/mihirneal/MindEyeV2/src/train_logs/$MODEL_NAME" \
    --subj=1 \
    --train_batch_size=8 \
    --test_batch_size=8 \
    --mixup_pct=-1 \
    --num_epochs=120\
    --use_cont \
    --seed=0 \
    --report_to="wandb" 


