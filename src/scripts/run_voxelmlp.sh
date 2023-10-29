#!/bin/bash
export DATA_DIR="/fsx/proj-fmri/shared/mindeyev2_dataset"
export MODEL_NAME="clip"
cd /fsx/proj-fmri/mihirneal/MindEyeV2/src
accelerate launch train_voxelmlp.py \
    --data_path="$DATA_DIR" \
    --model_name="$MODEL_NAME" \
    --output_dir="/fsx/proj-fmri/mihirneal/MindEyeV2/src/train_logs/$MODEL_NAME" \
    --subj=1 \
    --batch_size=128 \
    --mixup_pct=.66 \
    --num_epochs=24 \
    --clip_seq_dim=257 \
    --clip_emb_dim=768 \
    --ckpt_interval=999 \
    --blurry_recon \
    --report_to="wandb" \
    --use_deepspeed
