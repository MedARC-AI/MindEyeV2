export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="/fsx/proj-fmri/mihirneal/MEV2_exp/src/color_t2i/saved_ckpt/"
export HF_DATASETS_CACHE="/scratch/huggingface"

accelerate launch train_t2i_adapter_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir="/fsx/proj-fmri/mihirneal/controlNetData.py" \
    --mixed_precision="fp16" \
    --resolution=1024 \
    --tracker_project_name="T2I_COLOR" \
    --learning_rate=1e-5 \
    --max_train_steps=35000 \
    --validation_image "/fsx/proj-fmri/shared/controlNetData/target/img_t998.jpg" "/fsx/proj-fmri/shared/controlNetData/target/img_t376.jpg" "/fsx/proj-fmri/shared/controlNetData/target/img_t20.jpg" \
    --validation_prompt "A crystal bowl filled with oranges on top of a table." "An elephant and a rhinoceros stand not far from each other. " "A refrigerator, oven and microwave sitting in a kitchen." \
    --validation_steps=100 \
    --image_column="target" \
    --conditioning_image_column="target" \
    --caption_column="prompt" \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --report_to="wandb" \
    --seed=42 \
    --checkpointing_step=500 \
    --push_to_hub \
    --use_deepspeed \
    --resume_from_checkpoint latest 