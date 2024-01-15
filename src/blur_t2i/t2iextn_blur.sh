export OUTPUT_DIR="/weka/proj-fmri/mihirneal/MindEyeV2/src/blur_t2i/saved_ckpt/"
export HF_DATASETS_CACHE="/scratch/huggingface"

accelerate launch train_t2i_adapter_sdxl.py \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir="/weka/proj-fmri/mihirneal/MindEyeV2/src/controlNetData.py" \
    --mixed_precision="fp16" \
    --resolution=768 \
    --tracker_project_name="t2iextn_blur" \
    --learning_rate=1e-5 \
    --max_train_steps=35000 \
    --validation_image "/weka/proj-fmri/shared/controlNetData/target/img_t998.jpg" "/weka/proj-fmri/shared/controlNetData/target/img_t376.jpg" "/weka/proj-fmri/shared/controlNetData/target/img_t20.jpg" \
    --validation_steps=2 \
    --image_column="target" \
    --conditioning_image_column="source" \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --report_to="wandb" \
    --seed=42 \
    --checkpointing_step=5 \
    --use_deepspeed \
    --resume_from_checkpoint latest 