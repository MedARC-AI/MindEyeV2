
jupyter nbconvert Train.ipynb --to python
export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=10 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES="0 "

# singlesubject finetuning
# model_name="pretrained_subj09irf_40sess_hypatia_no_blurry_noirfpt_all"
# echo model_name=${model_name}
# python Train.py --data_path=../dataset --cache_dir=../cache --model_name=${model_name} --no-multi_subject --subj=9 --batch_size=${BATCH_SIZE} --max_lr=6e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/multisubject_subj01_hypatia_no_blurry2 --mode=all --no-blurry_recon #--resume_from_ckpt

pretrain_model_name="multisubject_subj10irf_hypatia_imageryrf_all_no_blurry"
echo model_name=${pretrain_model_name}
python Train.py --data_path=../dataset --cache_dir=../cache --model_name=${pretrain_model_name} --multi_subject --subj=10 --batch_size=${BATCH_SIZE} --max_lr=6e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --train_imageryrf --mode=all --no-blurry_recon

# singlesubject finetuning
model_name="pretrained_subj10irf_40sess_hypatia_imageryrf_all_no_blurry"
echo model_name=${model_name}
python Train.py --data_path=../dataset --cache_dir=../cache --model_name=${model_name} --no-multi_subject --subj=10 --batch_size=${BATCH_SIZE} --max_lr=6e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/${pretrain_model_name} --mode=all --no-blurry_recon #--resume_from_ckpt

pretrain_model_name="multisubject_subj11irf_hypatia_imageryrf_all_no_blurry"
echo model_name=${pretrain_model_name}
python Train.py --data_path=../dataset --cache_dir=../cache --model_name=${pretrain_model_name} --multi_subject --subj=11 --batch_size=${BATCH_SIZE} --max_lr=6e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --train_imageryrf --mode=all --no-blurry_recon

# singlesubject finetuning
model_name="pretrained_subj11irf_40sess_hypatia_imageryrf_all_no_blurry"
echo model_name=${model_name}
python Train.py --data_path=../dataset --cache_dir=../cache --model_name=${model_name} --no-multi_subject --subj=11 --batch_size=${BATCH_SIZE} --max_lr=6e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/${pretrain_model_name} --mode=all --no-blurry_recon #--resume_from_ckpt