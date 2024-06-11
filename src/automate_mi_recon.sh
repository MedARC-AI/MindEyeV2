jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert enhanced_recon_inference_mi.ipynb --to python
export CUDA_VISIBLE_DEVICES="0"

for subj in 9; do
    # for gen_rep in 4 5 6 7 8 9; do
    #pretrained_subj0${subj}irf_40sess_hypatia_no_blurry_noirfpt_vision pretrained_subj0${subj}irf_40sess_hypatia_no_blurry_noirfpt_all
    for model in pretrained_subj09irf_40sess_hypatia_imageryrf_imagery_no_blurry pretrained_subj09irf_40sess_hypatia_imageryrf_vision_no_blurry; do
        for mode in "imagery" "vision"; do
            python recon_inference_mi.py \
                --model_name $model \
                --subj $subj \
                --mode $mode \
                --cache_dir ../cache \
                --data_path ../dataset \
                --hidden_dim 1024 \
                --n_blocks 4 \
                --no-blurry_recon

            python enhanced_recon_inference_mi.py \
                --model_name $model \
                --subj $subj \
                --mode $mode \
                --no-blurry_recon
         done
    done
done
