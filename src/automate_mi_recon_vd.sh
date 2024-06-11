jupyter nbconvert recon_inference_mi_vd.ipynb --to python
export CUDA_VISIBLE_DEVICES="1"

# for subj in 1; do
#     # for gen_rep in 4 5 6 7 8 9; do
#     for model in final_subj0${subj}_pretrained_40sess_24bs; do
#         for mode in "imagery" "vision"; do
#             python recon_inference_mi_vd.py \
#                 --model_name $model \
#                 --subj $subj \
#                 --mode $mode \
#                 --cache_dir ../cache \
#                 --data_path ../dataset \
#                 --hidden_dim 4096 \
#                 --n_blocks 4 \
#          done
#     done
# done

for subj in 1; do
    # for gen_rep in 4 5 6 7 8 9; do
    #pretrained_subj0${subj}irf_40sess_hypatia_no_blurry_noirfpt_vision pretrained_subj0${subj}irf_40sess_hypatia_no_blurry_noirfpt_all
    for model in pretrained_subj0${subj}_40sess_hypatia_vd; do
        for mode in "imagery" "vision"; do
            python recon_inference_mi_vd.py \
                --model_name $model \
                --subj $subj \
                --mode $mode \
                --cache_dir ../cache \
                --data_path ../dataset \
                --hidden_dim 1024 \
                --n_blocks 4

         done
    done
done
