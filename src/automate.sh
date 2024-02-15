# python mi_inference.py
# python mi_inference_enhanced.py

for subj in 1 2 5 7; do
    for mode in "vision" "imagery"; do
        for gen_rep in 0 1 2 3 4 5 6 7 8 9; do
            python recon_inference_mi.py \
                --model_name "final_subj0${subj}_pretrained_40sess_24bs" \
                --subj $subj \
                --mode $mode \
                --gen_rep $gen_rep \
                --cache_dir ../cache \
                --data_path ../dataset \
                --hidden_dim 4096 \
                --n_blocks 4 \
                --new_test

            python enhanced_recon_inference_mi.py \
                --model_name "final_subj0${subj}_pretrained_40sess_24bs" \
                --subj $subj \
                --mode $mode \
                --gen_rep $gen_rep 
        done
    done
done