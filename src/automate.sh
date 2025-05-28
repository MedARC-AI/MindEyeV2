# python mi_inference.py
# python mi_inference_enhanced.py
# for subj in 1 2 5 7; do
#     for gen_rep in 0 1 2 3 4 5 6 7 8 9; do
#         for mode in "imagery" "vision"; do
#             python recon_inference_mi.py \
#                 --model_name "final_subj0${subj}_pretrained_40sess_24bs" \
#                 --subj $subj \
#                 --mode $mode \
#                 --gen_rep $gen_rep \
#                 --cache_dir ../cache \
#                 --data_path ../dataset \
#                 --hidden_dim 4096 \
#                 --n_blocks 4 \
#                 --new_test

#             python enhanced_recon_inference_mi.py \
#                 --model_name "final_subj0${subj}_pretrained_40sess_24bs" \
#                 --subj $subj \
#                 --mode $mode \
#                 --gen_rep $gen_rep 
#         done
#     done
# done
export CUDA_VISIBLE_DEVICES="3"

for subj in 1 2 5 7; do
    for gen_rep in 0 1 2 3 4 5 6 7 8 9; do
        mode="imagery" 
        for trial_reps in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
            python recon_inference_mi.py \
                --model_name "final_subj0${subj}_pretrained_40sess_24bs" \
                --subj $subj \
                --mode $mode \
                --gen_rep $gen_rep \
                --trial_reps $trial_reps \
                --cache_dir /home/naxos2-raid25/kneel027/home/kneel027/MindEye_Imagery/cache \
                --data_path /home/naxos2-raid25/kneel027/home/kneel027/MindEye_Imagery/dataset \
                --hidden_dim 4096 \
                --n_blocks 4 \
                --new_test

            python enhanced_recon_inference_mi.py \
                --model_name "final_subj0${subj}_pretrained_40sess_24bs" \
                --subj $subj \
                --mode $mode \
                --gen_rep $gen_rep \
                --trial_reps $trial_reps
            done
        mode="vision" 
        for trial_reps in 1 2 3 4 5 6 7 8; do
            python recon_inference_mi.py \
                --model_name "final_subj0${subj}_pretrained_40sess_24bs" \
                --subj $subj \
                --mode $mode \
                --gen_rep $gen_rep \
                --trial_reps $trial_reps \
                --cache_dir /home/naxos2-raid25/kneel027/home/kneel027/MindEye_Imagery/cache \
                --data_path /home/naxos2-raid25/kneel027/home/kneel027/MindEye_Imagery/dataset \
                --hidden_dim 4096 \
                --n_blocks 4 \
                --new_test

            python enhanced_recon_inference_mi.py \
                --model_name "final_subj0${subj}_pretrained_40sess_24bs" \
                --subj $subj \
                --mode $mode \
                --gen_rep $gen_rep \
                --trial_reps $trial_reps
        done
    done
done