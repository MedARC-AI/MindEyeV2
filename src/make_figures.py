import matplotlib.pyplot as plt
import torch
import numpy as np
import PIL
import os
import utils

voxels, all_images = utils.load_nsd_mental_imagery(subject=1, mode="vision", stimtype="all", average=True, nest=False, data_root="../dataset/")
for mode in ["vision", "imagery"]:
    for model_name in ["final_subj01_pretrained_40sess_24bs", "final_subj01_pretrained_40sess_24bs", "pretrained_subj01_40sess_hypatia_no_blurry2", "pretrained_subj01_40sess_hypatia_imageryrf_vision_no_blurry2", "pretrained_subj01_40sess_hypatia_imageryrf_all_no_blurry2"]:
        all_enhancedrecons = torch.load(f"evals/{model_name}/{model_name}_all_enhancedrecons_{mode}.pt")


        fig, axs = plt.subplots(6, 6, figsize=(8, 8))
        bidx = 0
        for samp_list in [np.arange(0,6),np.arange(6,12),np.arange(12,18)]:
            for cidx, img_sample in enumerate(samp_list):
                gt_image = utils.torch_to_Image(all_images[img_sample])
                axs[bidx, cidx].imshow(gt_image)
                # axs[bidx, cidx].set_title(f"Ground Truth {img_sample+1}")
                axs[bidx, cidx].axis('off')  # Turn off axis

                reconstructed_image = utils.torch_to_Image(all_enhancedrecons[img_sample])
                axs[bidx+1, cidx].imshow(reconstructed_image)
                axs[bidx+1, cidx].axis('off')  # Turn off axis
            bidx += 2

        plt.subplots_adjust(top=0.9)  # Adjust the top parameter to create more space for the title
        plt.suptitle(f"{model_name} reconstructions for {mode} stimuli")
        plt.tight_layout()
        plt.show()
        print(f"saved evals/{model_name}/{model_name}_all_enhancedrecons_{mode}.png")
        plt.savefig(f"evals/{model_name}/{model_name}_all_enhancedrecons_{mode}.png")
