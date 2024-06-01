# MindEye2

**Paper (accepted to ICML 2024): https://arxiv.org/abs/2403.11207**

![](figs/recon_comparison_small_alt.png)<br>

## Installation

1. Git clone this repository:

```
git clone https://github.com/MedARC-AI/MindEyeV2.git
cd MindEyeV2
```

2. Download https://huggingface.co/datasets/pscotti/mindeyev2 and place them in the same folder as your git clone.
Warning: **Cloning the entire huggingface dataset will be over 100 GB of data!**
The below code will download the subset of files required to run all our training / inference / evaluation code (does not download pretrained models).

```
import os
from huggingface_hub import list_repo_files, hf_hub_download

repo_id, branch, exclude_dirs, exclude_files = "pscotti/mindeyev2", "main", ["train_logs", "evals"], ["human_trials_mindeye2.ipynb", "subj01_annots.npy", "shared1000.npy"]

def download_files(repo_id, branch, exclude_dirs):
    files = list_repo_files(repo_id, repo_type="dataset", revision=branch)
    for file_path in files:
        if not any(ex_dir in file_path for ex_dir in exclude_dirs) and not any(ex_file in file_path for ex_file in exclude_files):
            local_path = os.path.join(repo_id.split("/")[1], file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            hf_hub_download(repo_id, filename=file_path, repo_type="dataset", revision=branch, local_dir=os.path.dirname(local_path))

download_files(repo_id, branch, exclude_dirs)
```

3. Run ```. src/setup.sh``` to install a new "fmri" virtual environment. Make sure the virtual environment is activated with "source fmri/bin/activate".

## Usage

- ```src/Train.ipynb``` trains models (single-subject or multi-subject depending on your config). Check the argparser arguments to specify how you want to train the model (e.g., ```--num_sessions=1``` to train with 1-hour of data).
    - Final models used in the paper were trained on an 8xA100 80GB node and will OOM on weaker compute. You can train the model on weaker compute with very minimal performance impact by changing certain model arguments: We recommend lowering hidden_dim to 1024 (or even 512), removing the low-level submodule (``--no-blurry_recon``), and lowering the batch size.
    - To train a single-subject model, set ```--no-multi_subject``` and ```--subj=#``` where # is the subject from NSD you wish to train
    - To train a multi-subject model (i.e., pretraining), set ```--multi_subject``` and ```--subj=#``` where # is the one subject out of 8 NSD subjects to **not** include in the pretraining.
    - To fine-tune from a multi-subject model, set ```--no-multi_subject``` and ```--multisubject_ckpt=path_to_your_pretrained_ckpt_folder```
- ```src/recon_inference.ipynb``` will run inference on a pretrained model, outputting tensors of reconstructions/predicted captions/etc.
- ```src/final_evaluations.ipynb``` will visualize reconstructions output from ```src/recon_inference``` and compute quantitative metrics.
- See .slurm files for example scripts for running the .ipynb notebooks as batch jobs submitted to Slurm job scheduling.

## FAQ

1. What are the main differences between this and MindEye1?

MindEye2 achieves SOTA reconstruction and retrieval performance compared to past work (including MindEye1). MindEye2 also excels in low-sample settings, with good performance even with just 1 hour of training data by first pretraining the model on other participants data. MindEye2 also releases a SOTA unCLIP model (unclip6_epoch0_step110000.ckpt) by fine-tuning SDXL that raises the possible ceiling performance possible for reconstructing images from CLIP image latents. MindEye2 training is also more flexible thanks to our updated webdataset approach that allows one to easily obtain the brain activations corresponding to a sample's previous/future timepoints, other timepoints from looking at the same image, and behavior (button press, reaction time, etc.). 

2. What are the "behav", "past_behav", "future_behav", "old_behav" arrays?

Our webdatasets only contain behavioral information; the brain activations and the seen images get loaded separately from hdf5 files and then indexed from these behav arrays accordingly. The webdataset tar files contain behav/past_behav/future_behav/old_behav matrices, although we only used "behav" for training MindEye2 (the other matrices can still be useful however, so we include them for you.)

Below is the lookup table for these arrays, with variables referenced from the Natural Scenes Dataset manual: https://cvnlab.slite.page/p/fRv4lz5V2F/Untitled

```
0 = COCO IDX (73K) (used to index coco_images_224_float16.hdf5)
1 = SUBJECT
2 = SESSION
3 = RUN
4 = TRIAL
5 = GLOBAL TRIAL (used to index betas_all_subj_fp32_renorm.hdf5)
6 = TIME
7 = ISOLD
8 = ISCORRECT
9 = RT
10 = CHANGEMIND
11 = ISOLDCURRENT
12 = ISCORRECTCURRENT
13 = TOTAL1
14 = TOTAL2
15 = BUTTON
16 = IS_SHARED1000
```

-1 values in these arrays should be interpreted as NaNs.

E.g., behav[0,:,9] corresponds to the 1st sample in the current batch's corresponding response time for the participant to press a button for that image.

past_behav gives you the behavioral information for samples corresponding the immediate previous timepoints samples.

future_behav gives you the behavioral information for samples corresponding to the immediate future timepoint samples.

old_behav gives you the behavioral information for the other repetitions of the given sample (remember to ignore -1s).

The code to create the above webdatasets and the hdf5 full of voxel brain activations can be found in src/dataset_creation.ipynb.

## Citation

If you make use of this work please cite the MindEye2 and MindEye1 papers and the Natural Scenes Dataset paper.

MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data

Scotti, Tripathy, Torrico, Kneeland, Chen, Narang, Santhirasegaran, Xu, Naselaris, Norman, & Abraham. MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data. International Conference on Machine Learning. (2024). arXiv:2403.11207  

MindEye1: Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors

Scotti, Banerjee, Goode, Shabalin, Nguyen, Cohen, Dempster, Verlinde, Yundler, Weisberg, Norman, & Abraham. Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors. Advances in Neural Information Processing Systems, 36. (2023). arXiv:2305.18274. 

Natural Scenes Dataset: A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence

Allen, St-Yves, Wu, Breedlove, Prince, Dowdle, Nau, Caron, Pestilli, Charest, Hutchinson, Naselaris, & Kay. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).
