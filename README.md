# MindEye2

## Installation

1. Git clone this repository:

```
git clone https://github.com/MedARC-AI/MindEyeV2.git
```

2. Download https://huggingface.co/datasets/pscotti/mindeyev2 and place them in the same folder as your git clone.
Warning: **This will download over 120 GB of data!** You may want to only download some parts of the huggingface dataset (e.g., not all the pretrained models contained in "train_logs")

```
cd MindEyeV2
git clone https://huggingface.co/datasets/pscotti/mindeyev2 .
```

or for specifically downloading only parts of the dataset (will need to edit depending on what you want to download):
```
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", allow_patterns="*.tar",
    local_dir= "your_local_dir", local_dir_use_symlinks = False, resume_download = True)
hf_hub_download(repo_id="pscotti/mindeyev2", filename="coco_images_224_float16.hdf5", repo_type="dataset")
```

3. Run setup.sh to install a new "fmri" virtual environment. Make sure the virtual environment is activated with "source fmri/bin/activate".

## Usage

- ```src/Train.ipynb``` to trains models (both single-subject and multi-subject). Check the argparser arguments to specify how you want to train the model.
    - Final models used in the paper were trained on an 8xA100 80GB node and will OOM on weaker compute. You can train the model on weaker compute with minimal performance impact by changing certain model arguments: We recommend lowering hidden_dim to 1024 (or even 512), removing the low-level submodule (``--no-blurry_recons``), and lowering the batch size.
    - To train a single-subject model, set ```--no-multi_subject```
    - To train a multi-subject model (i.e., pretraining), set ```--multi_subject``` and set ```--subj=#``` where # is the one subject out of 8 NSD subjects to **not** include in the pretraining.
    - To fine-tune from a multi-subject model, set ```--no-multi_subject``` and set ```--multisubject_ckpt=path_to_your_pretrained_ckpt_folder```
- ```src/recon_inference.ipynb``` will run inference on a pretrained model, outputting tensors of reconstructions/predicted captions/etc.
- ```src/final_evaluations.ipynb``` will visualize reconstructions output from ```src/recon_inference`` and compute quantitative metrics.
