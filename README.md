# MindEyeV2

In-progress -- this repo is under active development in the MedARC discord server (feel free to join us and help develop MindEyeV2!)

1. Download relevant parts of https://huggingface.co/datasets/pscotti/mindeyev2 and place them in a folder. You will need to specify the path to this folder as "data_path" variable.

```
from huggingface_hub import snapshot_download 
snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", allow_patterns="*.tar", cache_dir = "./cache",
    local_dir= "your_local_dir", local_dir_use_symlinks = False, resume_download = True)
hf_hub_download(repo_id="pscotti/mindeyev2", filename="coco_images_224_float16.hdf5", repo_type="dataset")
hf_hub_download(repo_id="pscotti/mindeyev2", filename="betas_all_subj01_fp32.hdf5", repo_type="dataset") # repeat for other subj if needed
```

2. Run setup.sh to install a new "fmri" virtual environment

3. Make sure the virtual environment is activated with "source fmri/bin/activate"

4. Run Train.ipynb