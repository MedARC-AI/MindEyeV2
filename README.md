# MindEyeV2

In-progress -- this repo is under active development in the MedARC discord server (feel free to join us and help develop MindEyeV2!)

1. Download all of https://huggingface.co/datasets/pscotti/mindeyev2 and place them in a folder. You will need to specify the path to this folder as "data_path" variable.

```
from huggingface_hub import snapshot_download 
snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", cache_dir = "./cache" ,
    local_dir= "your_local_dir", local_dir_use_symlinks = False, resume_download = True)
```

2. Run setup.sh to install a new "fmri" conda environment.

3. Activate the conda environment with "conda activate fmri"

4. Run Train.ipynb

