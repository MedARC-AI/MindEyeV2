from huggingface_hub import snapshot_download, hf_hub_download
# snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", allow_patterns="*.tar", cache_dir = "./cache",
#     local_dir= "dataset", local_dir_use_symlinks = False, resume_download = True)
hf_hub_download(repo_id="pscotti/mindeyev2", filename="coco_images_224_float16.hdf5", repo_type="dataset", local_dir= "dataset")
hf_hub_download(repo_id="pscotti/mindeyev2", filename="betas_all_subj01_fp32.hdf5", repo_type="dataset", local_dir= "dataset") # repeat for other subj if needed
hf_hub_download(repo_id="pscotti/mindeyev2", filename="betas_all_subj02_fp32.hdf5", repo_type="dataset", local_dir= "dataset") # repeat for other subj if needed
hf_hub_download(repo_id="pscotti/mindeyev2", filename="betas_all_subj05_fp32.hdf5", repo_type="dataset", local_dir= "dataset") # repeat for other subj if needed
hf_hub_download(repo_id="pscotti/mindeyev2", filename="betas_all_subj07_fp32.hdf5", repo_type="dataset", local_dir= "dataset") # repeat for other subj if needed