import torch
from pycocotools.coco import COCO
import numpy as np
import shutil
from tqdm import tqdm
from models import Clipper
import os
import random
import sys
import h5py
from torchvision import transforms
from PIL import Image
import utils

sample_dir = "/weka/proj-fmri/shared/coco/sampled_imgs"
sampled_images = os.listdir(sample_dir)
for i in tqdm(range(len(sampled_images))):
    if len(os.listdir(sample_dir)) == 30000:
        print("already sampled")
        break
    img_path = os.path.join(directory_path, sampled_images[i])
    sam_img_path = os.path.join(sample_dir, f"img{i}.jpg")
    shutil.copy2(img_path, sam_img_path)


from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
device = torch.device("cuda")
cache_dir = "/weka/proj-fmri/shared/cache/vd"
try:
    vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained("/weka/proj-fmri/shared/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7/").to(device).to(torch.float16)
except:
    print("Downloading Versatile Diffusion to")
    vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
            "shi-labs/versatile-diffusion", cache_dir=cache_dir).to(device).to(torch.float16)
    
generator = torch.Generator(device="cuda").manual_seed(42)
vd_pipe.image_unet.eval()
vd_pipe.vae.eval()
vd_pipe.image_unet.requires_grad_(False)
vd_pipe.vae.requires_grad_(False)

vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("shi-labs/versatile-diffusion", subfolder="scheduler", cache_dir=cache_dir)
num_inference_steps = 20

# Set weighting of Dual-Guidance 
text_image_ratio = .0 # .5 means equally weight text and image, 0 means use only image


vd_savedir = "/weka/proj-fmri/shared/coco/vd_imgs"
sam_img = os.listdir(sample_dir)
vd_pipe.set_progress_bar_config(disable=True)
for i in tqdm(range(len(sam_img))):
    if len(os.listdir(vd_savedir)) == 30000:
        print("Sampled images already saved")
        break
    gen_img_path = os.path.join(vd_savedir, sam_img[i])
    if os.path.exists(gen_img_path):
        print(f"{sam_img[i]} already saved")
        continue
    img = Image.open(os.path.join(sample_dir, sam_img[i]))
    vd_img = vd_pipe(image=img, prompt="Testing here, very green prominent", text_to_image_strength=text_image_ratio, generator=generator, disable_tqdm=True).images[0]
    vd_img.save(os.path.join(gen_img_path))
