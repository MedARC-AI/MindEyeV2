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
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf

sample_dir = "/weka/proj-fmri/shared/coco/sampled_imgs"
sampled_images = os.listdir(sample_dir)
data_path="/weka/proj-fmri/shared/mindeyev2_dataset"
data_type = torch.float16
device = torch.device("cuda")

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)

f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'][:1]
images = torch.Tensor(images).to("cpu").to(data_type)

config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38

diffusion_engine = DiffusionEngine(network_config=network_config,
                               denoiser_config=denoiser_config,
                               first_stage_config=first_stage_config,
                               conditioner_config=conditioner_config,
                               sampler_config=sampler_config,
                               scale_factor=scale_factor,
                               disable_first_stage_autocast=disable_first_stage_autocast)
        # set to inference
diffusion_engine.to(device)
diffusion_engine.eval()

ckpt_path = '/weka/proj-fmri/shared/cache/sdxl_unclip/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])

image = images[:1].to(device)
batch={"jpg": image,
              "original_size_as_tuple": torch.ones(image.shape[0], 2).to(device) * image.shape[-1],
              "crop_coords_top_left": torch.zeros(image.shape[0], 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
print("vector_suffix", vector_suffix.shape)


unclip_savedir = "/weka/proj-fmri/shared/coco/xlunclip_imgs"
sam_img = os.listdir(sample_dir)
def convert_and_normalize(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    return img_tensor



for i in tqdm(range(len(sam_img))):
    if len(os.listdir(unclip_savedir)) == 30000:
        print("Sampled images already saved")
        break
    gen_path = os.path.join(unclip_savedir, sam_img[i])
    if os.path.exists(gen_path):
        print(f"{sam_img[i]} already saved")
        continue
        
    img_path = os.path.join(sample_dir, sam_img[i])
    proc_img = convert_and_normalize(img_path)
    clip_img = clip_img_embedder(proc_img.unsqueeze(0).cuda())
    samples = utils.unclip_recon(clip_img,
                             diffusion_engine,
                             vector_suffix)
    img_unclip = transforms.ToPILImage()(samples[0])
    img_unclip.save(gen_path)

