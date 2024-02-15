# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import json
import pickle
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from random import randrange
# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from models import *

### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:",device)

# %%
def load_nsd_mental_imagery(vector, subject, mode, stimtype="all", average=False, nest=False):
    data_path = "/export/raid1/home/kneel027/Second-Sight/data/"
    img_stim_file = data_path + "nsddata_stimuli/stimuli/nsd/nsdimagery_stimuli.pkl3"
    imagery_data_path = "/export/raid1/home/tsaharoy/NSD_Imagery/everything_NSD_imagery/"
    ex_file = open(img_stim_file, 'rb')
    imagery_dict = pickle.load(ex_file)
    ex_file.close()
    exps = imagery_dict['exps']
    cues = imagery_dict['cues']
    image_map  = imagery_dict['image_map']
    image_data = imagery_dict['image_data']
    cond_idx = {
    'visionsimple': np.arange(len(exps))[exps=='visA'],
    'visioncomplex': np.arange(len(exps))[exps=='visB'],
    'visionconcepts': np.arange(len(exps))[exps=='visC'],
    'visionall': np.arange(len(exps))[np.logical_or(np.logical_or(exps=='visA', exps=='visB'), exps=='visC')],
    'imagerysimple': np.arange(len(exps))[np.logical_or(exps=='imgA_1', exps=='imgA_2')],
    'imagerycomplex': np.arange(len(exps))[np.logical_or(exps=='imgB_1', exps=='imgB_2')],
    'imageryconcepts': np.arange(len(exps))[np.logical_or(exps=='imgC_1', exps=='imgC_2')],
    'imageryall': np.arange(len(exps))[np.logical_or(
                                        np.logical_or(
                                            np.logical_or(exps=='imgA_1', exps=='imgA_2'), 
                                            np.logical_or(exps=='imgB_1', exps=='imgB_2')), 
                                        np.logical_or(exps=='imgC_1', exps=='imgC_2'))]}
    
    x = torch.load(f"{data_path}preprocessed_data/subject{subject}/nsd_imagery.pt").requires_grad_(False).to("cpu")
    cond_im_idx = {n: [image_map[c] for c in cues[idx]] for n,idx in cond_idx.items()}
    y = torch.load(f"{data_path}preprocessed_data/{vector}_18.pt").requires_grad_(False).to("cpu")
    # Prune down to specific experimental mode/stimuli type
    x = x[cond_idx[mode+stimtype]]
    conditionals = cond_im_idx[mode+stimtype]
    if stimtype == "simple":
        y = y[:6]
    elif stimtype == "complex":
        y = y[6:12]
    elif stimtype == "concepts":
        y = y[12:]
    
    # trial_count = int(x.shape[0]/sample_count)
    # Average across trials
    if average or nest:
        x, y, sample_count = condition_average(x, y, conditionals, nest=nest)
    else:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
    print(x.shape)
    return x, y

def condition_average(x, y, cond, nest=False):
    idx, idx_count = np.unique(cond, return_counts=True)
    idx_list = [np.array(cond)==i for i in np.sort(idx)]
    if nest:
        avg_x = torch.zeros((len(idx), idx_count.max(), x.shape[1]), dtype=torch.float32)
    else:
        avg_x = torch.zeros((len(idx), 1, x.shape[1]), dtype=torch.float32)
    for i, m in enumerate(idx_list):
        if nest:
            avg_x[i] = x[m]
        else:
            avg_x[i] = torch.mean(x[m], axis=0)
    if isinstance(y, list):
        nested_y = []
        for i, m in enumerate(idx_list):
            indexed_y = [y[j] for j in range(len(y)) if m[j]]
            nested_y.append(indexed_y)
        y = nested_y
        
    return avg_x, y, len(idx_count)


# %%
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=2048,
)
parser.add_argument(
    "--new_test",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seq_len",type=int,default=1,
)
parser.add_argument(
    "--mode",type=str,default="vision",
)
parser.add_argument(
    "--gen_rep",type=int,default=0,
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# seed all random functions
seed=randrange(10000)
utils.seed_everything(seed=seed)

# make output directory
os.makedirs("evals",exist_ok=True)
os.makedirs(f"evals/{model_name}",exist_ok=True)

# %%
# voxels = {}
# # Load hdf5 data for betas
# f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
# betas = f['betas'][:]
# betas = torch.Tensor(betas).to("cpu")
# num_voxels = betas[0].shape[-1]
# voxels[f'subj0{subj}'] = betas
# print(f"num_voxels for subj0{subj}: {num_voxels}")

if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
else: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
    
# print(test_url)
# def my_split_by_node(urls): return urls
# test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
#                     .decode("torch")\
#                     .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
#                     .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
# test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
# print(f"Loaded test dl for subj{subj}!\n")

# %%
# # Prep images but don't load them all to memory
# f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
# images = f['images']

# # Prep test voxels and indices of test images
# test_images_idx = []
# test_voxels_idx = []
# for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
#     test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
#     test_voxels_idx = np.append(test_images_idx, behav[:,0,5].cpu().numpy())
#     test_images_idx = np.append(test_images_idx, behav[:,0,0].cpu().numpy())
# test_images_idx = test_images_idx.astype(int)
# test_voxels_idx = test_voxels_idx.astype(int)

# assert (test_i+1) * num_test == len(test_voxels) == len(test_images_idx)
# print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))

voxels, stimulus = load_nsd_mental_imagery(vector = "images", subject=subj, mode=mode, stimtype="all", average=False, nest=True)
num_voxels = voxels.shape[-1]

# %%
clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)
clip_seq_dim = 256
clip_emb_dim = 1664

if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
        
model = MindEyeModule()

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features, seq_len): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(seq_len)], dim=1)
        return out
        
model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim, seq_len=seq_len)

from diffusers.models.vae import Decoder
class BrainNetwork(nn.Module):
    def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=n_blocks, drop=.15, 
                 clip_size=768):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
        
        if blurry_recon:
            self.blin1 = nn.Linear(h*seq_len,4*28*28,bias=True)
            self.bdropout = nn.Dropout(.3)
            self.bnorm = nn.GroupNorm(1, 64)
            self.bupsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[32, 64, 128],
                layers_per_block=1,
            )
            self.b_maps_projector = nn.Sequential(
                nn.Conv2d(64, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=True),
            )
            
    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
        
    def forward(self, x):
        # make empty tensors
        c,b,t = torch.Tensor([0.]), torch.Tensor([[0.],[0.]]), torch.Tensor([0.])
        
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        c = self.clip_proj(backbone)

        if blurry_recon:
            b = self.blin1(x)
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm(b)
            b_aux = self.b_maps_projector(b).flatten(2).permute(0,2,1)
            b_aux = b_aux.view(len(b_aux), 49, 512)
            b = (self.bupsampler(b), b_aux)
        
        return backbone, c, b

model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=seq_len, 
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim) 
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
model.to(device)

utils.count_params(model.diffusion_prior)
utils.count_params(model)

# Load pretrained model ckpt
tag='last'
outdir = os.path.abspath(f'../train_logs/{model_name}')
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir+f'/{tag}.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
except: # probably ckpt is saved using deepspeed format
    import deepspeed
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")

# %%
# setup text caption networks
from transformers import AutoProcessor, AutoModelForCausalLM
from modeling_git import GitForCausalLMClipEmb
processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
clip_text_model.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
clip_text_model.eval().requires_grad_(False)
clip_text_seq_dim = 257
clip_text_emb_dim = 1024

class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0,2,1))
        return x
        
clip_convert = CLIPConverter()
state_dict = torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
clip_convert.load_state_dict(state_dict, strict=True)
clip_convert.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
del state_dict

# %%
# prep unCLIP
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
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)

ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])

batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
print("vector_suffix", vector_suffix.shape)

# %%
# get all reconstructions
model.to(device)
model.eval().requires_grad_(False)

# all_images = None
all_blurryrecons = None
all_recons = None
all_predcaptions = []
all_clipvoxels = None

minibatch_size = 1
num_samples_per_image = 1
plotting = False

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for idx in tqdm(range(0,voxels.shape[0])):
        voxel = voxels[idx]
        voxel = voxel.to(device)
        for rep in range(voxel.shape[0]):
            voxel_ridge = model.ridge(voxel[None,None,rep],0) # 0th index of subj_list
            backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
            if rep==0:
                clip_voxels = clip_voxels0
                backbone = backbone0
                blurry_image_enc = blurry_image_enc0[0]
            else:
                clip_voxels += clip_voxels0
                backbone += backbone0
                blurry_image_enc += blurry_image_enc0[0]
        clip_voxels /= voxel.shape[0]
        backbone /= voxel.shape[0]
        blurry_image_enc /= voxel.shape[0]
                
        # Save retrieval submodule outputs
        if all_clipvoxels is None:
            all_clipvoxels = clip_voxels
        else:
            all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels))
        
        # Feed voxels through OpenCLIP-bigG diffusion prior
        prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                        text_cond = dict(text_embed = backbone), 
                        cond_scale = 1., timesteps = 20)
        
        pred_caption_emb = clip_convert(prior_out)
        generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_predcaptions = np.hstack((all_predcaptions, generated_caption))
        print(generated_caption)
        
        # Feed diffusion prior outputs through unCLIP

        samples = utils.unclip_recon(prior_out,
                            diffusion_engine,
                            vector_suffix,
                            num_samples=num_samples_per_image)
        if all_recons is None:
            all_recons = samples.cpu()
        else:
            all_recons = torch.vstack((all_recons, samples.cpu()))
        if plotting:
            for s in range(num_samples_per_image):
                plt.figure(figsize=(2,2))
                plt.imshow(transforms.ToPILImage()(samples[s]))
                plt.axis('off')
                plt.show()

        if blurry_recon:
            blurred_image = (autoenc.decode(blurry_image_enc/0.18215).sample/ 2 + 0.5).clamp(0,1)
            
            im = torch.Tensor(blurred_image)
            if all_blurryrecons is None:
                all_blurryrecons = im.cpu()
            else:
                all_blurryrecons = torch.vstack((all_blurryrecons, im.cpu()))
            if plotting:
                plt.figure(figsize=(2,2))
                plt.imshow(transforms.ToPILImage()(im))
                plt.axis('off')
                plt.show()

        if plotting: 
            print(model_name)
            err # dont actually want to run the whole thing with plotting=True

# resize outputs before saving
imsize = 256
print(all_recons.shape, all_blurryrecons.shape)
all_recons = transforms.Resize((imsize,imsize))(all_recons).float()
if blurry_recon: 
    all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()
        
# saving
print(all_recons.shape)
# torch.save(all_images,"evals/all_images.pt")
if blurry_recon:
    torch.save(all_blurryrecons,f"evals/{model_name}/{model_name}_all_blurryrecons_{mode}_{gen_rep}.pt")
torch.save(all_recons,f"evals/{model_name}/{model_name}_all_recons_{mode}_{gen_rep}.pt")
torch.save(all_predcaptions,f"evals/{model_name}/{model_name}_all_predcaptions_{mode}_{gen_rep}.pt")
torch.save(all_clipvoxels,f"evals/{model_name}/{model_name}_all_clipvoxels_{mode}_{gen_rep}.pt")
print(f"saved {model_name} mi outputs for rep {gen_rep}!")

if not utils.is_interactive():
    sys.exit(0)


