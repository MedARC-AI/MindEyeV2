#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # Code to convert this notebook to .py if you want to run it via command line or with Slurm
# from subprocess import call
# command = "jupyter nbconvert Train_MLPMixer.ipynb --to python"
# call(command,shell=True)


# # Import packages & functions

# In[1]:


import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import umap
from scipy.spatial.distance import cdist

import webdataset as wds
import gc

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from diffusers import StableDiffusionXLPipeline


from accelerate import Accelerator, DeepSpeedPlugin

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils


# In[2]:


### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1

# ## UNCOMMENT BELOW SECTION AND COMMENT OUT DEEPSPEED SECTION TO AVOID USING DEEPSPEED ###
accelerator = Accelerator(split_batches=False, mixed_precision="fp16", gradient_accumulation_steps=4)
global_batch_size = batch_size = 32
data_type = torch.float16 # change depending on your mixed_precision

### DEEPSPEED INITIALIZATION ###
# if num_devices <= 1 and utils.is_interactive():
#     global_batch_size = batch_size = 128
#     print(f"Setting batch_size to {batch_size}")
#     # can emulate a distributed environment for deepspeed to work in jupyter notebook
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = str(np.random.randint(10000)+9000)
#     os.environ["RANK"] = "0"
#     os.environ["LOCAL_RANK"] = "0"
#     os.environ["WORLD_SIZE"] = "1"
#     os.environ["GLOBAL_BATCH_SIZE"] = str(global_batch_size) # set this to your batch size!
# else:
#     global_batch_size = os.environ["GLOBAL_BATCH_SIZE"]    
#     batch_size = int(os.environ["GLOBAL_BATCH_SIZE"]) // num_devices

# # alter the deepspeed config according to your global and local batch size
# if local_rank == 0:
#     with open('deepspeed_config_stage2.json', 'r') as file:
#         config = json.load(file)
#     config['train_batch_size'] = int(os.environ["GLOBAL_BATCH_SIZE"])
#     config['train_micro_batch_size_per_gpu'] = batch_size
#     config['bf16'] = {'enabled': False}
#     config['fp16'] = {'enabled': True}
#     with open('deepspeed_config_stage2.json', 'w') as file:
#         json.dump(config, file)
# else:
#     # give some time for the local_rank=0 gpu to prep new deepspeed config file
#     time.sleep(10)
# deepspeed_plugin = DeepSpeedPlugin("deepspeed_config_stage2.json")
# accelerator = Accelerator(split_batches=False)


# In[3]:


print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
num_workers = num_devices
print(accelerator.state)
world_size = accelerator.state.num_processes
distributed = False

# set data_type to match your mixed precision (automatically set based on deepspeed config)
if accelerator.mixed_precision == "bf16":
    data_type = torch.bfloat16
elif accelerator.mixed_precision == "fp16":
    data_type = torch.float16
else:
    data_type = torch.float32

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
 # only print if local_rank=0


# # Configurations

# In[4]:


# if running this interactively, can specify jupyter_args here for argparser to use

    # create random model_name
model_name = "diffusion_non_contrastive"
print("model_name:", model_name)

    # global_batch_size and batch_size should already be defined in the above cells
    # other variables can be specified in the following string:
jupyter_args = f"--data_path=/fsx/proj-fmri/shared/mindeyev2_dataset \
                    --model_name={model_name} \
                    --subj=1 --batch_size={batch_size} \
                    --clip_scale=1. \
                    --max_lr=3e-5 --mixup_pct=.66 --num_epochs=3200"

jupyter_args = jupyter_args.split()
print(jupyter_args)



# In[5]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--data_path", type=str, default="/fsx/proj-fmri/shared/natural-scenes-dataset",
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,5,7],
)
parser.add_argument(
    "--batch_size", type=int, default=32,
    help="Batch size can be increased by 10x if only training v2c and not diffusion diffuser",
)
parser.add_argument(
    "--mixup_pct",type=float,default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--clip_scale",type=float,default=1.,
    help="multiply contrastive loss by this number",
)
parser.add_argument(
    "--num_epochs",type=int,default=120,
    help="number of epochs of training",
)
parser.add_argument(
    "--hidden_dim",type=int,default=4096,
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
)
parser.add_argument(
    "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)


args = parser.parse_args(jupyter_args)

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)


# In[6]:


outdir = os.path.abspath(f'../train_logs/{model_name}')
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)
# # Prep data, models, and dataloaders

# ## Dataloader

# In[7]:


if subj==1:
    num_train = 24958
    num_test = 2770
test_batch_size = num_test

def my_split_by_node(urls): return urls
    
train_url = f"{data_path}/wds/subj0{subj}/train/" + "{0..36}.tar"
# train_url = f"{data_path}/wds/subj0{subj}/train/" + "{0..1}.tar"
print(train_url)

train_data = wds.WebDataset(train_url,resampled=False,nodesplitter=my_split_by_node)\
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)

test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
print(test_url)

test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, drop_last=True, pin_memory=True)


# ### check dataloaders are working

# In[8]:


test_vox_indices = []
test_73k_images = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_vox_indices = np.append(test_vox_indices, behav[:,0,5].cpu().numpy())
    test_73k_images = np.append(test_73k_images, behav[:,0,0].cpu().numpy())
test_vox_indices = test_vox_indices.astype(np.int16)
print(test_i, (test_i+1) * test_batch_size, len(test_vox_indices))
print("---\n")

train_vox_indices = []
train_73k_images = []
for train_i, (behav, past_behav, future_behav, old_behav) in enumerate(train_dl):
    train_vox_indices = np.append(train_vox_indices, behav[:,0,5].long().cpu().numpy())
    train_73k_images = np.append(train_73k_images, behav[:,0,0].cpu().numpy())
train_vox_indices = train_vox_indices.astype(np.int16)
print(train_i, (train_i+1) * batch_size, len(train_vox_indices))


# ## Load data and images

# In[9]:


# load betas
f = h5py.File(f'{data_path}/betas_all_subj0{subj}.hdf5', 'r')
# f = h5py.File(f'{data_path}/betas_subj0{subj}_thresholded_wholebrain.hdf5', 'r')

voxels = f['betas'][:]
print(f"subj0{subj} betas loaded into memory")
voxels = torch.Tensor(voxels).to("cpu").to(data_type)
print("voxels", voxels.shape)
num_voxels = voxels.shape[-1]

# load orig images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'][:]
images = torch.Tensor(images).to("cpu").to(data_type)
print("images", images.shape)


# ## Load models

# ### CLIP image embeddings  model

# In[10]:


from models import Clipper
clip_model = Clipper("ViT-L/14", device=torch.device(f"cuda:{local_rank}"), hidden_state=True, norm_embs=True)
# clip_seq_dim = 257
# clip_emb_dim = 768 #1024
hidden_dim = 2048
seq_len=1
prior_mult = 1e-4
plot_umap = True
wandb_log = True

from diffusers import StableDiffusionXLPipeline, AutoencoderKL
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).cuda()
vae.requires_grad_(False)
vae.eval()

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
 


# ### SD VAE

# In[11]:


# if blurry_recon:
#     from diffusers import AutoencoderKL
#     autoenc = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir="/fsx/proj-fmri/shared/cache")
#     # autoenc.load_state_dict(torch.load('../train_logs/sdxl_vae_normed/best.pth')["model_state_dict"])
#     autoenc.eval()
#     autoenc.requires_grad_(False)
#     autoenc.to(device)
#     utils.count_params(autoenc)

# In[12]:




# In[31]:


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
        
model = MindEyeModule()


# In[32]:


class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_size, out_features): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linear = torch.nn.Linear(input_size, out_features)
    def forward(self, x):
        return self.linear(x)
        
model.ridge = RidgeRegression(voxels.shape[1], out_features=hidden_dim)

utils.count_params(model)

b = torch.randn((2,1,voxels.shape[1]))
print(b.shape, model.ridge(b).shape)


# In[33]:


from functools import partial
from diffusers.models.vae import Decoder
class BrainNetwork(nn.Module):
    def __init__(self, text, pool, out_dim=2048, in_dim=15724, seq_len=2, h=4096, n_blocks=4, drop=.15, clip_size=2048):
        super().__init__()
        self.is_text = text
        self.is_pool = pool
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        
        # Initial linear layer to match the input dimensions to hidden dimensions
        # self.lin0 = nn.Linear(in_dim, seq_len * h)
        
        # Mixer Blocks
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        if self.is_text:
            self.clin1 = nn.Linear(h * seq_len, out_dim, bias=True)
            self.clin2 = nn.Linear(h * seq_len, 768*257, bias=True)
            self.clip_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 768)
        )
        if self.is_pool:
            self.clin3 = nn.Sequential(
                nn.Linear(h * seq_len, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Linear(out_dim, 1280),
            )

        # low-rank matrices
        # self.rank = 500
        # self.U = nn.Parameter(torch.randn(self.rank, out_dim))
        # self.V = nn.Parameter(torch.randn(h * seq_len, self.rank))
        # self.S = nn.Parameter(torch.randn(out_dim))

        

        self.text_proj = nn.Sequential(
            nn.LayerNorm(clip_size),
            nn.GELU(),
            nn.Linear(clip_size, 2048),
            # nn.Dropout(0.5),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            # nn.Dropout(0.5),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, clip_size)
        )

#         self.pooled_proj = nn.Sequential(
#             nn.LayerNorm(157696),
#             nn.GELU(),
#             nn.Linear(157696, h),
#             nn.LayerNorm(h),
#             nn.GELU(),
#             nn.Linear(h, h),
#             nn.LayerNorm(h),
#             nn.GELU(),
#             nn.Linear(h, 1280)
#         )

        
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
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
        
    def forward(self, x):
        # make empty tensors for blur and depth outputs
        
        # Initial linear layer
        # x = self.lin0(x)
        
        # Reshape to seq_len by dim
        # x = x.reshape(-1, self.seq_len, self.h)
        
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
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        if self.is_text:
            c = self.clin1(x)
            i = self.clin2(x)
            i_proj = self.clip_proj(i.reshape(len(i), -1, 768))
            c = c.reshape(len(c), -1, self.clip_size)
            txt_proj = self.text_proj(c)
            return txt_proj, i_proj

        if self.is_pool:
            c_p = self.clin3(x)
            return c_p


        
b_text = BrainNetwork(text=True, pool=False, h=hidden_dim, in_dim=2048, seq_len=seq_len, clip_size=2048, out_dim=2048*77).cuda()
# b_pool = BrainNetwork(text=False, pool=True, h=hidden_dim, in_dim=2048, seq_len=seq_len, clip_size=2048, out_dim=2048*77).cuda()
utils.count_params(b_text)
# utils.count_params(b_pool)

# test that the model works on some fake data
b = torch.randn((2,seq_len,hidden_dim)).cuda()
print("b.shape",b.shape)
emb, img = b_text(b)
print(emb.shape, img.shape)

# In[34]:


from models import BrainDiffusionPrior, VersatileDiffusionPriorNetwork
timesteps=100
out_dim = 768 
depth = 6
dim_head = 64
heads = out_dim//64


prior_network = VersatileDiffusionPriorNetwork(
            dim=2048,
            align="text",
            depth=depth,
            dim_head=128,
            heads=16,
            causal=False,
            num_tokens = 77,
            learned_query_mode="pos_emb"
        )

model.diffusion_prior_txt = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=2048,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=b_text,
)


# In[35]:


print("params of diffusion prior:")
if local_rank==0:
    utils.count_params(model.diffusion_prior_txt)
    
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.diffusion_prior_txt.net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.diffusion_prior_txt.net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.diffusion_prior_txt.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
]

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(num_epochs*(num_train/num_devices/batch_size))),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps=int(np.floor(num_epochs*(num_train/num_devices/batch_size/4)))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )
        
print("\nDone with model preparations!")

if local_rank==0 and wandb_log: # only use main process for wandb logging
    import wandb
    wandb_project = 'mindeyev2'
    wandb_run = model_name
    wandb_notes = ''
    
    print(f"wandb {wandb_project} run {wandb_run}")
    wandb.login(host='https://stability.wandb.io')#, relogin=True)
    wandb_config = {
      "model_name": model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "max_lr": max_lr,
      "mixup_pct": mixup_pct,
      "num_train": num_train,
      "num_test": num_test
    }
    print("wandb_config:\n",wandb_config)
    if True: # wandb_auto_resume
        print("wandb_id:",model_name)
        wandb.init(
            id = model_name,
            project=wandb_project,
            name=wandb_run,
            config=wandb_config,
            notes=wandb_notes,
            resume="allow",
        )
    else:
        wandb.init(
            project=wandb_project,
            name=wandb_run,
            config=wandb_config,
            notes=wandb_notes,
        )
else:
    wandb_log = False

epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))




model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
model, optimizer, train_dl, lr_scheduler
)

annots = np.load("/fsx/proj-fmri/shared/mindeyev2_dataset/COCO_73k_annots_curated.npy")

def extract_first_non_empty(input_array):    
    non_empty_strings = input_array[input_array != '']
    return non_empty_strings[0] if len(non_empty_strings) > 0 else None

def text_embeds(sdXLpipeline, text):
    txt_emb, _, pool_emb, _= sdXLpipeline.encode_prompt(prompt=text)
    return txt_emb, pool_emb

print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
test_image, test_voxel = None, None
test_cap = []


for epoch in progress_bar:
    model.train()
    
    sims_base = 0.
    val_sims_base = 0.
    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    val_fwd_percent_correct = 0.
    val_bwd_percent_correct = 0.
    loss_nce_sum = 0.
    loss_prior_sum = 0.
    val_loss_nce_sum = 0.
    val_loss_prior_sum = 0.

    
    for train_i, (behav, past_behav, future_behav, old_behav) in enumerate(train_dl):
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            repeat_index = train_i % 3
            
            voxel = voxels[behav[:,0,5].cpu().long()].to(device)
            img_id = behav[:,0,0].cpu().long()
            image = images[img_id].to(device).float()
            caps = []
            for i in img_id:
                ann = annots[i]
                cap = extract_first_non_empty(ann)
                caps.append(cap)
            
            if epoch < int(mixup_pct * num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel)

            clip_target = clip_model.embed_image(image).float()
            txt_emb, _ = text_embeds(pipe, caps)

            assert not torch.any(torch.isnan(txt_emb))
            assert not torch.any(torch.isnan(clip_target))
            model = accelerator.unwrap_model(model)
            voxel_ridge = model.ridge(voxel)
            voxel_ridge = voxel_ridge.unsqueeze(1)

            clip_txt_voxels, clip_voxels_proj = model.diffusion_prior_txt.modules.voxel2clip(voxel_ridge) if distributed else model.diffusion_prior_txt.voxel2clip(voxel_ridge)

            loss_txt_prior, aligned_clip_txt_voxels = model.diffusion_prior_txt(text_embed=clip_txt_voxels, image_embed=txt_emb)
            aligned_clip_txt_voxels /= model.diffusion_prior_txt.modules.image_embed_scale if distributed else model.diffusion_prior_txt.image_embed_scale

            clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if epoch < int(mixup_pct * num_epochs):
                loss_nce = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006, 
                    perm=perm, betas=betas, select=select)
            else:
                epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                loss_nce = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=epoch_temp)
        
            loss_nce_sum += loss_nce.item()
            loss_prior_sum += loss_txt_prior.item()
            if loss_txt_prior >= 1e4:
                prior_mult = 1e-3
            elif 1e3 <= loss_txt_prior < 9.99e3:
                prior_mult = 1e-2
            elif 1e2 <= loss_txt_prior < 9.99e2:
                prior_mult = 1e-1
            else:
                prior_mult = 1e-1
            loss = loss_txt_prior
            
            # utils.check_loss(loss_nce)
            if utils.check_loss(loss_txt_prior):
                continue
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            sims_base += nn.functional.cosine_similarity(clip_target_norm,clip_voxels_norm).mean().item()

            # forward and backward top 1 accuracy        
            labels = torch.arange(len(clip_target_norm)).to(device) 
            fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm,clip_target_norm), labels, k=1)
            bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1)
    
            if lr_scheduler_type is not None:
                lr_scheduler.step()

    model.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                # all test samples should be loaded per batch such that test_i should never exceed 0
                assert len(behav) == num_test
                
                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels[behav[:,0,5].cpu().long()]
                    image = behav[:,0,0].cpu().long()


                    
                    unique_image, sort_indices = torch.unique(image, return_inverse=True)
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if test_image is None:
                            test_image = images[im][None]
                            test_voxel = torch.mean(voxel[locs],axis=0)[None]
                            test_cap.append(extract_first_non_empty(annots[im]))
                            
                        else:
                            test_image = torch.vstack((test_image, images[im][None]))
                            test_cap.append(extract_first_non_empty(annots[im]))
                            test_voxel = torch.vstack((test_voxel, torch.mean(voxel[locs],axis=0)[None]))
    
                # random sample of 300
                random_indices = torch.arange(len(test_voxel))[:32]
                index_list = random_indices.tolist()
                voxel = test_voxel[random_indices].to(device)
                image = test_image[random_indices].to(device)
                cap = [test_cap[i] for i in index_list]
                # assert len(voxel) == 16

                clip_target = clip_model.embed_image(image.float())
                txt_emb, _ = text_embeds(pipe, caps)
                model = accelerator.unwrap_model(model)
                voxel_ridge = model.ridge(voxel).unsqueeze(1)
                
                clip_txt_voxels, clip_voxels_proj = model.diffusion_prior_txt.modules.voxel2clip(voxel_ridge) if distributed else model.diffusion_prior_txt.voxel2clip(voxel_ridge)

                val_loss_txt_prior, aligned_clip_txt_voxels = model.diffusion_prior_txt(text_embed=clip_txt_voxels, image_embed=txt_emb)
                aligned_clip_txt_voxels /= model.diffusion_prior_txt.modules.image_embed_scale if distributed else model.diffusion_prior_txt.image_embed_scale

                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                if epoch < int(mixup_pct * num_epochs):
                    val_loss_nce = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006, 
                        perm=None, betas=None, select=None)
                else:
                    val_loss_nce = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                val_loss_nce_sum += val_loss_nce.item()
                val_loss_prior_sum += val_loss_txt_prior.item()
                val_loss =  val_loss_txt_prior

                
                # utils.check_loss(val_loss_nce)                
                if utils.check_loss(val_loss_txt_prior):
                    continue              
                test_losses.append(val_loss.item())

            # if utils.is_interactive(): clear_output(wait=True)
            print("---")


            assert (test_i+1) == 1
            logs = {
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "train/loss_prior": loss_prior_sum / (train_i + 1),
                "train/loss_nce": loss_nce_sum / (train_i + 1),
                "test/val_loss_prior": val_loss_prior_sum / (test_i + 1),
                "test/val_loss_nce": val_loss_nce_sum / (test_i + 1)
                }
            
            if wandb_log: wandb.log(logs)
            progress_bar.set_postfix(**logs)
            if plot_umap:
                    plt.clf()
                    print('umap plotting...')
                    ground_truth = txt_emb[[0]].squeeze(0).detach().cpu().numpy()
                    aligned = aligned_clip_txt_voxels[[0]].squeeze(0).detach().cpu().numpy()
                    combined_embeddings = np.vstack((aligned, ground_truth))

                    # Initialize the UMAP transformer
                    reducer = umap.UMAP(random_state=42)

                    # Fit the transformer to the combined data
                    embedding = reducer.fit_transform(combined_embeddings)

                    # Split the transformed data back into the original two sets
                    aligned_umap_1 = embedding[:len(aligned), :]
                    ground_truth_umap_2 = embedding[len(ground_truth):, :]
                    distances = cdist(aligned_umap_1, ground_truth_umap_2, metric='euclidean')
                    average_distance = np.mean(distances)
                    # Plotting the results
                    print(f'Average Euclidean distance between the two sets of embeddings: {average_distance}. Epoch: {epoch}')
                    plt.scatter(aligned_umap_1[:, 0], aligned_umap_1[:, 1], label='Aligned Embed')
                    plt.scatter(ground_truth_umap_2[:, 0], ground_truth_umap_2[:, 1], label='Ground Truth')
                    plt.legend()
                    plt.title('UMAP projection of two embeddings')

                    wandb.log({"UMAP plot": wandb.Image(plt)})

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    gc.collect()

print("\n===Finished!===\n")
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    model = accelerator.unwrap_model(model)
    torch.save(model.state_dict(), "/fsx/proj-fmri/shared/models/fmri-txt-img/diff_contrastive.pth")

