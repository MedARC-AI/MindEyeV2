import os
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import PIL
import clip
import open_clip
from functools import partial
import random
import json

# class BrainMLP(nn.Module):
#     def __init__(self, out_dim=257*768, in_dim=15724, clip_size=768, h=4096):
#         super().__init__()
#         self.lin0 = nn.Sequential(
#             nn.Linear(in_dim, h, bias=False),
#             nn.LayerNorm(h),
#             nn.GELU(inplace=True),
#             nn.Dropout(0.5))
#         self.mlp = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(h, h),
#                 nn.LayerNorm(h),
#                 nn.GELU(inplace=True),
#                 nn.Dropout(0.15)
#             ) for _ in range(4)])
#         self.lin1 = nn.Linear(h, out_dim, bias=True)
#         self.proj = nn.Sequential(
#             nn.LayerNorm(clip_size),
#             nn.GELU(),
#             nn.Linear(clip_size, 2048),
#             nn.LayerNorm(2048),
#             nn.GELU(),
#             nn.Linear(2048, 2048),
#             nn.LayerNorm(2048),
#             nn.GELU(),
#             nn.Linear(2048, clip_size))
#     def forward(self, x):
#         x = self.lin0(x)
#         residual = x
#         for res_block in range(self.n_blocks):
#             x = self.mlp[res_block](x)
#             x += residual
#             residual = x
#         diffusion_prior_input = self.lin1(x.reshape(len(x), -1))
#         disjointed_clip_fmri = self.proj(diffusion_prior_input.reshape(
#                                         len(x),-1, self.clip_size))
#         return diffusion_prior_input, disjointed_clip_fmri



class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False,
                 hidden_state=False, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)
        
        if clip_variant=="ViT-L/14" and hidden_state:
            # from transformers import CLIPVisionModelWithProjection
            # image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",cache_dir="/fsx/proj-medarc/fmri/cache")
            from transformers import CLIPVisionModelWithProjection
            sd_cache_dir = '/fsx/proj-fmri/shared/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_cache_dir, subfolder='image_encoder').eval()
            image_encoder = image_encoder.to(device)
            for param in image_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.image_encoder = image_encoder
        elif hidden_state:
            raise Exception("hidden_state embeddings only works with ViT-L/14 right now")
        
        clip_model, preprocess = clip.load(clip_variant, device=device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)
            
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc
        self.hidden_state = hidden_state
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.device= device
        
        def versatile_normalize_embeddings(encoder_output):
            embeds = encoder_output.last_hidden_state
            embeds = image_encoder.vision_model.post_layernorm(embeds)
            embeds = image_encoder.visual_projection(embeds)
            return embeds
        self.versatile_normalize_embeddings = versatile_normalize_embeddings

    def resize_image(self, image):
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return transforms.Resize(self.clip_size)(image.to(self.device))

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        if self.hidden_state:
            # clip_emb = self.preprocess((image/1.5+.25).to(self.device)) # for some reason the /1.5+.25 prevents oversaturation
            clip_emb = self.preprocess((image).to(self.device))
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self.versatile_normalize_embeddings(clip_emb)
        else:
            clip_emb = self.preprocess(image.to(self.device))
            clip_emb = self.clip.encode_image(clip_emb)
        # input is now in CLIP space, but mind-reader preprint further processes embeddings:
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        if self.norm_embs:
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        return clip_emb

    def embed_text(self, text_samples):
        clip_text = clip.tokenize(text_samples).to(self.device)
        clip_text = self.clip.encode_text(clip_text)
        if self.clamp_embs:
            clip_text = torch.clamp(clip_text, -1.5, 1.5)
        if self.norm_embs:
            clip_text = nn.functional.normalize(clip_text, dim=-1)
        return clip_text

    def embed_curated_annotations(self, annots):
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)
    
class OpenClipper(torch.nn.Module):
    def __init__(self, clip_variant, norm_embs=False, device=torch.device('cpu')):
        super().__init__()
        print(clip_variant, device)
        assert clip_variant == 'ViT-H-14' # not setup for other models yet
                
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', 
                                        pretrained='laion2b_s32b_b79k', device=device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        # overwrite preprocess to accept torch inputs instead of PIL Image
        preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
            
        self.clip = clip_model
        self.norm_embs = norm_embs
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        
    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        image = self.preprocess(image).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip.encode_image(image)
        if self.norm_embs:
            image_features = nn.functional.normalize(image_features, dim=-1)
        return image_features

    def embed_text(self, text_samples):
        text = self.tokenizer(text_samples).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip.encode_text(text)
        if self.norm_embs:
            text_features = nn.functional.normalize(text_features, dim=-1)
        return text_features

    def embed_curated_annotations(self, annots):
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)