import numpy as np
import pandas as pd
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import random
import os
import pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math
import webdataset as wds

import json
from PIL import Image
import requests
import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def batchwise_pearson_correlation(Z, B):
    # Calculate means
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    B_mean = torch.mean(B, dim=1, keepdim=True)

    # Subtract means
    Z_centered = Z - Z_mean
    B_centered = B - B_mean

    # Calculate Pearson correlation coefficient
    numerator = Z_centered @ B_centered.T
    Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
    B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
    denominator = Z_centered_norm @ B_centered_norm.T

    pearson_correlation = (numerator / denominator)
    return pearson_correlation

def batchwise_cosine_similarity(Z,B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def prenormed_batchwise_cosine_similarity(Z,B):
    return (Z @ B.T).T

def cosine_similarity(Z,B,l=0):
    Z = nn.functional.normalize(Z, p=2, dim=1)
    B = nn.functional.normalize(B, p=2, dim=1)
    # if l>0, use distribution normalization
    # https://twitter.com/YifeiZhou02/status/1716513495087472880
    Z = Z - l * torch.mean(Z,dim=0)
    B = B - l * torch.mean(B,dim=0)
    cosine_similarity = (Z @ B.T).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def get_non_diagonals(a):
    a = torch.triu(a,diagonal=1)+torch.tril(a,diagonal=-1)
    # make diagonals -1
    a=a.fill_diagonal_(-1)
    return a

def gather_features(image_features, voxel_features, accelerator):  
    all_image_features = accelerator.gather(image_features.contiguous())
    if voxel_features is not None:
        all_voxel_features = accelerator.gather(voxel_features.contiguous())
        return all_image_features, all_voxel_features
    return all_image_features

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def soft_siglip_loss(preds, targs, temp, bias):
    temp = torch.exp(temp)
    
    logits = (preds @ targs.T) * temp + bias
    # diagonals (aka paired samples) should be >0 and off-diagonals <0
    labels = (targs @ targs.T) - 1 + (torch.eye(len(targs)).to(targs.dtype).to(targs.device))

    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels[:len(preds)])) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels[:,:len(preds)])) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco_hard_siglip_loss(preds, targs, temp, bias, perm, betas):
    temp = torch.exp(temp)
    
    probs = torch.diag(betas)
    probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

    logits = (preds @ targs.T) * temp + bias
    labels = probs * 2 - 1
    #labels = torch.eye(len(targs)).to(targs.dtype).to(targs.device) * 2 - 1
    
    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels)) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels)) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco(voxels, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_clip_target(clip_target, perm, select, betas):
    clip_target_shuffle = clip_target[perm]
    clip_target[select] = clip_target[select] * betas[select].reshape(-1, 1) + \
        clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
    return clip_target

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss
    
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))
    return trainable
    
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def resize(img, img_size=128):
    if img.ndim == 3: img = img[None]
    return nn.functional.interpolate(img, size=(img_size, img_size), mode='nearest')

pixcorr_preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
])
def pixcorr(images,brains,nan=True):
    all_images_flattened = pixcorr_preprocess(images).reshape(len(images), -1)
    all_brain_recons_flattened = pixcorr_preprocess(brains).view(len(brains), -1)
    if nan:
        corrmean = torch.nanmean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    else:
        corrmean = torch.mean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    return corrmean

def select_annotations(annots, random=True):
    """
    There are 5 annotations per image. Select one of them for each image.
    """
    for i, b in enumerate(annots):
        t = ''
        if random:
            # select random non-empty annotation
            while t == '':
                rand = torch.randint(5, (1,1))[0][0]
                t = b[rand]
        else:
            # select first non-empty annotation
            for j in range(5):
                if b[j] != '':
                    t = b[j]
                    break
        if i == 0:
            txt = np.array(t)
        else:
            txt = np.vstack((txt, t))
    txt = txt.flatten()
    return txt

from generative_models.sgm.util import append_dims

def unclip_recon(x, diffusion_engine, vector_suffix,
                 num_samples=1, offset_noise_level=0.04):
    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples,4,96,96).to(device) # starting noise, can change to VAE outputs of initial image for img2img

        # clip_img_tokenized = clip_img_embedder(image) 
        # tokens = clip_img_tokenized
        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples
    
def versatile_diffusion_recon(brain_clip_embeddings, 
                              proj_embeddings, 
                              img_lowlevel, 
                              text_token,
                              img2img_strength, 
                              clip_extractor, 
                              vae, 
                              unet, 
                              noise_scheduler, 
                              generator,
                              num_inference_steps,
                              recons_per_sample=16,
                              guidance_scale = 3.5,
                              seed=42):
    for samp in range(len(brain_clip_embeddings)):
        brain_clip_embeddings[samp] = brain_clip_embeddings[samp]/(brain_clip_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        
    input_embedding = brain_clip_embeddings
    if text_token is not None:
        prompt_embeds = text_token.repeat(recons_per_sample, 1, 1)
    else:
        prompt_embeds = torch.zeros(len(input_embedding),77,768)
        
    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    
    if do_classifier_free_guidance:
        input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
        prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)
    
    # dual_prompt_embeddings
    input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)
    # 4. Prepare timesteps
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

    # 5b. Prepare latent variables
    batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
    shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
    if img_lowlevel is not None: # use img_lowlevel for img2img initialization
        img_lowlevel = torch.nn.functional.interpolate(img_lowlevel, size=(512, 512), mode='bilinear', align_corners=False)
        init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = noise_scheduler.timesteps[t_start:]
        latent_timestep = timesteps[:1].repeat(batch_size)
        
        img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
        init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
        init_latents = vae.config.scaling_factor * init_latents
        init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

        noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                            generator=generator, dtype=input_embedding.dtype)
        init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
        latents = init_latents
    else:
        timesteps = noise_scheduler.timesteps
        latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                generator=generator, dtype=input_embedding.dtype)
        latents = latents * noise_scheduler.init_noise_sigma
    # 7. Denoising loop
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t).to(device)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    recons = decode_latents(latents,vae).detach().cpu()
    
    brain_recons = recons.unsqueeze(0)
    
    # pick best reconstruction out of several
    best_picks = np.zeros(1).astype(np.int16)

    v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings),-1),dim=-1)
    sims=[]
    for im in range(recons_per_sample): 
        currecon = clip_extractor.embed_image(brain_recons[0,[im]].float()).to(proj_embeddings.device).to(proj_embeddings.dtype)
        currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
        cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
        sims.append(cursim.item())
    best_picks[0] = int(np.nanargmax(sims))  
     
    recon_img = brain_recons[:, best_picks[0]]
    
    return recon_img, brain_recons, best_picks

def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

#  Numpy Utility 
def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 
        
# Torch fwRF
def get_value(_x):
    return np.copy(_x.data.cpu().numpy())

def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125):
    teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T)/temp
    teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T)/temp
    student_teacher_aug = (student_preds @ teacher_aug_preds.T)/temp
    student_teacher_aug_t = (teacher_aug_preds @ student_preds.T)/temp

    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def format_tiled_figure(images, captions, rows, cols, red_line_index=None, buffer=10, mode=0, title=None, font_size=60):
    """
    Assembles a tiled figure of images with optional captions and a red background behind a specified column or row.

    :param images: List of PIL Image objects, ordered row-wise.
    :param captions: List of captions, length and usage depends on mode.
    :param rows: Number of rows in the image grid.
    :param cols: Number of columns in the image grid.
    :param red_line_index: Index of the row or column to highlight with a red background (0-indexed).
    :param buffer: Buffer value in pixels for space between images.
    :param mode: Mode of the figure assembly.
    :param title: Title of the figure, used in mode 1 and mode 3.
    :return: PIL Image object of the assembled figure.
    """
    
    # Find the smallest width and height among all images
    min_width, min_height = min(img.size for img in images)

    # Resize all images to the smallest dimensions
    images = [img.resize((min_width, min_height), Image.ANTIALIAS) for img in images]

    # Font setup
    # font_size = 60  # Base font size for readability
    row_caption_font_size = font_size  
    title_font_size = int(1.3 * font_size) 
    title_font = ImageFont.truetype("arial.ttf", title_font_size)
    row_caption_font = ImageFont.truetype("arial.ttf", row_caption_font_size)

    # Calculate dimensions for the entire canvas
    caption_height = row_caption_font_size if mode in [0, 1] else 0
    title_height = int(title_font_size * 1.3) if mode in [1, 3] and title is not None or mode in [2] and captions is not None else 0  # Adjusted to include mode 3
    row_title_width = int(row_caption_font_size * 1.5) if mode == 3 else 0
    extra_buffer_w = buffer if (red_line_index is not None and mode in [0, 1, 2]) else 0
    extra_buffer_h = buffer if (red_line_index is not None and mode == 3) else 0

    # Calculate the total canvas width and height
    total_width = cols * (min_width + buffer) + row_title_width + buffer + extra_buffer_w
    total_height = rows * (min_height + buffer) + title_height + rows * caption_height + buffer + extra_buffer_h

    # Create a new image with a white background
    canvas = Image.new('RGB', (total_width, total_height), color='white')

    # Prepare the drawing context
    draw = ImageDraw.Draw(canvas)

    # Draw the title for modes 1 and 3
    if mode in [1, 3] and title is not None:  # Adjusted to include mode 3
        text_width, text_height = draw.textsize(title, font=title_font)
        draw.text(((total_width - text_width) // 2, (title_height - text_height) // 2), title, font=title_font, fill='black')

    # Draw red background before placing images if a red line index is specified
    if red_line_index is not None:
        if mode in [0, 1, 2]:  # Red column
            red_x = row_title_width + red_line_index * (min_width + buffer)
            red_y = title_height
            red_width = min_width + buffer * 2
            red_height = total_height - title_height
            canvas.paste(Image.new('RGB', (red_width, red_height), color='red'), (red_x, red_y))
        elif mode == 3:  # Red row
            red_x = row_title_width
            red_y = title_height + red_line_index * (min_height + buffer)
            red_width = total_width - row_title_width
            red_height = min_height + buffer * 2
            canvas.paste(Image.new('RGB', (red_width, red_height), color='red'), (red_x, red_y))

    # Insert images into the canvas
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(images):
                continue

            img = images[idx]
            x = col * (min_width + buffer) + row_title_width + buffer
            y = row * (min_height + buffer) + title_height + buffer

            # Adjust the x position if there is a red column
            if mode in [0, 1, 2] and red_line_index is not None and col > red_line_index:
                x += extra_buffer_w

            # Adjust the y position if there is a red row
            if mode == 3 and red_line_index is not None and row > red_line_index:
                y += extra_buffer_h

            # Paste the image
            canvas.paste(img, (x, y))
    # Draw the vertical text for row titles if mode is 3
    if mode == 3:
        for row, caption in enumerate(captions):
            # Calculate the caption size using the default font
            width, height = row_caption_font.getsize(caption)

            text_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_image)
            draw.text((0, 0), text=caption, font=row_caption_font, fill='black')

            # Rotate the text image to be vertical
            text_image = text_image.rotate(90, expand=1)

            # Calculate the y position for the vertical text
            y = row * (min_height + buffer) + (min_width - width )//2 + title_height
            if row > 0:
                y += buffer

            # Calculate the x position, accounting for the increased text size
            x = 0

            # Paste the rotated text image onto the canvas
            canvas.paste(text_image, (x, y), text_image)

    # Draw captions for each image for modes 0 and 1
    if mode in [0, 1]:
        for idx, caption in enumerate(captions):
            col = idx % cols
            row = idx // cols
            text_width, text_height = draw.textsize(caption, font=row_caption_font)
            x = col * (min_width + buffer) + row_title_width + buffer + (min_width - text_width) // 2
            y = (row + 1) * (min_height + buffer) + title_height - text_height // 2
            draw.text((x, y), caption, font=row_caption_font, fill='black')

    # Draw column titles if mode is 2
    if mode == 2:
        for col, caption in enumerate(captions):
            text_width, text_height = draw.textsize(caption, font=row_caption_font)
            x = col * (min_width + buffer) + row_title_width + buffer + (min_width - text_width) // 2
            y = buffer
            draw.text((x, y), caption, font=row_caption_font, fill='black')

    return canvas

def condition_average(x, y, cond, nest=False, num_reps=1000):
    idx, idx_count = np.unique(cond, return_counts=True)
    num_reps = min(num_reps, idx_count.max())
    idx_list = [np.array(cond)==i for i in np.sort(idx)]
    if nest:
        avg_x = torch.zeros((len(idx), num_reps, x.shape[1]), dtype=torch.float32)
    else:
        avg_x = torch.zeros((len(idx), 1, x.shape[1]), dtype=torch.float32)
    arranged_y = torch.zeros((len(idx)), y.shape[1], y.shape[2], y.shape[3])
    for i, m in enumerate(idx_list):
        num_reps = min(num_reps, len(m))
        if nest:
            if np.sum(m) == num_reps:
                avg_x[i] = x[m]
            else:
                avg_x[i,:max(np.sum(m), num_reps)] = x[m][:num_reps]
        else:
            avg_x[i] = torch.mean(x[m][:num_reps], axis=0)
        arranged_y[i] = y[m[0]]

    return avg_x, y, len(idx_count)

#subject: nsd subject index between 1-8
#mode: vision, imagery
#stimtype: all, simple, complex, concepts
#average: whether to average across trials, will produce x that is (stimuli, 1, voxels)
#nest: whether to nest the data according to stimuli, will produce x that is (stimuli, trials, voxels)
#data_root: path to where the dataset is saved.
def load_nsd_mental_imagery(subject, mode, stimtype="all", average=False, num_reps = 16, nest=False, data_root="../dataset/"):
    # This file has a bunch of information about the stimuli and cue associations that will make loading it easier
    img_stim_file = f"{data_root}/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"
    ex_file = open(img_stim_file, 'rb')
    imagery_dict = pickle.load(ex_file)
    ex_file.close()
    # Indicates what experiments trials belong to
    exps = imagery_dict['exps']
    # Indicates the cues for different stimuli
    cues = imagery_dict['cues']
    # Maps the cues to the stimulus image information
    image_map  = imagery_dict['image_map']
    # Organize the indices of the trials according to the modality and the type of stimuli
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
    # Load normalized betas
    x = torch.load(f"{data_root}/preprocessed_data/subject{subject}/nsd_imagery.pt").requires_grad_(False).to("cpu")
    # Find the trial indices conditioned on the type of trials we want to load
    cond_im_idx = {n: [image_map[c] for c in cues[idx]] for n,idx in cond_idx.items()}
    conditionals = cond_im_idx[mode+stimtype]
    # Stimuli file is of shape (18,3,425,425), these can be converted back into PIL images using transforms.ToPILImage()
    y = torch.load(f"{data_root}/nsddata_stimuli/stimuli/imagery_stimuli_18.pt").requires_grad_(False).to("cpu")
    # Prune the beta file down to specific experimental mode/stimuli type
    x = x[cond_idx[mode+stimtype]]
    # # If stimtype is not all, then prune the image data down to the specific stimuli type
    if stimtype == "simple":
        y = y[:6]
    elif stimtype == "complex":
        y = y[6:12]
    elif stimtype == "concepts":
        y = y[12:]

    # Average or nest the betas across trials
    if average or nest:
        x, y, sample_count = condition_average(x, y, conditionals, nest=nest, num_reps=num_reps)
    else:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        y = y[conditionals]

    print(x.shape, y.shape)
    return x, y

#subject: nsd subject index between 1-8
#average: whether to average across trials, will produce x that is (stimuli, 1, voxels)
#nest: whether to nest the data according to stimuli, will produce x that is (stimuli, trials, voxels)
#data_root: path to where the dataset is saved.
def load_nsd_synthetic(subject, average=False, nest=False, data_root="../dataset/"):
    y = torch.zeros((284, 3, 714, 1360))
    y[:220] = torch.load(f"{data_root}/nsddata_stimuli/stimuli/nsdsynthetic/nsd_synthetic_stim_part1.pt")
    #The last 64 stimuli are slightly different for each subject, so we load these separately for each subject
    y[220:] = torch.load(f"{data_root}/nsddata_stimuli/stimuli/nsdsynthetic/nsd_synthetic_stim_part2_sub{subject}.pt")
    
    x = torch.load(f"{data_root}/preprocessed_data/subject{subject}/nsd_synthetic.pt").requires_grad_(False).to("cpu")
    conditionals = loadmat(f'{data_root}/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat')['masterordering'][0].astype(int) - 1
    
    if average or nest:
        x, y, sample_count = condition_average(x, y, conditionals, nest=nest)
    else:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        y = y[conditionals]
    print(x.shape, y.shape)
    return x, y    

#subject: subject index between 1-3, or the subject identifier: subj01, subj02, subj03. These are NOT the NSD subjects as this is a different datasets
#mode: vision, imagery
#mask: True or False, if true masks the betas to visual cortex, otherwise returns the whole scanned region
#stimtype: stimuli, cue, object
    # - stimuli will return the images with content that was either seen or imagined, this is what was presented to the subject in vision trials
    # - cue will return only the background images with the cue and no content, this is what was presented to the subject in imagery trials
    # - object will return only the object in the image with no cue or location brackets. This should be used for model training where we dont want the model to learn the brackets or the cue.
#average: whether to average across trials, will produce x that is (stimuli, 1, voxels)
#nest: whether to nest the data according to stimuli, will produce x that is (stimuli, trials, voxels)
    # WARNING: Not all stimuli have the same number of repeats, so the middle dimension for the trial repetitions will contain empty values for some stimuli, be sure to account for this when loading
def load_imageryrf(subject, mode, mask=True, stimtype="object", average=False, nest=False, split=False, data_root="../dataset/"):
    
    # This file has a bunch of information about the stimuli and cue associations that will make loading it easier
    img_conditional_file = f"{data_root}/imageryrf_single_trial/stimuli/imageryrf_conditions.pkl3"
    ex_file = open(img_conditional_file, 'rb')
    conditional_dict = pd.compat.pickle_compat.load(ex_file) 
    ex_file.close()
    stimuli_metadata = conditional_dict['stimuli_metadata']
    # If subject identifier is int, grab the string identifer
    if isinstance(subject, int):
        subject = f"subj0{subject}"
    subject_cond = conditional_dict[subject]
    # Indicates what experiments trials belong to
    exps = subject_cond['experiment_cond']
    # Maps the cues to the stimulus image information
    image_map  = subject_cond['stimuli_cond'].to(int)
    # Identify and condition on the stimuli that will be the test set
    test_idx = torch.tensor([0,7,15,23,35,47,51,63])
    object_idx = torch.tensor(stimuli_metadata['object_idx'].values)
    test_indices = [idx for idx, value in enumerate(object_idx) if value in test_idx]
    
    # Organize the indices of the trials according to the modality and the type of stimuli
    cond_idx = {
    'vision': np.arange(len(exps))[np.char.find(exps, 'pcp') != -1],
    'imagery': np.arange(len(exps))[np.char.find(exps, 'img') != -1],
    'all': np.arange(len(exps)),
    'visiontrain': np.arange(len(exps))[np.logical_and(np.char.find(exps, 'pcp') != -1, ~np.isin(image_map, test_indices))],
    'visiontest': np.arange(len(exps))[np.logical_and(np.char.find(exps, 'pcp') != -1, np.isin(image_map, test_indices))],
    'imagerytrain': np.arange(len(exps))[np.logical_and(np.char.find(exps, 'img') != -1, ~np.isin(image_map, test_indices))],
    'imagerytest': np.arange(len(exps))[np.logical_and(np.char.find(exps, 'img') != -1, np.isin(image_map, test_indices))],
    'alltrain': np.arange(len(exps))[~np.isin(image_map, test_indices)],
    'alltest': np.arange(len(exps))[np.isin(image_map, test_indices)]}
    # Load normalized betas
    if mask:
        x = torch.load(f"{data_root}/imageryrf_single_trial/{subject}/single_trial_betas_masked.pt").requires_grad_(False).to("cpu")
    else:
        x = torch.load(f"{data_root}/imageryrf_single_trial/{subject}/single_trial_betas.pt").requires_grad_(False).to("cpu")
    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    y = torch.load(f"{data_root}/imageryrf_single_trial/stimuli/{stimtype}_images.pt").requires_grad_(False).to("cpu")
    # Find the stimuli indices conditioned on the mode of trials we want to load
    if split:
        conditionals_train = image_map[cond_idx[mode+'train']]
        conditionals_test = image_map[cond_idx[mode+'test']]
        x_train = x[cond_idx[mode+'train']]
        x_test = x[cond_idx[mode+'test']]
        y_train = y[~torch.isin(torch.arange(len(y)), torch.tensor(test_indices))]
        y_test = y[test_indices]
    else:
        conditionals = image_map[cond_idx[mode]]
        # Prune the beta file down to specific experimental mode/stimuli type
        x = x[cond_idx[mode]]
        
    # Average or nest the betas across trials
    if average or nest:
        if split:
            x_train, y_train, sample_count = condition_average(x_train, y_train, conditionals_train, nest=nest)
            x_test, y_test, sample_count = condition_average(x_test, y_test, conditionals_test, nest=nest)
        else:
            x, y, sample_count = condition_average(x, y, conditionals, nest=nest)
    else:
        if split:
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
            y_train = y[conditionals_train]
            y_test = y[conditionals_test]
            
        else:
            x = x.reshape((x.shape[0], x.shape[1]))
            y = y[conditionals]
    
    if split:
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        return x_train, y_train, x_test, y_test
    else:
        print(x.shape, y.shape)
        return x, y