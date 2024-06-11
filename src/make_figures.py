import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import utils
figure_dims = (16,8)
# voxels, all_images = utils.load_nsd_mental_imagery(subject=1, mode="vision", stimtype="all", average=True, nest=False, data_root="../dataset/")
_, _, _, all_images = utils.load_imageryrf(subject=1, mode="vision", stimtype="object", average=False, nest=False, split=True)
imsize = 150
if all_images.shape[-1] != imsize:
    all_images = transforms.Resize((imsize,imsize))(transforms.CenterCrop(all_images.shape[2])(all_images)).float()
for mode in ["imagery", "vision"]:
    for model_name in ["pretrained_subj09irf_40sess_hypatia_no_blurry_noirfpt_all", "pretrained_subj09irf_40sess_hypatia_no_blurry_noirfpt_vision", "pretrained_subj09irf_40sess_hypatia_no_blurry_noirfpt_imagery"]:
        all_recons = torch.load(f"evals/{model_name}/{model_name}_all_enhancedrecons_{mode}.pt")
        if all_recons.shape[-1] != imsize:
            all_recons = transforms.Resize((imsize, imsize))(transforms.CenterCrop(all_images.shape[2])(all_recons)).float()

        num_images = all_recons.shape[0]
        num_rows = (2 * num_images + 11) // 12

        # Interleave tensors
        merged = torch.stack([val for pair in zip(all_images, all_recons) for val in pair], dim=0)

        # Calculate grid size
        grid = torch.zeros((num_rows * 12, 3, all_recons.shape[-1], all_recons.shape[-1]))

        # Populate the grid
        grid[:2*num_images] = merged
        grid_images = [transforms.functional.to_pil_image(grid[i]) for i in range(num_rows * 12)]

        # Create the grid image
        grid_image = Image.new('RGB', (all_recons.shape[-1] * 12, all_recons.shape[-1] * num_rows))  # 12 images wide

        # Paste images into the grid
        for i, img in enumerate(grid_images):
            grid_image.paste(img, (all_recons.shape[-1] * (i % 12), all_recons.shape[-1] * (i // 12)))

       # Create title row image
        title_height = 150
        title_image = Image.new('RGB', (grid_image.width, title_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(title_image)
        font = ImageFont.truetype("arial.ttf", 38)  # Change font size to 3 times bigger (15*3)
        title_text = f"Model: {model_name}, Mode: {mode}"
        bbox = draw.textbbox((0, 0), title_text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((grid_image.width - text_width) / 2, (title_height - text_height) / 2), title_text, fill="black", font=font)

        # Combine title and grid images
        final_image = Image.new('RGB', (grid_image.width, grid_image.height + title_height))
        final_image.paste(title_image, (0, 0))
        final_image.paste(grid_image, (0, title_height))

        final_image.save(f"../figs/{model_name}_{len(all_recons)}recons_{mode}.png")
        print(f"saved ../figs/{model_name}_{len(all_recons)}recons_{mode}.png")

