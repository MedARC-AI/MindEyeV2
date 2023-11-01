import torch
import torch.nn.functional as F
import models
import os
import accelerate
from accelerate import Accelerator
from pathlib import Path
import numpy as np
import argparse
import random
import functools
import logging
import torch.nn as nn
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
import webdataset as wds
from collections import OrderedDict
from torchvision import transforms
from tqdm.auto import tqdm
import kornia
from kornia.augmentation.container import AugmentationSequential
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import shutil
import h5py
import boto3
import utils
import tarfile
from models import Voxel2StableDiffusionXL
from diffusers import AutoencoderKL
from convnext import ConvnextXL
from diffusers.utils import is_wandb_available

logger = get_logger(__name__)

if is_wandb_available():
    import wandb

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Brain Voxel MLP")
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="name of model, used for ckpt saving and wandb logging (if enabled)",
    )
    parser.add_argument(
    "--data_path", type=str, default="/fsx/proj-fmri/shared/natural-scenes-dataset/webdataset_avg_split",
    help="Path to where NSD data is stored / where to download it to",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="voxel_sdxlVae",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--s3_dir",
        type=str,
        default="/users/mihirneal/checkpoints/brainmlp",
        help="S3 directly where the checkpoints are offloaded.",
    )
    parser.add_argument(
        "--subj",type=int, default=1, choices=[1,2,5,7],
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size can be increased by 10x if only training v2c and not diffusion diffuser",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--resume_from_ckpt",action="store_true",
        help="if not using wandb and want to resume from a ckpt",
    )
    parser.add_argument(
        "--mixup_pct",type=float,default=.66,
        help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--use_cont",action="store_true",
        help="",
    )
    parser.add_argument(
        "--use_reconst",action="store_true",
        help="",
    )
    parser.add_argument(
        "--use_sobel_loss",action="store_true",
        help="",
    )
    parser.add_argument(
        "--cont_model", type=str, default="cnx",
        help="Type of ConvNext model to use",
    )
    parser.add_argument(
        "--ups_mode",type=str,default="4x",
        help="Upscaling factor for VAE embeds",
    )
    parser.add_argument(
        "--use_blurred_training",action="store_true",
        help="",
    )
    parser.add_argument(
        "--clip_scale",type=float,default=1.,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--use_image_aug",action="store_true",
        help="whether to use image augmentation",
    )
    parser.add_argument(
        "--num_epochs",type=int,default=12,
        help="number of epochs of training",
    )
    parser.add_argument(
        "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
    )
    parser.add_argument(
        "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
    )
    parser.add_argument(
        "--ckpt_interval",type=int,default=5,
        help="save backup ckpt and reconstruct every x epochs",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--best_val_loss", default=1e10, type=int, help="Highest validation loss limit")
    parser.add_argument("--best_ssim", default=0, type=int, help="Highest SSIM")
    parser.add_argument(
        "--mixed_precision",type=str,default="fp16",
        help="Half floats are faster in operation",
    )
    parser.add_argument(
        "--seed",type=int,default=42,
    )
    parser.add_argument(
        "--clip_seq_dim",type=int,default=257,
    )
    parser.add_argument(
        "--clip_emb_dim",type=int,default=768,
    )
    parser.add_argument(
        "--hidden_dim",type=int,default=4096,
    )
    parser.add_argument(
        "--start_lr",type=float,default=1e-3,
    )
    parser.add_argument(
        "--max_lr",type=float,default=5e-4,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cycle",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cycle, ""cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="SDVAE_Map",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for."
        ),
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help=(
            "Consider using Deepspeed if the dataset can't fit into a single GPU."
            "Deepspeed compatibility is built into acceelerator however, some changes are required which are activated by setting it to True."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def get_dataloaders(args):
    train_dir = [os.path.join(args.data_path, f"train/train_subj0{args.subj}_{i}.tar") for i in range(18)]  # 0 to 17
    val_dir = os.path.join(args.data_path, f"val/val_subj0{args.subj}_0.tar")  # Just one file here
    train_dir.append(val_dir)  # Adding the validation directory to the training list

    test_dir = [os.path.join(args.data_path, f"test/test_subj0{args.subj}_{i}.tar") for i in range(2)]  # 0 to 1
    meta_dir = os.path.join(args.data_path, f"metadata_subj0{args.subj}.json")
    logger.info("Prepping NSD Dataset")
    num_train = 8559 + 300
    num_val = 982

    train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
        args.train_batch_size,
        num_devices=torch.cuda.device_count(),
        num_workers=torch.cuda.device_count(),
        train_url=train_dir,
        val_url=test_dir,
        meta_url=meta_dir,
        val_batch_size=max(16, args.test_batch_size),
        cache_dir='./cache',
        seed=args.seed,
        voxels_key='nsdgeneral.npy',
        local_rank=0,
        num_train=num_train,
        num_val=num_val
    ) 
    return train_dl, val_dl, num_train, num_val


def create_tarball(source_dir, tarball_name):
    with tarfile.open(tarball_name, "w") as tar:  
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def offload_to_s3(args, source_dir, bucket, s3_path, global_step):
    tar_file_loc = os.path.join(args.output_dir, f"checkpoint-{global_step}.tar")
    create_tarball(source_dir, tar_file_loc)
    
    s3_client = boto3.client('s3')
    del_dir = os.path.join(args.output_dir, f"checkpoint-{global_step - args.checkpointing_steps}")
    try:
        response = s3_client.upload_file(tar_file_loc, bucket, s3_path)
    except FileNotFoundError:
        print("The tarball was not found")
    except NoCredentialsError:
        print("Credentials not available")
    else:
        if response:
            print("Offloaded to S3")
    
    try:
        os.remove(tar_file_loc)
        shutil.rmtree(del_dir)
        print(f"Directory {del_dir} deleted locally.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")

    return True

def get_dinov2_embeds(dinov2, image):
    with torch.no_grad():
        dv2_embeds = dinov2.forward_features(image)["x_norm_patchtokens"]
        return dv2_embeds


def main(args):
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    in_dims = {'1': 15724, '2': 14278, '5': 13039, '7':12682}
    sdxl_vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=weight_dtype).to(accelerator.device)
    sdxl_vae.requires_grad_(False)
    sdxl_vae.eval()

    if args.use_cont:
        args.mixup_pct = -1
        if args.cont_model == 'cnx':
            cnx = ConvnextXL('/fsx/proj-fmri/shared/models/convnextv2/convnext_xlarge_alpha0.75_fullckpt.pth')
            cnx.requires_grad_(False)
            cnx.eval()
            cnx.to(accelerator.device)
        train_augs = AugmentationSequential(
            # kornia.augmentation.RandomCrop((480, 480), p=0.3),
            # kornia.augmentation.Resize((512, 512)),
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.2),
            kornia.augmentation.RandomSolarize(p=0.2),
            kornia.augmentation.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=0.1),
            kornia.augmentation.RandomResizedCrop((512, 512), scale=(0.5, 1.0)),
            data_keys=["input"],
        )

    voxel2sdxl = Voxel2StableDiffusionXL(use_cont=args.use_cont, in_dim=in_dims[str(args.subj)], ups_mode=args.ups_mode)

    train_dl, val_dl, num_train, num_val = get_dataloaders(args)
   


    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
    {'params': [p for n, p in voxel2sdxl.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in voxel2sdxl.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.start_lr)
    
    num_devices = torch.cuda.device_count()
    total_steps=int(args.num_epochs*((num_train//args.train_batch_size)//num_devices))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, 
        pct_start=2/args.num_epochs,
    )
    

    voxel2sdxl, optimizer, train_dl, lr_scheduler = accelerator.prepare(voxel2sdxl, optimizer, train_dl, lr_scheduler)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


    epoch = 0
    losses, val_losses, lrs = [], [], []
    mean = torch.tensor([0.485, 0.456, 0.406]).to(accelerator.device).reshape(1,3,1,1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(accelerator.device).reshape(1,3,1,1)

    progress_bar = tqdm(range(epoch,args.num_epochs), desc="Epochs", disable=not accelerator.is_local_main_process)


    for epoch in progress_bar:
        voxel2sdxl.train()
        
        loss_mse_sum = 0
        loss_reconst_sum = 0
        loss_cont_sum = 0
        loss_sobel_sum = 0
        val_loss_mse_sum = 0
        val_loss_reconst_sum = 0
        val_ssim_score_sum = 0

        reconst_fails = []

        for train_i, (voxel, image, _) in enumerate(train_dl):
            optimizer.zero_grad()

            image = image.to(accelerator.device)
            image_512 = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)
            voxel = voxel.to(accelerator.device)
            voxel = utils.voxel_select(voxel)
            if epoch <= args.mixup_pct * args.num_epochs:
                voxel, perm, betas, select = utils.mixco(voxel)
            else:
                select = None

            with torch.cuda.amp.autocast(enabled=True):
                autoenc_image = kornia.filters.median_blur(image_512, (15, 15)) if args.use_blurred_training else image_512
                image_enc = sdxl_vae.encode(2*autoenc_image-1).latent_dist.sample() * 0.18215
                if args.use_cont:
                    image_enc_pred, transformer_feats = voxel2sdxl(voxel, return_transformer_feats=True)
                else:
                    image_enc_pred = voxel2sdxl(voxel)
               
                if epoch <= args.mixup_pct * args.num_epochs:
                    image_enc_shuf = image_enc[perm]
                    betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                    image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                        image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)
                
                if args.use_cont:
                    image_norm = (image_512 - mean)/std
                    image_aug = (train_augs(image_512) - mean)/std
                    _, cnx_embeds = cnx(image_norm)
                    _, cnx_aug_embeds = cnx(image_aug)
                    cont_loss = utils.soft_cont_loss(
                        F.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        F.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        F.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        temp=0.075,
                        distributed=False
                    )
                    del image_aug, cnx_embeds, transformer_feats
                else:
                    cont_loss = torch.tensor(0)

                # mse_loss = F.mse_loss(image_enc_pred, image_enc)/0.18215
                mse_loss = F.l1_loss(image_enc_pred, image_enc)
                del image_512, voxel

                if args.use_reconst: #epoch >= 0.1 * num_epochs:
                    # decode only non-mixed images
                    if select is not None:
                        selected_inds = torch.where(~select)[0]
                        reconst_select = selected_inds[torch.randperm(len(selected_inds))][:4] 
                    else:
                        reconst_select = torch.arange(len(image_enc_pred))
                    image_enc_pred = F.interpolate(image_enc_pred[reconst_select], scale_factor=0.5, mode='bilinear', align_corners=False)
                    reconst = sdxl_vae.decode(image_enc_pred/0.18215).sample
                    # reconst_loss = F.mse_loss(reconst, 2*image[reconst_select]-1)
                    reconst_image = kornia.filters.median_blur(image[reconst_select], (7, 7)) if use_blurred_training else image[reconst_select]
                    reconst_loss = F.l1_loss(reconst, 2*reconst_image-1)
                    if reconst_loss != reconst_loss:
                        reconst_loss = torch.tensor(0)
                        reconst_fails.append(train_i) 
                    if args.use_sobel_loss:
                        sobel_targ = kornia.filters.sobel(kornia.filters.median_blur(image[reconst_select], (3,3)))
                        sobel_pred = kornia.filters.sobel(reconst/2 + 0.5)
                        sobel_loss = F.l1_loss(sobel_pred, sobel_targ)
                    else:
                        sobel_loss = torch.tensor(0)
                else:
                    reconst_loss = torch.tensor(0)
                    sobel_loss = torch.tensor(0)
                

                loss = mse_loss/0.18215 + 2*reconst_loss + 0.1*cont_loss + 16*sobel_loss
                # utils.check_loss(loss)

                loss_mse_sum += mse_loss.item()
                loss_reconst_sum += reconst_loss.item()
                loss_cont_sum += cont_loss.item()
                loss_sobel_sum += sobel_loss.item()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                if accelerator.is_main_process:
                    logs = OrderedDict(
                        train_loss=np.mean(losses[-(train_i+1):]),
                        val_loss=np.nan,
                        lr=lrs[-1],
                    )
                    # progress_bar.set_postfix(**logs)
            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

        if accelerator.is_main_process: 
            voxel2sdxl.eval()
            for val_i, (voxel, image, _) in enumerate(val_dl): 
                with torch.inference_mode():
                    image = image.to(accelerator.device)
                    image = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)              
                    voxel = voxel.to(accelerator.device)
                    voxel = voxel.mean(1)
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        image_enc = sdxl_vae.encode(2*image-1).latent_dist.sample() * 0.18215
                        image_enc_pred, _ = voxel2sdxl(voxel)
                        # if hasattr(voxel2sdxl, 'module'):
                        #     image_enc_pred = voxel2sdxl.module(voxel)
                        # else:
                        #     image_enc_pred = voxel2sdxl(voxel)
                        print("image_enc_pred: ", image_enc_pred.shape)
                        print("image_enc: ", image_enc.shape)
                        mse_loss = F.mse_loss(image_enc_pred, image_enc)
                        
                        if args.use_reconst:
                            reconst = sdxl_vae.decode(image_enc_pred[-16:]/0.18215).sample
                            image = image[-16:]
                            reconst_loss = F.mse_loss(reconst, 2*image-1)
                            ssim_score = ssim((reconst/2 + 0.5).clamp(0,1), image, data_range=1, size_average=True, nonnegative_ssim=True)
                        else:
                            reconst = None
                            reconst_loss = torch.tensor(0)
                            ssim_score = torch.tensor(0)

                        val_loss_mse_sum += mse_loss.item()
                        val_loss_reconst_sum += reconst_loss.item()
                        val_ssim_score_sum += ssim_score.item()
                              
                        val_losses.append(mse_loss.item() + reconst_loss.item())        
                        utils.check_loss(val_losses)
                logs = OrderedDict(
                    train_loss=np.mean(losses[-(train_i+1):]),
                    val_loss=np.mean(val_losses[-(val_i+1):]),
                    lr=lrs[-1],
                )
                # progress_bar.set_postfix(**logs)

            # if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
            #     # save best model
            #     val_loss = np.mean(val_losses[-(val_i+1):])
            #     val_ssim = val_ssim_score_sum / (val_i + 1)
            #     if val_loss < best_val_loss:
            #         best_val_loss = val_loss
            #         save_ckpt('best')
            #     else:
            #         print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')
            #     if val_ssim > best_ssim:
            #         best_ssim = val_ssim
            #         save_ckpt('best_ssim')
            #     else:
            #         print(f'not best - val_ssim: {val_ssim:.3f}, best_ssim: {best_ssim:.3f}')

            #     save_ckpt('last')
            #     # Save model checkpoint every `ckpt_interval`` epochs or on the last epoch
            #     if (ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0) or epoch == num_epochs - 1:
            #         save_ckpt(f'epoch{(epoch+1):03d}')
            #     try:
            #         orig = image
            #         if reconst is None:
            #             reconst = autoenc.decode(image_enc_pred[-16:].detach()/0.18215).sample
            #             orig = image[-16:]
            #         pred_grid = make_grid(((reconst/2 + 0.5).clamp(0,1)*255).byte(), nrow=int(len(reconst)**0.5)).permute(1,2,0).cpu().numpy()
            #         orig_grid = make_grid((orig*255).byte(), nrow=int(len(orig)**0.5)).permute(1,2,0).cpu().numpy()
            #         comb_grid = np.concatenate([orig_grid, pred_grid], axis=1)
            #         del pred_grid, orig_grid
            #         Image.fromarray(comb_grid).save(f'{outdir}/reconst_epoch{(epoch+1):03d}.png')
            #     except:
            #         print("Failed to save reconst image")
            #         print(traceback.format_exc())

            logs = {
                "train/loss": np.mean(losses[-(train_i+1):]),
                "val/loss": np.mean(val_losses[-(val_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "train/loss_mse": loss_mse_sum / (train_i + 1),
                "train/loss_reconst": loss_reconst_sum / (train_i + 1),
                "train/loss_cont": loss_cont_sum / (train_i + 1),
                "train/loss_sobel": loss_sobel_sum / (train_i + 1),
                "val/loss_mse": val_loss_mse_sum / (val_i + 1),
                "val/loss_reconst": val_loss_reconst_sum / (val_i + 1),
                "val/ssim": val_ssim_score_sum / (val_i + 1),
            }
            if accelerator.is_main_process:
                accelerator.log(logs, step=epoch)

            if len(reconst_fails) > 0 and accelerator.is_main_process:
                print(f'Reconst fails {len(reconst_fails)}/{train_i}: {reconst_fails}')


    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.end_training()
    if accelerator.is_main_process:
        accelerator.save_model(voxel2sdxl, args.output_dir, safe_serialization=True)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
