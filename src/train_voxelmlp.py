import torch
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
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt
import shutil
import h5py
import boto3
import utils
import tarfile
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
    "--data_path", type=str, default="/fsx/proj-fmri/shared/natural-scenes-dataset",
    help="Path to where NSD data is stored / where to download it to",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="brainmlp-model",
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
        "--blurry_recon",action="store_true",
        help="whether to output blurry reconstructions",
    )
    parser.add_argument(
        "--depth_recon",action="store_true",
        help="whether to output depth reconstructions",
    )
    parser.add_argument(
        "--blur_scale",type=float,default=100.,
        help="multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--depth_scale",type=float,default=100.,
        help="multiply loss from depth recons by this number",
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
        "--train_batch_size", type=int, default=128, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=2770, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--best_test_loss", default=1e9, type=int, help="Highest loss possible")
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
        "--max_lr",type=float,default=3e-4,
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
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
        default="BrainMLP_DINOV2",
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


def get_train_dataset(args):
    def my_split_by_node(urls): return urls
    train_dir = os.path.join(args.data_path, "wds/subj0" + str(args.subj), "train/") 
    test_dir = os.path.join(args.data_path, "wds/subj0" + str(args.subj), "test/") 
    train_dataset_url = train_dir + "{0..36}.tar"
    test_url = test_dir + "0.tar"

    train_data = wds.WebDataset(train_dataset_url,resampled=False,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    

    test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    
    hdf_vox_path = os.path.join(args.data_path, "betas_all_subj0" + str(args.subj) + ".hdf5")
    hdf_vox_file = h5py.File(hdf_vox_path, 'r')
    voxels = hdf_vox_file['betas'][:]
    logger.info(f"subj0{args.subj} betas loaded into memory")
    voxels = torch.Tensor(voxels).to("cpu").to(torch.float16)
    print("voxels", voxels.shape)
    num_voxels = voxels.shape[-1]

    hdf_img_path = os.path.join(args.data_path, "coco_images_224_float16.hdf5")
    hdf_img_file = h5py.File(hdf_img_path, 'r')
    images = hdf_img_file['images'][:]
    images = torch.Tensor(images).to("cpu").to(torch.float16)
    print("images", images.shape)

    return train_data, test_data, voxels, images


def augment_img(args, image):
    if args.use_image_aug:
        import kornia
        from kornia.augmentation.container import AugmentationSequential
        img_augment = AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((224,224), (0.6,1), p=0.3),
            kornia.augmentation.Resize((224, 224)),
            kornia.augmentation.RandomHorizontalFlip(p=0.3),
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
            kornia.augmentation.RandomGrayscale(p=0.3),
            same_on_batch=False,
            data_keys=["input"],
        )
        aug_img = img_augment(image)

    return aug_img


def check_dataloaders(args, train_dl, test_dl):
    test_vox_indices = []
    test_73k_images = []
    for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
        test_vox_indices = np.append(test_vox_indices, behav[:,0,5].cpu().numpy())
        test_73k_images = np.append(test_73k_images, behav[:,0,0].cpu().numpy())
    test_vox_indices = test_vox_indices.astype(np.int16)
    print(test_i, (test_i+1) * args.test_batch_size, len(test_vox_indices))

    train_vox_indices = []
    train_73k_images = []
    for train_i, (behav, past_behav, future_behav, old_behav) in enumerate(train_dl):
        train_vox_indices = np.append(train_vox_indices, behav[:,0,5].long().cpu().numpy())
        train_73k_images = np.append(train_73k_images, behav[:,0,0].cpu().numpy())
    train_vox_indices = train_vox_indices.astype(np.int16)
    print(train_i, (train_i+1) * args.train_batch_size, len(train_vox_indices))

    all_vox_indices = np.hstack((train_vox_indices, test_vox_indices))
    all_images = np.hstack((train_73k_images, test_73k_images))

    return all_vox_indices, all_images

def add_saturation(image, alpha=2):
    gray_image = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]
    gray_image = gray_image.unsqueeze(1).expand_as(image)
    saturated_image = alpha * image + (1 - alpha) * gray_image
    return torch.clamp(saturated_image, 0, 1)

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
    if args.blurry_recon:# or depth_recon:
        from diffusers import VQModel
        autoenc = VQModel.from_pretrained("/fsx/proj-fmri/shared/cache/models--microsoft--vq-diffusion-ithq/snapshots/3f796fb49ee559370dc638dea1d8116af131d993/vqvae", torch_dtype=weight_dtype)
        autoenc.eval()
        autoenc.requires_grad_(False)
        autoenc.to(accelerator.device)
        utils.count_params(autoenc)


    train_data, test_data, voxels, images = get_train_dataset(args)
    
    model = models.MindEyeModule()
    accelerator.unwrap_model(model).ridge = models.RidgeRegression(voxels.shape[1], out_features=args.hidden_dim)
    accelerator.unwrap_model(model).backbone = models.BrainNetwork(args, h=args.hidden_dim, in_dim=args.hidden_dim, clip_size=1536, out_dim=1536*256, blurry_dim=64*7*7)
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(weight_dtype).to(accelerator.device)

    if args.depth_recon:
        from controlnet_aux.midas import MidasDetector
        
        midas_depth = MidasDetector.from_pretrained(
        "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large", cache_dir="/fsx/proj-fmri/shared/cache").to(accelerator.device)
        midas_depth.model.eval()
        midas_depth.model.requires_grad_(False)
        midas_depth.model.to(accelerator.device)
        pass

    if args.depth_recon:
        # if utils.is_interactive(): display(utils.torch_to_Image(images[[30]]))

        input_batch = images[[30,31]].float().to(accelerator.device)
        accelerator.print(input_batch.shape)
        
        midas_emb = midas_depth.model(input_batch).unsqueeze(1)
        accelerator.print(midas_emb.shape)

        prediction = utils.resize(midas_emb, 32) #/30).clamp(0,1).half() # 30 is roughly prediction.max()
        accelerator.print(prediction.shape)
        
        prediction = (prediction / prediction.view(prediction.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1).expand_as(prediction))
        midas_emb_size = prediction.flatten(1).shape[1]
        accelerator.print("midas_emb", prediction.shape, prediction.min(), prediction.max())
        accelerator.print("midas_emb_size", midas_emb_size)
    
    # if utils.is_interactive(): display(utils.torch_to_Image(utils.resize(prediction, 224))) 

    # if args.blurry_recon:
    #     prediction = utils.resize(midas_emb, 128).repeat(1,3,1,1)
    #     prediction = (prediction / prediction.view(prediction.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1).expand_as(prediction))
    #     prediction_enc = autoenc.encode(2*prediction-1).latents * 0.18215
    #     print("vae midas_emb", prediction_enc.shape, prediction_enc.min(), prediction_enc.max())
    
        # if utils.is_interactive(): display(utils.torch_to_Image((autoenc.decode(prediction_enc/0.18215).sample / 2 + 0.5).clamp(0,1)))

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    # if args.use_8bit_adam:
    #     try:
    #         import bitsandbytes as bnb
    #     except ImportError:
    #         raise ImportError(
    #             "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
    #         )

    #     optimizer_class = bnb.optim.AdamW8bit
    # else:
    #     optimizer_class = torch.optim.AdamW

    # if args.use_deepspeed:
    #     optimizer = accelerate.utils.DummyOptim(
    #             opt_grouped_parameters,
    #             lr=args.max_lr,
    #             betas=(args.adam_beta1, args.adam_beta2),
    #             weight_decay=args.adam_weight_decay,
    #             eps=args.adam_epsilon,
    #         )
    # else:
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)
    
    
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=False, drop_last=True, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, drop_last=True, pin_memory=True)

    # if args.use_deepspeed:
    #     total_steps=int(np.floor(args.num_epochs*(24958/accelerator.num_processes/args.train_batch_size)))
    #     lr_scheduler = accelerate.utils.DummyScheduler(
    #             optimizer, total_num_steps=total_steps, final_div_factor=1000, last_epoch=-1, pct_start=2/args.num_epochs
    #         )
    # else:
    num_devices = torch.cuda.device_count()
    total_steps=int(np.floor(args.num_epochs*(24958/num_devices/args.train_batch_size)))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, 
        pct_start=2/args.num_epochs,
        # steps_per_epoch=int(total_steps / args.num_epochs),
        # epochs=args.num_epochs
        )
    

    model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


    epoch = 0
    losses, test_losses, lrs = [], [], []
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))

    progress_bar = tqdm(range(epoch,args.num_epochs), desc="Epochs", disable=not accelerator.is_local_main_process)
    test_image, test_voxel = None, None
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    # from models import Clipper
    # clip_model = Clipper("ViT-L/14", device=accelerator.device, hidden_state=True, norm_embs=True)
    # clip_seq_dim = 257

    for epoch in progress_bar:
        model.train()
        
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        test_fwd_percent_correct = 0.
        test_bwd_percent_correct = 0.

        loss_clip_total = 0.
        loss_blurry_total = 0.
        loss_depth_total = 0.
        test_loss_clip_total = 0.
        test_loss_blurry_total = 0.
        test_loss_depth_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1
        
        for train_i, (behav, past_behav, future_behav, old_behav) in enumerate(train_dl):
            with torch.cuda.amp.autocast(dtype=weight_dtype):
                optimizer.zero_grad()
        
                voxel = voxels[behav[:,0,5].cpu().long()].to(accelerator.device)
                
                image = images[behav[:,0,0].cpu().long()].to(accelerator.device).float()
        
                if args.blurry_recon:
                    # blurry_image_enc = autoenc.encode(2*utils.resize(image,128)-1).latent_dist.mode() * 0.18215
                    blurry_image_enc = autoenc.encode(2*utils.resize(add_saturation(image),128)-1).latents * 0.18215

                if args.depth_recon:
                    # depth_images = utils.resize(midas_depth.model(image).unsqueeze(1).repeat(1,3,1,1), 128)
                    depth_images = utils.resize(midas_depth.model(image).unsqueeze(1), 32)
                    depth_images = (depth_images / depth_images.view(depth_images.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1).expand_as(depth_images))
                    depth_image_enc = depth_images # autoenc.encode(2*depth_images-1).latents * 0.18215
                
                if args.use_image_aug: 
                    image = augment_img(args, image)
        
                clip_target = get_dinov2_embeds(dinov2=dinov2_model, image=image)
                assert not torch.any(torch.isnan(clip_target))
        
                if epoch < int(args.mixup_pct * args.num_epochs):
                    voxel, perm, betas, select = utils.mixco(voxel)

                model = accelerator.unwrap_model(model)
                voxel_ridge = model.ridge(voxel)
        
                clip_voxels, blurry_image_enc_, depth_image_enc_ = model.backbone(voxel_ridge)
                # accelerator.print(clip_voxels.shape)
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                if epoch < int(args.mixup_pct * args.num_epochs):                
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006, 
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(args.mixup_pct*args.num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                loss_clip_total += loss_clip.item()
                loss_clip *= args.clip_scale
                loss = loss_clip
        
                if args.blurry_recon:
                    downsampled_image = nn.functional.interpolate(image, size=(8, 8), mode='bilinear', align_corners=False)
                    re_upsampled_image = add_saturation(nn.functional.interpolate(downsampled_image, size=(128, 128), mode='nearest'))
                    re_upsampled_enc = autoenc.encode(2*re_upsampled_image-1).latents * 0.18215
                    
                    loss_blurry = (l1(blurry_image_enc_, blurry_image_enc) + l1(blurry_image_enc_, re_upsampled_enc))
                    loss_blurry += l1(torch.var(blurry_image_enc), torch.var(blurry_image_enc_))
                    loss_blurry_total += loss_blurry.item()
                    loss_blurry *= args.blur_scale
                    loss += loss_blurry

                if args.depth_recon:
                    loss_depth = l1(depth_image_enc_, depth_image_enc)
                    # loss_depth += l1(torch.var(depth_image_enc_), torch.var(depth_image_enc))
                    loss_depth_total += loss_depth.item()
                    loss_depth *= args.depth_scale
                    loss += loss_depth
        
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_target_norm)).to(clip_voxels_norm.device) 
                fwd_percent_correct += utils.topk(torch.abs(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm)), labels, k=1).item()
                bwd_percent_correct += utils.topk(torch.abs(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm)), labels, k=1).item()
        
                if args.blurry_recon:
                    # with torch.no_grad():
                    # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                    random_samps = np.random.choice(np.arange(len(voxel)), size=args.batch_size//5, replace=False)
                    # random_samps = np.arange(batch_size//5)
                    blurry_recon_images = (autoenc.decode(blurry_image_enc_[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                    # pixcorr_origsize_nanmean is computationally less intense than utils.pixcorr and uses nanmean instead of mean
                    pixcorr = utils.pixcorr_origsize_nanmean(image[random_samps], blurry_recon_images)
                    # pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    loss += (1 - pixcorr)
                    blurry_pixcorr += pixcorr.item()
                    utils.check_loss(pixcorr)

                utils.check_loss(loss)
                accelerator.backward(loss)
                optimizer.step()
        
                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])
        
                lr_scheduler.step()

        model.eval()
        if accelerator.is_main_process:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype): 
                for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                    # all test samples should be loaded per batch such that test_i should never exceed 0
                    # assert len(behav) == num_test
                    assert len(behav) == 2770
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
                            else:
                                test_image = torch.vstack((test_image, images[im][None]))
                                test_voxel = torch.vstack((test_voxel, torch.mean(voxel[locs],axis=0)[None]))
        
                    # random sample of 300
                    random_indices = torch.arange(len(test_voxel))[:300]
                    voxel = test_voxel[random_indices].to(accelerator.device)
                    image = test_image[random_indices].to(accelerator.device)
                    assert len(image) == 300

                    if args.blurry_recon:
                        # blurry_image_enc = autoenc.encode(2*utils.resize(image,128)-1).latent_dist.mode() * 0.18215
                        blurry_image_enc = autoenc.encode(2*utils.resize(add_saturation(image),128)-1).latents * 0.18215

                    if args.depth_recon:
                        # depth_images = utils.resize(midas_depth.model(image).unsqueeze(1).repeat(1,3,1,1), 128)
                        depth_images = utils.resize(midas_depth.model(image).unsqueeze(1), 32)
                        depth_images = (depth_images / depth_images.view(depth_images.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1).expand_as(depth_images))
                        depth_image_enc = depth_images # autoenc.encode(2*depth_images-1).latents * 0.18215
                
                    image = image.to(accelerator.device)
                    voxel = voxel.to(accelerator.device)
                    clip_target = get_dinov2_embeds(dinov2=dinov2_model, image=image.float())
                    model = accelerator.unwrap_model(model)
                    voxel_ridge = model.ridge(voxel)
                    
                    clip_voxels, blurry_image_enc_, depth_image_enc_ = model.backbone(voxel_ridge)
                    
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
            
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006)
                    test_loss_clip_total += loss_clip.item()
                    loss_clip = loss_clip * args.clip_scale
                    loss = loss_clip

                    if args.blurry_recon:
                        downsampled_image = nn.functional.interpolate(image, size=(8, 8), mode='bilinear', align_corners=False)
                        re_upsampled_image = add_saturation(nn.functional.interpolate(downsampled_image, size=(128, 128), mode='nearest'))
                        re_upsampled_enc = autoenc.encode(2*re_upsampled_image-1).latents * 0.18215
                        
                        loss_blurry = (l1(blurry_image_enc_, blurry_image_enc) + l1(blurry_image_enc_, re_upsampled_enc))
                        loss_blurry += l1(torch.var(blurry_image_enc), torch.var(blurry_image_enc_))
                        test_loss_blurry_total += loss_blurry.item()
                        loss_blurry *= args.blur_scale
                        loss += loss_blurry
        
                        # halving the batch size because the decoder is computationally heavy
                        blurry_recon_images = (autoenc.decode(blurry_image_enc_[:len(voxel)//2]/0.18215).sample / 2 + 0.5).clamp(0,1)
                        blurry_recon_images = torch.vstack((blurry_recon_images, (autoenc.decode(blurry_image_enc_[len(voxel)//2:]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        pixcorr = utils.pixcorr(image, blurry_recon_images)
                        loss += (1 - pixcorr)
                        test_blurry_pixcorr += pixcorr.item()

                    if args.depth_recon:
                        loss_depth = l1(depth_image_enc_, depth_image_enc)
                        # loss_depth += l1(torch.var(depth_image_enc_), torch.var(depth_image_enc))
                        test_loss_depth_total += loss_depth.item()
                        loss_depth *= args.depth_scale
                        loss += loss_depth
            
                    # forward and backward top 1 accuracy        
                    labels = torch.arange(len(clip_target_norm)).to(clip_voxels_norm.device) 
                    test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                    utils.check_loss(loss)                
                    test_losses.append(loss.item())

                # if utils.is_interactive(): clear_output(wait=True)
                
                assert (test_i+1) == 1
                logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "test/loss": np.mean(test_losses[-(test_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "test/num_steps": len(test_losses),
                    "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                    "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                    "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                    "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                    "train/loss_clip_total": loss_clip_total / (train_i + 1),
                    "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                    "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                    "test/loss_blurry_total": test_loss_blurry_total / (test_i + 1),
                    "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                    "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                    "train/loss_depth_total": loss_depth_total / (train_i + 1),
                    "test/loss_depth_total": test_loss_depth_total / (test_i + 1),
                    }
                accelerator.log(logs, step=epoch)
        
                if args.blurry_recon:    
                    # transform blurry recon latents to images and plot it
                    fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                    jj=-1
                    for j in [0,1,2,3]:
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(blurry_image_enc[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(blurry_image_enc_[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')
                    
                    if is_wandb_available():
                        logs[f"test/recons"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
                        plt.close()
                    else:
                        plt.show()

                if args.depth_recon:
                    # transform blurry recon latents to images and plot it
                    fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                    # axes[0].imshow(utils.torch_to_Image((autoenc.decode(depth_image_enc[[0]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                    # axes[1].imshow(utils.torch_to_Image((autoenc.decode(depth_image_enc_[[0]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                    jj=-1
                    for j in [0,1,2,3]:
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image(utils.resize(depth_image_enc[[j]].view(1,1,32,32).clamp(0,1), 224)))
                        axes[jj].axis('off')
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image(utils.resize(depth_image_enc_[[j]].view(1,1,32,32).clamp(0,1), 224)))
                        axes[jj].axis('off')
                    if is_wandb_available():
                        logs[f"test/depth_recons"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
                        plt.close()
                    else:
                        plt.show()
                
                # progress_bar.set_postfix(**logs)
        
                # # Save model checkpoint and reconstruct
                # if epoch % ckpt_interval == 0:
                #     if not utils.is_interactive():
                #         save_ckpt(f'last')
                        
                # if wandb_log: wandb.log(logs)

        
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.end_training()
    if accelerator.is_main_process:
        accelerator.save_model(model, args.output_dir, safe_serialization=True)
        # gc.collect()

if __name__ == "__main__":
    args = parse_args()
    main(args)
