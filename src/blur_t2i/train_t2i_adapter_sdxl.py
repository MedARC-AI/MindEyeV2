#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import tarfile
import boto3
import h5py
import accelerate
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    T2IAdapter,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import sys
sys.path.append('./generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf
import utils


MAX_SEQ_LENGTH = 77

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def convert_path(emb_path):
    base_dir = os.path.dirname(os.path.dirname(emb_path))
    img_no = os.path.basename(emb_path).split('dvt')[1].split('.')[0]
    new_path = os.path.join(base_dir, 'target', f'img_t{img_no}.jpg')

    return new_path
    
def process_image(pil_img):
    # Convert to RGB
    rgb_image = pil_img.convert("RGB")

    # Resize image
    resize = transforms.Resize((768, 768))
    resized_image = resize(rgb_image)

    # Convert to tensor
    to_tensor = transforms.ToTensor()
    tensor_image = to_tensor(resized_image)

    return tensor_image

def log_validation(diffusion_engine, t2iadapter_extn, clip_model, vector_suffix, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    adapter = accelerator.unwrap_model(t2iadapter_extn)
    
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        
    image_logs = []
    for image, validation_image in zip(images, validation_images):
        validation_image = Image.open(validation_image)
        image = Image.open(image)
        proc_val_img = process_image(validation_image)
        proc_img = process_image(image)
        img_ten = clip_model(proc_img.unsqueeze(0).to(device))

        adap_vec = t2iadapter_extn(proc_img)


        # print(prep_emb.shape)
        images = []

        for _ in range(args.num_validation_images):
            
            img_samples, _, _ = utils.unclip_recon(img_ten,
                             diffusion_engine,
                             vector_suffix, adapter_vectors=y)
            images.append(img_samples)

        image_logs.append(
            {"validation_image": validation_image, "images": images}
        )

    if args.report_to == "wandb":
        formatted_images = []

        for log in image_logs:
            images = log["images"]
            validation_prompt = log["validation_prompt"]

            formatted_images.append(wandb.Image(validation_image, caption="T2I segmented conditioning"))

            for image in images:
                image = wandb.Image(image)
                formatted_images.append(image)

        accelerator.log({"validation": formatted_images})
    else:
        logger.warn(f"image logging not implemented for {args.report_to}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="t2iadapter-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--detection_resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=("Number of subprocesses to use for data loading."),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
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
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="target", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="source",
        help="The column of the dataset containing the adapter conditioning image.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=20,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the t2iadapter conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="unclip_t2iadapter",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
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

    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    # if args.dataset_name is not None and args.train_data_dir is not None:
    #     raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    # if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
    #     raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    # if args.validation_prompt is not None and args.validation_image is None:
    #     raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    # if args.validation_prompt is None and args.validation_image is not None:
    #     raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    # if (
    #     args.validation_image is not None
    #     and args.validation_prompt is not None
    #     and len(args.validation_image) != 1
    #     and len(args.validation_prompt) != 1
    #     and len(args.validation_image) != len(args.validation_prompt)
    # ):
    #     raise ValueError(
    #         "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
    #         " or the same number of `--validation_prompt`s and `--validation_image`s"
    #     )

    # if args.resolution % 8 != 0:
    #     raise ValueError(
    #         "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the t2iadapter encoder."
    #     )

    return args





def get_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset



def prepare_train_dataset(dataset, clip_model, accelerator):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize((768, 768), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[args.image_column]]
        images = [image_transforms(image) for image in images]
        clip_images = [clip_model(image.unsqueeze(0)) for image in images]

        # conditioning_images = [image.convert("RGB").resize((8,8)).resize((args.resolution, args.resolution), resample=Image.Resampling.NEAREST) for image in examples[args.conditioning_image_column]]
        conditioning_images = [image.convert("RGB") for image in examples[args.conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["clip_values"] = clip_images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    clip_values = torch.stack([example["clip_values"] for example in examples])
    clip_values = clip_values.squeeze(1)
    clip_values = clip_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()


    return {
        "pixel_values": pixel_values,
        "clip_values": clip_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        # "prompt_ids": prompt_ids,
        # "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }

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



class downSampleExtn(nn.Module):
    def __init__(self):
        super(downSampleExtn, self).__init__()
        self.conv2 = nn.Conv2d(in_channels=640, out_channels=640, kernel_size=3, stride=1, padding=1)
        self.downsample2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Reduce to [1, 640, 48, 48]

        self.conv3 = nn.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, stride=1, padding=1)
        self.downsample3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Reduce to [1, 1280, 24, 24]

        self.conv4 = nn.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, stride=1, padding=1)
        self.downsample4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Reduce to [1, 1280, 24, 24]

    def forward(self, x2, x3, x4):
        x2 = self.downsample2(self.conv2(x2))
        x3 = self.downsample3(self.conv3(x3))
        x4 = self.downsample4(self.conv4(x4))
        return x2, x3, x4



class T2IAdapterExtn(nn.Module):
    def __init__(self, t2iadapterXL, extn):
        super(T2IAdapterExtn, self).__init__()
        self.t2iadapterXL = t2iadapterXL
        self.extn = extn

    def forward(self, x):
        ten_arr = self.t2iadapterXL(x)
        x1, x2, x3 = self.extn(ten_arr[1], ten_arr[2], ten_arr[3])
        unet_residuals = [ten_arr[0], x1, x2, x3]
        return unet_residuals
        

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    device = accelerator.device
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    logger.info("Initializing t2iadapter weights.")
    t2iadapter = T2IAdapter(
            in_channels=3,
            channels=(320, 640, 1280, 1280),
            num_res_blocks=2,
            downscale_factor=8,
            adapter_type="full_adapter_xl",
        )
    t2iadapter.train()
    downSample = downSampleExtn()
    downSample.train()
    t2iadapter_extn = T2IAdapterExtn(t2iadapter, downSample)
    f = h5py.File(f'/weka/proj-fmri/shared/mindeyev2_dataset/coco_images_224_float16.hdf5', 'r')
    images = f['images'][:1]
    images = torch.Tensor(images).to("cpu").to(weight_dtype)

    t2iadapter_extn.requires_grad_(True)
    t2iadapter_extn.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # if accelerator.unwrap_model(t2iadapter_extn).dtype != torch.float32:
    #     raise ValueError(
    #         f"Controlnet loaded as datatype {accelerator.unwrap_model(t2iadapter_extn).dtype}. {low_precision_error_string}"
    #     )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = t2iadapter_extn.parameters()


    optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to("cpu")

    #unCLIP setup
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
    clip_img_embedder.requires_grad_(False)
    diffusion_engine.train().requires_grad_(True)
    
    ckpt_path = '/weka/proj-fmri/shared/cache/sdxl_unclip/unclip6_epoch0_step110000.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])
    
    image = images[:1].to(device)
    batch={"jpg": image,
                  "original_size_as_tuple": torch.ones(image.shape[0], 2).to(device) * image.shape[-1],
                  "crop_coords_top_left": torch.zeros(image.shape[0], 2).to(device)}
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    
    # print("vector_suffix", vector_suffix.shape)

    #unCLIP setup

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    # def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
    #     original_size = (args.resolution, args.resolution)
    #     target_size = (args.resolution, args.resolution)
    #     crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
    #     img_batch = batch[args.image_column]

    #     prompt_embeds, pooled_prompt_embeds = encode_prompt(
    #         prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
    #     )
    #     add_text_embeds = pooled_prompt_embeds

    #     # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    #     add_time_ids = list(original_size + crops_coords_top_left + target_size)
    #     add_time_ids = torch.tensor([add_time_ids])

    #     prompt_embeds = prompt_embeds.to(accelerator.device)
    #     add_text_embeds = add_text_embeds.to(accelerator.device)
    #     add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
    #     add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
    #     unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    #     return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    # def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    #     sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    #     schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    #     timesteps = timesteps.to(accelerator.device)

    #     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma

    # Let's first compute all the embeddings so that we can free up the text encoders
    # # from memory.
    # text_encoders = [text_encoder_one, text_encoder_two]
    # tokenizers = [tokenizer_one, tokenizer_two]
    train_dataset = get_train_dataset(args, accelerator)
    # compute_embeddings_fn = functools.partial(
    #     compute_embeddings,
    #     proportion_empty_prompts=args.proportion_empty_prompts,
    #     text_encoders=text_encoders,
    #     tokenizers=tokenizers,
    # )
    # with accelerator.main_process_first():
    #     from datasets.fingerprint import Hasher

    #     # fingerprint used by the cache for the other processes to load the result
    #     # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
    #     new_fingerprint = Hasher.hash(args)
    #     train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)

    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(train_dataset, clip_img_embedder, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`.
    t2iadapter_extn, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        t2iadapter_extn, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        # tracker_config.pop("validation_prompt")
        # tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint") and os.path.isdir(os.path.join(args.output_dir, d))]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(t2iadapter_extn):
                if True:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                    clip_values = batch["clip_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]
                    clip_values = batch["clip_values"]

                # # encode pixel values with batch size of at most 8 to avoid OOM
                # latents = []
                # for i in range(0, pixel_values.shape[0], 8):
                #     latents.append(pixel_values[i : i + 8])
                # latents = torch.cat(latents, dim=0)
                # latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                # noise = torch.randn_like(latents)
                # bsz = latents.shape[0]

                # # Cubic sampling to sample a random timestep for each image.
                # # For more details about why cubic sampling is used, refer to section 3.4 of https://arxiv.org/abs/2302.08453
                # timesteps = torch.rand((bsz,), device=latents.device)
                # timesteps = (1 - timesteps**3) * noise_scheduler.config.num_train_timesteps
                # timesteps = timesteps.long().to(noise_scheduler.timesteps.dtype)
                # timesteps = timesteps.clamp(0, noise_scheduler.config.num_train_timesteps - 1)

                # # Add noise to the latents according to the noise magnitude at each timestep
                # # (this is the forward diffusion process)
                # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # # Scale the noisy latents for the UNet
                # sigmas = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                # inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Adapter conditioning.
                t2iadapter_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                # # pixel_values = pixel_values.squeeze(1)
                # print("t2I image: ", t2iadapter_image.shape)
                # print("clip image: ", clip_values.shape)
                # print("target image: ", pixel_values.shape)
                #print(t2iadapter_image.shape)

                    
                img_recon = []
                sigmas = []
                denoised_latents = []

                for i in range(clip_values.shape[0]):
                    adapter_vectors = t2iadapter_extn(t2iadapter_image[i])
                    x, y, z = utils.unclip_recon(clip_values[i].unsqueeze(0),
                                 diffusion_engine,
                                 vector_suffix, adapter_vectors=adapter_vectors)
                    print("sigma grad: ", y.requires_grad)
                    print("denoised_latents grad: ", z.requires_grad)
                    print("adapter_vectors grad: ", adapter_vectors[i].requires_grad)
                    y = y.view(1,1,1)
                    img_recon.append(x)
                    sigmas.append(y)
                    z = z.squeeze(0)
                    denoised_latents.append(z)

                img_recon = torch.stack(img_recon)
                sigmas = torch.stack(sigmas)
                # print("y sigma: ", sigmas.shape)
                denoised_latents = torch.stack(denoised_latents)

                # Predict the noise residual
                # model_pred = unet(
                #     inp_noisy_latents,
                #     timesteps,
                #     encoder_hidden_states=batch["prompt_ids"],
                #     added_cond_kwargs=batch["unet_added_conditions"],
                #     down_block_additional_residuals=down_block_additional_residuals,
                # ).sample

                # Denoise the latents
                # denoised_latents = model_pred * (-sigmas) + noisy_latents
                weighing = sigmas**-2.0

                # # Get the target for loss depending on the prediction type
                # if noise_scheduler.config.prediction_type == "epsilon":
                target = diffusion_engine.encode_first_stage(pixel_values.float())
                print("target grad: ", target.requires_grad)
                target = target.to(weight_dtype)
                # print("target: ", target.shape)
                # print("denoised_latents: ", denoised_latents.shape)
                # print("weighing: ", weighing.shape)
                # elif noise_scheduler.config.prediction_type == "v_prediction":
                #     target = noise_scheduler.get_velocity(latents, noise, timesteps)
                # else:
                #     raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = t2iadapter_extn.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    if len(os.listdir(args.output_dir)) > 1:
                        upload_destination = os.path.join("users/mihirneal/checkpoints/t2i_unclip_blur/", f"checkpoint-{global_step}")
                        logger.info("upload ckpt")
                        # offload_to_s3(args, source_dir=save_path, bucket="proj-fmri", s3_path=upload_destination, global_step=global_step)
                
                if args.validation_image is not None and global_step % args.validation_steps == 0:
                    t2iadapter_extn.eval()
                    image_logs = log_validation(
                        diffusion_engine,
                        t2iadapter_extn,
                        clip_img_embedder,
                        vector_suffix,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        t2iadapter_extn = accelerator.unwrap_model(t2iadapter_extn)
        torch.save(t2iadapter_extn, os.path.join(args.output_dir, "t2iextn_blur.pth"))


    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)