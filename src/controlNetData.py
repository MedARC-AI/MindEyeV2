import pandas as pd
import datasets
import os
import pickle as pkl
import torch

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "target": datasets.Image(),
        "source": datasets.Image(),
    },
)

METADATA_DIR = "/weka/proj-fmri/shared/controlNetData/data3.pkl"
SOURCE_DIR = "/weka/proj-fmri/shared/controlNetData/source"
TARGET_DIR = "/weka/proj-fmri/shared/controlNetData/target"

# METADATA_URL = hf_hub_url(
#     "fusing/fill50k",
#     filename="train.jsonl",
#     repo_type="dataset",
# )

# IMAGES_URL = hf_hub_url(
#     "fusing/fill50k",
#     filename="images.zip",
#     repo_type="dataset",
# )

# CONDITIONING_IMAGES_URL = hf_hub_url(
#     "fusing/fill50k",
#     filename="conditioning_images.zip",
#     repo_type="dataset",
# )

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class CocoTest(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = METADATA_DIR
        target_dir = TARGET_DIR
        source_dir = SOURCE_DIR


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                   #  "target_dir": TARGET_DIR,
                   # "source_dir": SOURCE_DIR,
                    # "num_examples": 118287, 
                    "num_examples": 5000, 
                },
            ),
        ]

    def _generate_examples(self, metadata_path, num_examples):
        data = []
        with open(metadata_path, 'rb') as f:
            loaded_data = pkl.load(f)
            for line in loaded_data[:num_examples]:
                data.append(line)

        last_processed_filename = None
        for _, item in enumerate(data):
            source_filename = item['source']
            if source_filename.startswith('/fsx/'):
                source_filename = '/weka/' + source_filename[len('/fsx/'):]
            target_filename = item['target']
            if target_filename.startswith('/fsx/'):
                target_filename = '/weka/' + target_filename[len('/fsx/'):]
            if target_filename == last_processed_filename:
                continue
            else:
                last_processed_filename = target_filename
            # eva_filename = item['eva']
            # clip_filename = item['clip']
            # dv2_filename = item['dv2']
            # prompt = item['prompt']
            
            

            tgt_img = open(target_filename, "rb").read()
            src_img = open(source_filename, "rb").read()
          #  h_img = open(heatmap_filename, "rb").read()
            yield item["target"], {
                # "prompt": prompt,
                "target": {
                    "path": target_filename,
                    "bytes": tgt_img,
                },

                "source": {
                    "path": source_filename,
                    "bytes": src_img,
                },
            }
