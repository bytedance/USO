# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import dataclasses
from typing import Literal

from dotenv import load_dotenv
load_dotenv() 


from accelerate import Accelerator
from transformers import HfArgumentParser
from PIL import Image
import json
import itertools
import torch

from uso.flux.pipeline import USOPipeline, preprocess_ref
from transformers import SiglipVisionModel, SiglipImageProcessor
from tqdm import tqdm


def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im


@dataclasses.dataclass
class InferenceArgs:
    prompt: str | None = None
    image_paths: list[str] | None = None
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 1024
    height: int = 1024
    num_steps: int = 25
    guidance: float = 4
    seed: int = 3407
    save_path: str = "output/inference"
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 128
    pe: Literal["d", "h", "w", "o"] = "d"
    content_ref: int = 512
    ckpt_path: str | None = None
    use_siglip: bool = True
    instruct_edit: bool = False
    hf_download: bool = False  # set to false, we must not auto download the weights (゜-゜) 


def main(args: InferenceArgs):
    accelerator = Accelerator()

    # init SigLIP model
    siglip_processor = None
    siglip_model = None
    if args.use_siglip:

        # ⚠️ Weights now load from local paths via .env instead of downloading
        siglip_path = os.getenv("SIGLIP_PATH", "google/siglip-so400m-patch14-384")
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
        siglip_model = SiglipVisionModel.from_pretrained(siglip_path)

        siglip_model.eval()
        siglip_model.to(accelerator.device)
        print("SigLIP model loaded successfully")

    pipeline = USOPipeline(
        args.model_type,
        accelerator.device,
        args.offload,
        only_lora=args.only_lora,
        lora_rank=args.lora_rank,
        hf_download=args.hf_download,
    )
    if args.use_siglip and siglip_model is not None:
        pipeline.model.vision_encoder = siglip_model

    assert (
        args.prompt is not None or args.eval_json_path is not None
    ), "Please provide either prompt or eval_json_path"

    if args.eval_json_path is not None:
        with open(args.eval_json_path, "rt") as f:
            data_dicts = json.load(f)
        data_root = os.path.dirname(args.eval_json_path)
    else:
        data_root = ""
        data_dicts = [{"prompt": args.prompt, "image_paths": args.image_paths}]

    print(
        f"process: {accelerator.num_processes}/{accelerator.process_index}, \
    process images: {len(data_dicts)}/{len(data_dicts[accelerator.process_index::accelerator.num_processes])}"
    )

    data_dicts = data_dicts[accelerator.process_index :: accelerator.num_processes]

    accelerator.wait_for_everyone()
    local_task_count = len(data_dicts) * args.num_images_per_prompt
    if accelerator.is_main_process:
        progress_bar = tqdm(total=local_task_count, desc="Generating Images")

    for (i, data_dict), j in itertools.product(
        enumerate(data_dicts), range(args.num_images_per_prompt)
    ):
        ref_imgs = []
        for _, img_path in enumerate(data_dict["image_paths"]):
            if img_path != "":
                img = Image.open(os.path.join(data_root, img_path)).convert("RGB")
                ref_imgs.append(img)
            else:
                ref_imgs.append(None)
        siglip_inputs = None
        if args.use_siglip and siglip_processor is not None:
            with torch.no_grad():
                siglip_inputs = [
                    siglip_processor(img, return_tensors="pt").to(pipeline.device) 
                    for img in ref_imgs[1:] if isinstance(img, Image.Image)
                    ]

        ref_imgs_pil = [
            preprocess_ref(img, args.content_ref) for img in ref_imgs[:1] if isinstance(img, Image.Image)
        ]

        if args.instruct_edit:
            args.width, args.height = ref_imgs_pil[0].size
            args.width, args.height = args.width * (1024 / args.content_ref), args.height * (1024 / args.content_ref)
        image_gen = pipeline(
            prompt=data_dict["prompt"],
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed + j,
            ref_imgs=ref_imgs_pil,
            pe=args.pe,
            siglip_inputs=siglip_inputs,
        )
        if args.concat_refs:
            image_gen = horizontal_concat([image_gen, *ref_imgs])

        if "save_dir" in data_dict:
            config_save_path = os.path.join(args.save_path, data_dict["save_dir"] + f"_{j}.json")
            image_save_path = os.path.join(args.save_path, data_dict["save_dir"] + f"_{j}.png")
        else:
            os.makedirs(args.save_path, exist_ok=True)
            config_save_path = os.path.join(args.save_path, f"{i}_{j}.json")
            image_save_path = os.path.join(args.save_path, f"{i}_{j}.png")

        # save config and image
        os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
        image_gen.save(image_save_path)
        # ensure the prompt and image_paths are saved in the config file
        args.prompt = data_dict["prompt"]
        args.image_paths = data_dict["image_paths"]
        args_dict = vars(args)
        with open(config_save_path, "w") as f:
            json.dump(args_dict, f, indent=4)

        if accelerator.is_main_process:
            progress_bar.update(1)
    if accelerator.is_main_process:
        progress_bar.close()


if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
