import json
import os
import copy
import torch
import argparse
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from xtuner.model.utils import guess_load_checkpoint
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from einops import rearrange
import numpy as np
from pathlib import Path

def _process_image(image):
    image = image.resize(size=(512, 512))
    pixel_values = torch.from_numpy(np.array(image)).float()
    pixel_values = pixel_values / 255
    pixel_values = 2 * pixel_values - 1
    pixel_values = rearrange(pixel_values, 'h w c -> c h w')
    return pixel_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('config', help='log file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default='Using the red color, draw one continuous path from the green start to the red end along walkable white cells only. Do not cross walls.')
    parser.add_argument("--src_img", type=str,default='UniREditBench/original_image/maze/1.png')
    parser.add_argument("--cfg_prompt", type=str, default="blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas.")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--output', type=str, default='output.jpg')

    args = parser.parse_args()

    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)

    config = Config.fromfile("configs/models/deepgen_scb.py")

    print(f'Device: {accelerator.device}', flush=True)
    model = BUILDER.build(config.model)
    if args.checkpoint is not None:
        if args.checkpoint.endswith('.pt'):
            state_dict = torch.load(args.checkpoint)
        else:
            state_dict = guess_load_checkpoint(args.checkpoint) 
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")
    model = model.to(device=accelerator.device)
    model = model.to(model.dtype)
    model.eval()
    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

    src_img = Image.open(args.src_img).convert('RGB')
    src_img = _process_image(src_img)

    prompts = [args.prompt.strip()] * 4

    image_pixel_srcs = [src_img[None]]* 4
    image_pixel_srcs = torch.stack(image_pixel_srcs).to(model.dtype)


    images = model.generate(prompt=prompts, cfg_prompt=[args.cfg_prompt] * len(prompts), pixel_values_src=image_pixel_srcs,
                                cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                                progress_bar=False,
                                generator=generator, height=args.height, width=args.width)

    images = rearrange(images, 'b c h w -> b h w c')

  
    images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    
    for i, image in enumerate(images):
        Image.fromarray(image).save(f"{args.output}/case_{i}.png")
