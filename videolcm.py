from diffusers.utils import load_image, export_to_video, export_to_gif

from pipeline import StableVideoDiffusionPipeline
from lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler

import torch
import os
import random
from typing import Optional
from safetensors import safe_open

def get_safetensors_files():
    models_dir = "./safetensors"
    safetensors_files = [
        f for f in os.listdir(models_dir) if f.endswith(".safetensors")
    ]
    return safetensors_files

def model_select(selected_file):
    print("load model weights", selected_file)
    pipe.unet.cpu()
    file_path = os.path.join("./safetensors", selected_file)
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    missing, unexpected = pipe.unet.load_state_dict(state_dict, strict=True)
    pipe.unet.cuda()
    del state_dict
    return



max_64_bit_int = 2**63 - 1
seed = random.randint(0, max_64_bit_int)
generator = torch.manual_seed(seed)

# Load the conditioning image
image = load_image("image.jpg")
# frames = pipe(image, decode_chunk_size=8, motion_bucket_id=127, noise_aug_strength=0.0).frames[0]
with torch.autocast("cuda"):
    frames = pipe(
        image,
        decode_chunk_size=8,
        generator=generator,
        motion_bucket_id=127,
        height=576,
        width=1024,
        num_inference_steps=4,
        min_guidance_scale=1,
        max_guidance_scale=1.2,
    ).frames[0]
export_to_gif(frames, 'generated.gif')
export_to_video(frames, "generated.mp4", fps=6)
