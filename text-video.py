import torch

from diffusers.utils import load_image, export_to_video
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers import AutoPipelineForText2Image
import torch
import sys 

import os
import gc

args = sys.argv
task_id = args[1]
if len(args) > 2:
    prompt_text = args[2]
if len(args) > 3:
    pipeline = args[3]

def gen_image(prompt, vertical=False, isPlayground=True):
    # w = 1344
    # h = 768
    w = 1024
    h = 576
    ow = 1024
    oh = 576
    if vertical:
        w = 768
        h = 1344
        ow = 576
        oh = 1024
        
    pipe = None
    if isPlayground:
        pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.float16,
        variant="fp16",
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3, width=w, height=h).images[0]
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        image = pipe(prompt=prompt, width=w, height=h).images[0]
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    image = image.resize((ow, oh))
    image.save("image.jpg")

if '--video-v' in pipeline or '--video-v-q' in pipeline:
    is_vertical = True
gen_image(prompt_text, is_vertical, True)

# print('prompt text', prompt_text)

# pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# ).to("cuda")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipeline_text2image(prompt=prompt_text, width=1024, height=576).images[0]
# image.save('image.jpg')
