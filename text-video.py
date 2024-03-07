import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

from diffusers import AutoPipelineForText2Image
import torch
import sys 

args = sys.argv
task_id = args[1]
if len(args) > 2:
    prompt_text = args[2]
if len(args) > 3:
    filename = args[3]

print('prompt text', prompt_text)

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt_text, width=1024, height=576).images[0]
image.save('image.jpg')
