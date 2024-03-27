from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch
import sys 

pipe = DiffusionPipeline.from_pretrained(
"playgroundai/playground-v2.5-1024px-aesthetic",
torch_dtype=torch.float16,
variant="fp16",
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]
image.save('image-playground.jpg')
