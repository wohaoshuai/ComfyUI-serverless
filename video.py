import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif

pipe = StableVideoDiffusionPipeline.from_pretrained(
  "stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("image.jpg")

frames = pipe(image, decode_chunk_size=8, motion_bucket_id=127, noise_aug_strength=0.0).frames[0]
export_to_gif(frames, 'generated.gif')
export_to_video(frames, "generated.mp4", fps=6)
