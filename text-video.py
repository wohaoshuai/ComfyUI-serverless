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
