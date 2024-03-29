from comfy_serverless import ComfyConnector
import json
from compressor import *
from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import os
import torch
import time
import gc

app = Flask(__name__)
CORS(app)

# from diffusers import StableVideoDiffusionPipeline
import oneflow as flow

from onediffx import compile_pipe, compiler_config
from onediffx.deep_cache import StableVideoDiffusionPipeline

# from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import sys 
from diffusers.utils import load_image, export_to_video, export_to_gif

from lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler
from typing import Optional
from safetensors import safe_open

islcm = False

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

if islcm:
    noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
        num_train_timesteps=40,
        sigma_min=0.002,
        sigma_max=700.0,
        sigma_data=1.0,
        s_noise=1.0,
        rho=7,
        clip_denoised=False,
    )
    pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1", scheduler=noise_scheduler, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    model_select("AnimateLCM-SVD-xt-1.1.safetensors")
    pipe = compile_pipe(pipe,)
else:
    pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe = compile_pipe(pipe,)

def get_raw_data(filename):
    # url_values = urllib.parse.urlencode(data)
    # file_path = f"{filename}"
    file_path = f"{filename}"

    # Open the WEBP image file, read it into a binary format
    with open(file_path, "rb") as image_file:
        # Read the image data
        image_data = image_file.read()

    # Convert the binary data to a base64-encoded string
    encoded_string = base64.b64encode(image_data).decode("utf-8")
    return encoded_string

def process_images(w=768, h=768):
    api = ComfyConnector()

    prompt = json.load(open('workflow_yumo_api.json'))
    # prompt["23"]["inputs"]["image"] = 'input_image.png'
    # if text:
    #     prompt["187"]["inputs"]["text"] = text
    # else:
    #     prompt["187"]["inputs"]["text"] = 'product shot with a creative background, 4k, leica, commercial photography'
    print('process_image')

    images = api.generate_webp(prompt)
    
    print("image length", len(images))
    if images and len(images) > 0:
        # Encode the images into a single file
        # encoded_images = encode_webps(images)
        # Save the encoded images to a string
        # encoded_text = save_encoded_images_to_string(encoded_images)

        return images
    else:
        return None

import cv2

import cv2

def resize_image(image_path, max_width=1024, max_height=576):
    """
    Resizes an image while maintaining its aspect ratio and ensuring that both the
    maximum width and maximum height are not exceeded.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        max_width (int, optional): The maximum width of the resized image. Default is 1024.
        max_height (int, optional): The maximum height of the resized image. Default is 576.

    Returns:
        numpy.ndarray: The resized image as a NumPy array.
    """
    
    image = cv2.imread(image_path)

    # Get the current dimensions
    height, width, _ = image.shape
    print('Original Image Size:', height, width)

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Calculate the new dimensions while maintaining the aspect ratio
    if aspect_ratio > max_width / max_height:  # Landscape
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    else:  # Portrait or square
        new_height = max_height
        new_width = int(max_height * aspect_ratio)

    # Ensure the new dimensions do not exceed the maximum values
    new_width = min(new_width, max_width)
    new_height = min(new_height, max_height)

    if height == 1365 and width == 1024:
        new_height = 768
        new_width = 576
    if height == 1820 and width == 1024:
        new_height = 1024
        new_width = 576
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

@app.route('/gen_encoded_images', methods=['POST'])
def gen_encoded_images():
    # Parse the JSON input
    data = request.get_json()
    # input_image_base64 = data['input_image']
    prompt = data['prompt']
    pipeline = data['pipeline']

    if '--img2video' in pipeline:
        print('--img2video')
    else:
        is_vertical = False
        if '--video-v' in pipeline or '--video-v-q' in pipeline:
            is_vertical = True
        # image = gen_image(prompt, is_vertical, True)
        # image.save('image.jpg')
        run_script('text-video.py', '', prompt, pipeline)

    # shuffle_image_base64 = data['shuffle_image']
    # text = data['prompt']
    # batch_size = data['batch_size']
    # print('input', input_image_base64)
    # print('text', text)

    # Decode base64 strings to images
    # shuffle_image = base64_to_image(shuffle_image_base64)
    # Save the images as local files
    # save_image(shuffle_image, 'shuffle_image.png')

    # input_image = base64_to_image(input_image_base64)
    # print('image', input_image)
    # save_image(input_image, 'input_image.png')

    # run_script('video.py', '', prompt, '')


    if not '--video-v' in pipeline:
        resized_image = resize_image("image.jpg")
        cv2.imwrite("image.jpg", resized_image)
        height, width, _ = resized_image.shape
        print(f"Resized image size: {width} x {height}")

    image = load_image("image.jpg")

    if '--video-v' in pipeline:
        height = 1024
        width = 576

    if islcm:
        frames = pipe(
            image,
            decode_chunk_size=8,
            motion_bucket_id=127,
            height=height,
            width=width,
            num_inference_steps=4,
            min_guidance_scale=1,
            max_guidance_scale=1.2,
        ).frames[0]
    else:
        frames = pipe(image, decode_chunk_size=4, motion_bucket_id=127, noise_aug_strength=0.0, height=height, width=width).frames[0]
    export_to_gif(frames, 'generated.gif')
    export_to_video(frames, "generated.mp4", fps=6)

    data = get_raw_data('generated.gif')



    # encoded_text = process_images()

    return jsonify({'encoded_text': data})
    # if encoded_text:
    #     return jsonify({'encoded_text': encoded_text})
    # else:
    #     return jsonify({'error': 'No images generated'})

def gen_image(prompt, vertical=False, isPlayground=True):
    w = 1344
    h = 768
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
        image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3, width=w, height=h).images[0]
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        image = pipe(prompt=prompt, width=w, height=h).images[0]
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return image.resize((ow, oh))


import subprocess
def run_script(script_name, output_name, prompt, filename):
    print('run_script', ['python', script_name, output_name, prompt, filename])
    result = subprocess.run(['python', script_name, output_name, prompt, filename], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    print("Stdout:")
    print(result.stdout)
    print("Stderr:")
    print(result.stderr)
    if result.returncode == 0:
        output_lines = result.stdout.strip().split('\n')
        last_line = output_lines[-1]
        return last_line
    else:
        return False

def base64_to_image(base64_string):
    # Decode the base64 string to an image
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


def save_image(image, filename):
    # Remove existing files with the same names if they exist
    input_image_path = os.path.join(os.path.expanduser('~/ComfyUI/input/'), 'input_image.png')

    if os.path.exists(input_image_path):
        os.remove(input_image_path)
    # if os.path.exists(shuffle_image_path):
    #     os.remove(shuffle_image_path)

    # Save the image as a local file
    save_path = os.path.join(os.path.expanduser('~/ComfyUI/input/'), filename)
    image.save(save_path)

@app.route('/get_encoded_images', methods=['GET'])
def get_encoded_images():
    encoded_text = process_images()
    if encoded_text:
        return jsonify({'encoded_text': encoded_text})
    else:
        return jsonify({'error': 'No images generated'})

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'status': 200})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)