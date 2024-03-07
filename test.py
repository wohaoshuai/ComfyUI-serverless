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

from diffusers import StableVideoDiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline, LCMScheduler
import torch
import sys 
from diffusers.utils import load_image, export_to_video, export_to_gif


pipe = StableVideoDiffusionPipeline.from_pretrained(
  "stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

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

@app.route('/gen_encoded_images', methods=['POST'])
def gen_encoded_images():
    # Parse the JSON input
    data = request.get_json()
    # input_image_base64 = data['input_image']
    prompt = data['prompt']


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

    # run_script('text-video.py', '', prompt, '')
    # run_script('video.py', '', prompt, '')
    image = gen_image(prompt) //pipeline_text2image(prompt=prompt, width=1024, height=576).images[0]
    image.save('image.jpg')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    del pipeline_text2image
    gc.collect()
    torch.cuda.empty_cache()

    image = load_image("image.jpg")
    frames = pipe(image, decode_chunk_size=8, motion_bucket_id=127, noise_aug_strength=0.0).frames[0]
    export_to_gif(frames, 'generated.gif')
    export_to_video(frames, "generated.mp4", fps=6)

    data = get_raw_data('generated.gif')



    # encoded_text = process_images()

    return jsonify({'encoded_text': data})
    # if encoded_text:
    #     return jsonify({'encoded_text': encoded_text})
    # else:
    #     return jsonify({'error': 'No images generated'})

def gen_image(prompt):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to("cuda") 
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl") #yes, it's a normal LoRA

    results = pipe(
        prompt="The spirit of a tamagotchi wandering in the city of Vienna",
        num_inference_steps=4,
        guidance_scale=0.0,
        width=1024,
        height=576
    )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return results.images[0]

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)