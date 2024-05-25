from comfy_serverless import ComfyConnector
import json
from compressor import *
from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import os

app = Flask(__name__)
CORS(app)

def process_images(text, batch_size = 16, w=768, h=768):
    api = ComfyConnector()

    prompt = json.load(open('ic_workflow.json'))

    # input image

    # input prompt

    # w / h

    # reference image

    prompt["16"]["inputs"]["seed"] = random.randint(1,4294967294)

    # prompt["14"]["inputs"]["width"] = w
    # prompt["14"]["inputs"]["height"] = h

    prompt["11"]["inputs"]["image"] = 'image.jpg'
    # prompt["198"]["inputs"]["image"] = 'shuffle.png'
    if text:
        prompt["6"]["inputs"]["text"] = text
    else:
        prompt["6"]["inputs"]["text"] = 'product shot with a creative background, 4k, leica, commercial photography'
    print('process_image')

    images = api.generate_images(prompt)
    print('outputresults', images)
    
    if images and len(images) > 0:
        # Encode the images into a single file
        encoded_images = encode_images(images)
        # Save the encoded images to a string
        encoded_text = save_encoded_images_to_string(encoded_images)
        return encoded_text
    else:
        return None

@app.route('/gen_encoded_images', methods=['POST'])
def gen_encoded_images():
    # Parse the JSON input
    data = request.get_json()
    input_image_base64 = data['input_image']
    # shuffle_image_base64 = data['shuffle_image']
    text = data['prompt']
    batch_size = data['batch_size']
    # print('input', input_image_base64)
    print('text', text)

    # Decode base64 strings to images
    input_image = base64_to_image(input_image_base64)
    # shuffle_image = base64_to_image(shuffle_image_base64)
    print('image', input_image)
    # Save the images as local files
    save_image(input_image, 'input_image.png')
    # save_image(shuffle_image, 'shuffle_image.png')

    encoded_text = process_images(text, batch_size=batch_size)

    if encoded_text:
        return jsonify({'encoded_text': encoded_text})
    else:
        return jsonify({'error': 'No images generated'})

def base64_to_image(base64_string):
    # Decode the base64 string to an image
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


def save_image(image, filename):
    # Remove existing files with the same names if they exist
    input_image_path = os.path.join(os.path.expanduser('~/ComfyUI/input/'), 'input_image.png')
    shuffle_image_path = os.path.join(os.path.expanduser('~/ComfyUI/input/'), 'shuffle_image.png')

    if os.path.exists(input_image_path):
        os.remove(input_image_path)
    if os.path.exists(shuffle_image_path):
        os.remove(shuffle_image_path)

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
