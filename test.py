from comfy_serverless import ComfyConnector
import json
from compressor import *
from flask import Flask, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

def process_images():
    api = ComfyConnector()
    prompt = json.load(open('workflow_api.json'))
    prompt["162"]["inputs"]["seed"] = random.randint(1,4294967294)
    api.replace_key_value(prompt, )
    
    images = api.generate_images(prompt)
    
    if images and len(images) > 0:
        # Encode the images into a single file
        encoded_images = encode_images(images)
        # Save the encoded images to a string
        encoded_text = save_encoded_images_to_string(encoded_images)
        return encoded_text
    else:
        return None

@app.route('/get_encoded_images', methods=['GET'])
def get_encoded_images():
    encoded_text = process_images()
    if encoded_text:
        return jsonify({'encoded_text': encoded_text})
    else:
        return jsonify({'error': 'No images generated'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)