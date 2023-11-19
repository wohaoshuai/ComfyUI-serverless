from comfy_serverless import ComfyConnector
import json
from compressor import save_encoded_images_to_file, encode_images, decode_images, read_encoded_images_from_file

api = ComfyConnector()
# running = api.is_api_running()
# print('running', running)
# prompt = json.load(open('clean-small.json'))
prompt = json.load(open('workflow_api.json'))
print(prompt)
images = api.generate_images(prompt) 
print('images', images)
# if images and len(images) > 0:
#     for i, image in enumerate(images):
#         image.save(f"outputs/test{i}.png")

# Assuming you have a list of images called "images"
if images and len(images) > 0:
    # Encode the images into a single file
    encoded_images = encode_images(images)
    
    # Save the encoded images to a file
    save_encoded_images_to_file(encoded_images, "encoded_images.txt")
    
    # Read the encoded images from the file
    # encoded_images = read_encoded_images_from_file("encoded_images.txt")
    
    # Decode the images back into a list of PIL Images
    # decoded_images = decode_images(encoded_images)
    
    # Save the decoded images to individual files
    # for i, image in enumerate(decoded_images):
    #     image.save(f"outputs/test{i}.png")
