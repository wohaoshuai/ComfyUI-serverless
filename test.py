from comfy_serverless import ComfyConnector
import json

api = ComfyConnector()
# running = api.is_api_running()
# print('running', running)
# prompt = json.load(open('clean-small.json'))
prompt = json.load(open('workflow_api.json'))
print(prompt)
images = api.generate_images(prompt) 
print('images', images)
if images and len(images) > 0:
    for i, image in enumerate(images):
        image.save(f"~/outputs/test{i}.png")