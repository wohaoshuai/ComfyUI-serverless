from comfy_serverless import ComfyConnector
import json

api = ComfyConnector()
# running = api.is_api_running()
# print('running', running)
prompt = json.load(open('test.json'))
images = api.generate_images(prompt) 
for i, image in enumerate(images):
    image.save(f"test{i}.png")