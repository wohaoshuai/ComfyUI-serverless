from comfy_serverless import ComfyConnector
import json

api = ComfyConnector()
# running = api.is_api_running()
# print('running', running)
prompt = json.load(open('clean-small.json'))
print(prompt)
images = api.generate_images(prompt) 
print('images')
# for i, image in enumerate(images):
#     image.save(f"test{i}.png")