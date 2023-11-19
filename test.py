from comfy_serverless import ComfyConnector

api = ComfyConnector()
running = api.is_api_running()
print('running', running)