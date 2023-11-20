nohup python ~/ComfyUI/main.py --listen > output_comfy.log 2>&1 &
cd ~/ComfyUI-serverless
nohup python test.py --listen > output_serverless.log 2>&1 &