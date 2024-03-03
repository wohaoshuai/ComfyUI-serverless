#!/bin/bash
nohup python /home/contact/ComfyUI/main.py --listen > output_comfy.log 2>&1 &
nohup python /home/contact/ComfyUI-serverless/test.py --listen > output_serverless.log 2>&1 &