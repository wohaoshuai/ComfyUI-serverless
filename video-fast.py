from diffusers import StableVideoDiffusionPipeline
import torch
import time
from PIL import Image
from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig
from diffusers.utils import load_image, export_to_video, export_to_gif

def compile_model(model):
    config = CompilationConfig.Default()

    # xformers and Triton are suggested for achieving best performance.
    # It might be slow for Triton to generate, compile and fine-tune kernels.
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")
    # NOTE:
    # When GPU VRAM is insufficient or the architecture is too old, Triton might be slow.
    # Disable Triton if you encounter this problem.
    try:
        import triton

        config.enable_triton = True
    except ImportError:
        print("Triton not installed, skip")

    model = compile(model, config)
    return model


repo_id = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
cache_dir = "./cache"
pipeline = StableVideoDiffusionPipeline.from_pretrained(
    repo_id, cache_dir=cache_dir, variant="fp16", torch_dtype=torch.float16
)
pipeline.to("cuda")

# Comment the following the line to disable stable-fast
pipe = compile_model(pipeline)


# generator = torch.manual_seed(42)

# Warm up call for stable-fast compilation with batch size = 1
# frames = pipeline(
#     [Image.open(i).convert("RGB") for i in image],
#     decode_chunk_size=8,
# ).frames

# batch size = 4 for the actual call
# image *= 4

begin = time.time()
# frames = pipeline(
#     [Image.open(i).convert("RGB") for i in image],
#     decode_chunk_size=8,
# ).frames
image = load_image("image.jpg")
frames = pipe(image, decode_chunk_size=8, motion_bucket_id=127, noise_aug_strength=0.0).frames[0]
end = time.time()
run_time = end - begin
print(f"run time: {run_time:.3f}s")

export_to_gif(frames, 'generated.gif')

peak_mem_allocated = torch.cuda.max_memory_allocated()
peak_mem_reserved = torch.cuda.max_memory_reserved()
print(f"peak GPU memory allocated: {peak_mem_allocated / 1024**3:.3f}GiB")
print(f"peak GPU memory reserved: {peak_mem_reserved / 1024**3:.3f}GiB")