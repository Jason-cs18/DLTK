# text-to-image function
import sys
import yaml
from typing import Dict

# from numba import cuda
from PIL import Image, ImageDraw


def predict(text_prompt: str, model_id: str, accelerate=False) -> None:
    """text-to-image with pytorch engine

    Args:
        text_prompt (str): a text prompt
        model_id (str): model name in huggingface model hub
        accelerate (bool): whether to use accelerating
    """
    from diffusers import AutoPipelineForText2Image
    import torch
 
    if accelerate: # all optimizations are not supporting old GPUs
        # using torch.compile 
        # torch._inductor.config.conv_1x1_as_mm = True
        # torch._inductor.config.coordinate_descent_tuning = True
        # torch._inductor.config.epilogue_fusion = False
        # torch._inductor.config.coordinate_descent_check_all_directions = True
        pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipeline.fuse_qkv_projections()
        # pipeline.unet.to(memory_format=torch.channels_last)
        # pipeline.vae.to(memory_format=torch.channels_last)
        # Compile the UNet and VAE.
        # pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=True)
        # pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipeline.unet.set_default_attn_processor()    
        pipeline.vae.set_default_attn_processor()
    
    image = pipeline(text_prompt).images[0]
    
    # free_gpu_memory
    del pipeline
    torch.cuda.empty_cache()

    return image