# text-to-image with runwayml/stable-diffusion-v1-5
# some advanced acceleration techniques are not applied 
# (progressive timestep distillation, model compression, reusing adjacent features)
# docs: https://huggingface.co/docs/diffusers/tutorials/fast_diffusion

import sys
import yaml
from typing import Dict

from PIL import Image, ImageDraw


def read_args_from_yaml(config_path: str) -> Dict[str, str]:
    """read arguments from config.yaml

    Args:
        config_path (str): path to configuration

    Returns:
        Dict[str, str]: configuration of inference
    """
    with open(config_path, 'r') as file:
        args_dict = yaml.safe_load(file)
    return args_dict


def infer_pytorch(text_prompt: str, model_id: str, accelerate=False) -> None:
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
    
    print("*"*20)
    print(f"NVIDIA GPU: {torch.cuda.get_device_name()} with {torch.cuda.get_device_properties(0).total_memory / 10**9:02f} G")
    print(f"Model: {model_id}")
    print(f"Accelerate: {accelerate}")
    print(f"Text Prompt: {text_prompt}")
    print(f"Torch Version: {torch.__version__}")
    # print(f"#Params: {pipeline.model.num_parameters() / 10**9}G")
    print("*"*20)
    image = pipeline(text_prompt).images[0]
    image.show()
    print(type(image))
    print(f"inference completed!")


def infer_onnx(text_prompt: str, model_id: str) -> None:
    """text-to-image with onnx engine

    Args:
        text_prompt (str): a text prompt
        model_id (str): model name in huggingface model hub
    """
    from optimum.onnxruntime import ORTStableDiffusionPipeline
    pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, revision="onnx")
    image = pipeline(text_prompt).images[0]
    image.show()
    print(f"inference completed!")


def main():
    engine = sys.argv[1]
    if engine == 'pytorch':
        default_config_path = 'conf/infer_config_pytorch.yaml'
    else:
        default_config_path = 'conf/infer_config_onnx.yaml'
    configs = read_args_from_yaml(default_config_path)
    text_prompt = configs['text_prompt']
    model_id = configs['model_id']
    inference_engine = configs['inference_engine']
    if inference_engine == 'pytorch':
        infer_pytorch(text_prompt, model_id)
    elif inference_engine == 'onnx':
        infer_onnx(text_prompt, model_id)
    else:
        print("unsupported inference engine (onnx and pytorch are supported)!")


if __name__ == '__main__':
    main()
