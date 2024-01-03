# text-to-video function
# import cv2
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video


def predict(text_prompt: str, accelerate=False) -> str:
    """text-to-video with pytorch engine

    Args:
        text_prompt (str): a text prompt
        model_id (str): model name in huggingface model hub
        accelerate (bool): whether to use accelerating
    """
    model_id = "damo-vilab/text-to-video-ms-1.7b"
    if accelerate: #TBD
        pass
    else:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        pipe = pipe.to("cuda")
        video_frames = pipe(text_prompt).frames
        video_path = export_to_video(video_frames)

    # free_gpu_memory
    del pipe
    torch.cuda.empty_cache()

    return video_path