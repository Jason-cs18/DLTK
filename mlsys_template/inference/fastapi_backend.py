import io
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile
# from transformers import AutoImageProcessor, DetrForObjectDetection

from rich.console import Console

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model_path = "/mnt/code/model_zoo/detr-resnet-50/"

console = Console()
app = FastAPI()

# class Detector:
#     def __init__(self) -> None:
#         pass

pipe = pipeline("object-detection", model=model_path, device=0)

@app.post("/")
async def detect(file: UploadFile = File(...)) -> list:
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    with torch.no_grad():
        outputs = pipe(img)
    
    console.log("inference done")
    return outputs