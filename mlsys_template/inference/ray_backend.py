import io
# import os
import warnings
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")

import ray
import requests
from fastapi import FastAPI, File, UploadFile
from ray import serve

import torch
from PIL import Image
from transformers import pipeline

from rich.console import Console

console = Console()

model_path = "/mnt/code/model_zoo/detr-resnet-50/"
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 1})
@serve.ingress(app)
class Detector:
    def __init__(self) -> None:
        self.pipe = pipeline("object-detection", model=model_path, device=0)
    
    @app.post("/predict")
    async def detect(self, file: UploadFile = File(...)) -> list:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content))
        with torch.no_grad():
            outputs = self.pipe(img)
        
        return outputs
    
    
detector_app = Detector.bind()