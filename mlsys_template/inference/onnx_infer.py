from time import time

import torch
import onnxruntime
from PIL import Image
from transformers import AutoImageProcessor, DetrForObjectDetection
from rich.console import Console

console = Console()

img_path = "../test_data/000000039769.jpg"
model_path = "/mnt/code/model_zoo/detr-resnet-50-onnx"
model_onnx_path = "/mnt/code/model_zoo/detr-resnet-50-onnx/model.onnx"
device = "cuda" if torch.cuda.is_available() else "cpu"

image = Image.open(img_path)

console.log("Load pre-trained DETR (ONNX)")
image_processor = AutoImageProcessor.from_pretrained(model_path)

session_options = onnxruntime.SessionOptions()
providers = ["CPUExecutionProvider"]
if device == 'cuda':
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
session = onnxruntime.InferenceSession(model_onnx_path, sess_options=session_options, providers=providers)	

console.log("preprocessing")
inputs = image_processor(images=image, return_tensors="pt")
# print(inputs.data)

console.log("inference")
warmup_times = 2
for i in range(warmup_times):
    pred = session.run(None, {'pixel_values': inputs.data['pixel_values'].numpy()})

measure_times = 5
start = time()
for i in range(measure_times):
    pred = session.run(None, {'pixel_values': inputs.data['pixel_values'].numpy()})
    
end = time()
console.log(f"Latency: {round((end - start) / measure_times * 1000, 2)} ms")