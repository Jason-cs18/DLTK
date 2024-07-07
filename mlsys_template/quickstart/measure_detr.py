# measure detr cost (latency and memory)
from time import time
from transformers import AutoImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from rich.console import Console

console = Console()

img_path = "../test_data/000000039769.jpg"
model_path = "/mnt/code/model_zoo/detr-resnet-50"
device = "cuda" if torch.cuda.is_available() else "cpu"

image = Image.open(img_path)

console.log("Load pre-trained DETR")
image_processor = AutoImageProcessor.from_pretrained(model_path)
model = DetrForObjectDetection.from_pretrained(model_path).to(device)

console.log("preprocessing")
inputs = image_processor(images=image, return_tensors="pt").to(device)

# console.log(f"GPU used {round(torch.cuda.memory_allocated(device)/1024/1024, 2)} MB memory")

console.log("Inference")
warmup_times = 2
for i in range(warmup_times):
    with torch.no_grad():
        outputs = model(**inputs)

measure_times = 5
start = time()
for i in range(measure_times):
    with torch.no_grad():
        outputs = model(**inputs)
    
end = time()
console.log(f"model size: {model.num_parameters()}")
console.log(f"Latency: {round((end - start) / measure_times * 1000, 2)} ms")
console.log(f"GPU used {round(torch.cuda.memory_allocated(device)/1024/1024, 2)} MB memory")

target_sizes = torch.tensor([image.size[::-1]])
console.log("postprocessing")
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#         f"Detected {model.config.id2label[label.item()]} with confidence "
#         f"{round(score.item(), 3)} at location {box}"
#     )