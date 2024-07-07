import gradio as gr
from transformers import AutoImageProcessor, DetrForObjectDetection
import torch
# import cv2
import numpy as np
from PIL import Image, ImageDraw

model_path = "/mnt/code/model_zoo/detr-resnet-50"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load model and image processor
image_processor = AutoImageProcessor.from_pretrained(model_path)
model = DetrForObjectDetection.from_pretrained(model_path).to(device)

def inference(image): # connect to the backend
    # inference logic
    # image_cv = cv2.imread(image)
    image = Image.open(image)
    draw = ImageDraw.Draw(image)
    # image_cv = np.array(image.convert('RGB'))[:, :, ::-1].copy()
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x1, y1, x2, y2 = box
        x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
        # print(x1, x2, y1, y2)
        draw.rectangle((x, y, x + w, y + h), outline="red", width=3)
        draw.text((x, y), model.config.id2label[label.item()], fill="black")
        # demo_image = cv2.rectangle(image_cv, (x, y), (x + w, y + h), (36,255,12), 1)
        # cv2.putText(demo_image, f"{model.config.id2label[label.item()]}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
    return image

demo = gr.Interface(
    fn = inference,
    inputs = gr.Image(height=300, type="filepath"),
    outputs = gr.Image(),
    title = "Object Detection Demo",
    description = "This is a simple web demo for object detection."
)

demo.launch(server_name="0.0.0.0", server_port=8805)