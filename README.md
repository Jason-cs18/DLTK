# DLTK
In this comprehensive guide, we provide an extensive overview of deep learning toolkits, covering model design and deployment, with practical examples of image classification, object detection, semantic segmentation, and text-to-image synthesis. This guide is perfect for beginners and experts alike looking to gain a deeper understanding of the various applications of deep learning in machine learning.

<p align="center">
  <img src="https://github.com/Jason-cs18/DLTK/blob/main/imgs/dlsys_outline.png" />
</p>

## Table of Contents
- [DLTK](#dltk)
  - [Table of Contents](#table-of-contents)
  - [Vision Transformer](#vision-transformer)
  - [Model Training via Pytorch](#model-training-via-pytorch)
    - [Fine-tune Vision Transformer on Custom Dataset](#fine-tune-vision-transformer-on-custom-dataset)
    - [Experiment Tracking with W\&B](#experiment-tracking-with-wb)
    - [Evaluate the Pre-Trained Vision Transformer on New Images](#evaluate-the-pre-trained-vision-transformer-on-new-images)
    - [Accelerate the Inference of Vision Transformer](#accelerate-the-inference-of-vision-transformer)
  - [Model Training via Pytorch-Lightning](#model-training-via-pytorch-lightning)
    - [Fine-tune Vision Transformer on Custom Dataset](#fine-tune-vision-transformer-on-custom-dataset-1)
    - [Experiment Tracking with W\&B](#experiment-tracking-with-wb-1)
    - [Evaluate the Pre-Trained Vision Transformer on New Images](#evaluate-the-pre-trained-vision-transformer-on-new-images-1)
    - [Accelerate the Inference of Vision Transformer](#accelerate-the-inference-of-vision-transformer-1)
  - [Model Inference via HuggingFace](#model-inference-via-huggingface)
    - [Image Classification with Vision Transformer](#image-classification-with-vision-transformer)
    - [Object Detection with DETR](#object-detection-with-detr)
    - [Image Segmentation with SegFormer](#image-segmentation-with-segformer)
    - [Story Writing with MPT-7B](#story-writing-with-mpt-7b)
    - [Text-to-Image Generation with Stable Diffusion](#text-to-image-generation-with-stable-diffusion)
  - [Web Demo via Gradio](#web-demo-via-gradio)
    - [Image Classification](#image-classification)
    - [Object Detection](#object-detection)
    - [Image Segmentation](#image-segmentation)
    - [Text-to-Image Generation](#text-to-image-generation)
    - [Story Writer](#story-writer)
  - [Model Deployment with NVIDIA Triton](#model-deployment-with-nvidia-triton)
    - [Model Conversion](#model-conversion)
    - [Model Optimization](#model-optimization)
    - [Performance Monitoring](#performance-monitoring)
  - [Model Deployment with Anyscale Ray](#model-deployment-with-anyscale-ray)
    - [Model Conversion](#model-conversion-1)
    - [Model Optimization](#model-optimization-1)
    - [Performance Monitoring](#performance-monitoring-1)
  - [Model Deployment with Alibaba MNN](#model-deployment-with-alibaba-mnn)
    - [Model Conversion](#model-conversion-2)
    - [Model Optimization](#model-optimization-2)
    - [Performance Monitoring](#performance-monitoring-2)
  - [References](#references)

## Vision Transformer
 Inspired by the success of Transformers [[1](#references)] in natural language processing (NLP), Vision Transformer [[2](#references)] was proposed by Google in 2020 to revolutionize computer vision tasks such as image classification, object detection, and semantic segmentation. Similar to text tokens, Vision Transformer tokenizes images through patch encoding and performs multi-head self-attention on the image patches. Unlike convolution neural networks (CNNs), Vision Transformers learn global representations of the image patches through self-attention and achieve a large receptive field, making them highly effective in computer vision tasks.

 In experiments, we start with an pre-trained Vision Transformer and fine-tune it on the [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset).

|Transformer|Vision Transformer|
|:---:|:---:|
|![](https://camo.githubusercontent.com/022f6ad1b0745d754a8c6cb474a8bd458b0de4d028558607456387a347b78d80/68747470733a2f2f64326c2e61692f5f696d616765732f7472616e73666f726d65722e737667)|![](https://camo.githubusercontent.com/5c9e02651b64a9113981be3d72942564778bee4b86a5211ad59d452da8f30a1f/68747470733a2f2f64326c2e61692f5f696d616765732f7669742e737667)|

## Model Training via Pytorch

### Fine-tune Vision Transformer on Custom Dataset

### Experiment Tracking with W&B

### Evaluate the Pre-Trained Vision Transformer on New Images

### Accelerate the Inference of Vision Transformer

## Model Training via Pytorch-Lightning

### Fine-tune Vision Transformer on Custom Dataset

### Experiment Tracking with W&B

### Evaluate the Pre-Trained Vision Transformer on New Images

### Accelerate the Inference of Vision Transformer

## Model Inference via HuggingFace

### Image Classification with Vision Transformer

### Object Detection with DETR

### Image Segmentation with SegFormer

### Story Writing with MPT-7B

### Text-to-Image Generation with Stable Diffusion

## Web Demo via Gradio

### Image Classification

### Object Detection

### Image Segmentation

### Text-to-Image Generation

### Story Writer

## Model Deployment with NVIDIA Triton

### Model Conversion

### Model Optimization

### Performance Monitoring

## Model Deployment with Anyscale Ray
Compared with [NVIDIA Triton](#references), [Ray](#references) supports more useful featues like distributed data processing, distributed training, scalable reinforcement learning.

### Model Conversion

### Model Optimization

### Performance Monitoring

## Model Deployment with Alibaba MNN

### Model Conversion

### Model Optimization

### Performance Monitoring

## References
1. [NeurIPS'17] [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) | Google Brain, Google Research, University of Toronto.
2. [ICLR'21] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy) | Google Research, Google Brain.
3. deep-learning-project-template https://github.com/Lightning-AI/deep-learning-project-template
4. Anyscale Ray https://github.com/ray-project/ray
5. NVIDIA Triton Inference Server https://github.com/triton-inference-server/server