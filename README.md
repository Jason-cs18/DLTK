# DLTK
This guide offers an in-depth introduction to deep learning toolkits, including model design and deployment, through a practical image classification example using Vision Transformer.

<p align="center">
  <img src="[http://some_place.com/image.png](https://github.com/Jason-cs18/DLTK/blob/main/imgs/dlsys_outline.png)" />
</p>

## Table of Contents
- [DLTK](#dltk)
  - [Table of Contents](#table-of-contents)
  - [Vision Transformer](#vision-transformer)
  - [Model Design via Pytorch](#model-design-via-pytorch)
    - [Fine-tune Vision Transformer on Custom Dataset](#fine-tune-vision-transformer-on-custom-dataset)
    - [Experiment Tracking with W\&B](#experiment-tracking-with-wb)
    - [Evaluate the Pre-Trained Vision Transformer on New Images](#evaluate-the-pre-trained-vision-transformer-on-new-images)
    - [Accelerate the Inference of Vision Transformer](#accelerate-the-inference-of-vision-transformer)
  - [Model Design via Pytorch-Lightning](#model-design-via-pytorch-lightning)
    - [Fine-tune Vision Transformer on Custom Dataset](#fine-tune-vision-transformer-on-custom-dataset-1)
    - [Experiment Tracking with W\&B](#experiment-tracking-with-wb-1)
    - [Evaluate the Pre-Trained Vision Transformer on New Images](#evaluate-the-pre-trained-vision-transformer-on-new-images-1)
    - [Accelerate the Inference of Vision Transformer](#accelerate-the-inference-of-vision-transformer-1)
  - [Model Design via HuggingFace](#model-design-via-huggingface)
    - [Fine-tune Vision Transformer on Custom Dataset](#fine-tune-vision-transformer-on-custom-dataset-2)
    - [Experiment Tracking with W\&B](#experiment-tracking-with-wb-2)
    - [Evaluate the Pre-Trained Vision Transformer on New Images](#evaluate-the-pre-trained-vision-transformer-on-new-images-2)
    - [Accelerate the Inference of Vision Transformer](#accelerate-the-inference-of-vision-transformer-2)
  - [Web Demo via Gradio](#web-demo-via-gradio)
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

|Transformer|Vision Transformer|
|:---:|:---:|
|![](https://camo.githubusercontent.com/022f6ad1b0745d754a8c6cb474a8bd458b0de4d028558607456387a347b78d80/68747470733a2f2f64326c2e61692f5f696d616765732f7472616e73666f726d65722e737667)|![](https://camo.githubusercontent.com/5c9e02651b64a9113981be3d72942564778bee4b86a5211ad59d452da8f30a1f/68747470733a2f2f64326c2e61692f5f696d616765732f7669742e737667)|

## Model Design via Pytorch

### Fine-tune Vision Transformer on Custom Dataset

### Experiment Tracking with W&B

### Evaluate the Pre-Trained Vision Transformer on New Images

### Accelerate the Inference of Vision Transformer

## Model Design via Pytorch-Lightning

### Fine-tune Vision Transformer on Custom Dataset

### Experiment Tracking with W&B

### Evaluate the Pre-Trained Vision Transformer on New Images

### Accelerate the Inference of Vision Transformer

## Model Design via HuggingFace

### Fine-tune Vision Transformer on Custom Dataset

### Experiment Tracking with W&B

### Evaluate the Pre-Trained Vision Transformer on New Images

### Accelerate the Inference of Vision Transformer

## Web Demo via Gradio

## Model Deployment with NVIDIA Triton

### Model Conversion

### Model Optimization

### Performance Monitoring

## Model Deployment with Anyscale Ray

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