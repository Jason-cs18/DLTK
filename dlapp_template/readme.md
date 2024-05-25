# ML deployment template
Unlike research papers, the machine learning product development aims at enhancing users' experience and cutting costs. Hence, the iterative deployment (train -> deploy -> optimize) is requisite. To reach this objective, we expect that machine learning engineers not only be familiar with training techniques but also have a good command of the deployment and optimization process.

Unfortunately, the existing tutorials or books typically involve one or a few aspects of ML deployment. As a result, it is difficult to learn the entire process.

In this document, we will teach you to deploy and optimize a deep learning model from model selection to model optimization. Even though we take video analytics as an example, you can readily modify the template for customized applications.

> If you are not familiar with deep learning, please go through [Harvard CS197: AI Research Experiences](https://www.cs197.seas.harvard.edu/) (the best course for AI research).

> If you want know more about AIGC models, please navigate to [LLM deployment template](./llm_deploy.md).


- Table of contents
    1. [Environment](#environment)
    2. [Choose a acceptable base model](#choose-a-acceptable-base-model)
    3. Build a web-based demo
        - Use Gradio
        - Use Streamlit
    4. Finetune on the custom dataset
        - Use Transformer
        - Use PyTorch-Lightning
    5. Accelerate with production-level inference engines
        - Use ONNX
        - Use TensorRT
        - Use OpenVINO
    6. Deploy on the inference server
    7. Monitor the application performance
    8. Profile the inference pipeline and optimize it with deep compression
    9. Use advanced parallel computing techniques to improve resource utilization
    10. References

## Environment
In production, we usually use docker to containerize application runtime environments. Thus, we build a docker image with all the necessary dependencies. If you do not familiar with docker, please go through the [docker tutorial](https://docs.docker.com/get-started/). 

> If you want to deploy multiple models with docker, please refer to [docker-compose](https://docs.docker.com/compose/).

|Deployment stages|Dockerfile|Command|
|:---|:---|:---|
|1, 2, 3, 6|`./docker/model_train/Dockerfile`|`docker image build -t production_train:latest -f ./docker/model_train/Dockerfile .`|
|4|`./docker/deploy_onnx/Dockerfile`||
|4|`./docker/deploy_openvino/Dockerfile`||
|4|`./docker/deploy_tensorrt/Dockerfile`||
|5|`./docker/model_triton/Dockerfile`||
|7|`./docker/model_optimize/Dockerfile`||
|8|`./docker/parallel_compute/Dockerfile`||

## 1. Choose a acceptable base model
Usually, we choose the most popular (e.g., most download or trending) model from the well-known model zoo (e.g., [HuggingFace](https://huggingface.co/models) and [ModelScope](https://modelscope.cn/models)). In our example, we choose the most downloaded model [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) from [HuggingFace](https://huggingface.co/models) as our base model for video analytics.

## 2. Build a web-based demo
Build a web-based demo with the pre-trained model is easy with the help of [Gradio](https://gradio.app/) and [Streamlit](https://streamlit.io/).

### Use Gradio

### Use Streamlit

## 3. Finetune on the custom dataset

Pre-trained models are usually trained on a large-scale dataset. However, it achieves only limited performance due to domain shifts. Thus, we may need to fine-tune the model on a custom dataset. As shown below, we provide two options to fine-tune the model.

### Use Transformer

### Use PyTorch-Lightning

## 4. Accelerate with production-level inference engines
PyTorch is a popular deep learning framework. However, it is slow when deploying on the production environment. Thus, we need to accelerate the inference process with production-level inference engines. As shown below, we provide three options to accelerate the inference process.

### Use ONNX

### Use TensorRT (NVIDIA)

### Use OpenVINO (Intel)

## 5. Deploy on the inference server
To enable cost-effective inference with server-based optimizations, we usually deploy the application on the inference server. As shown below, we teach you deploy the application on the NVIDIA Triton inference server.

## 6. Monitor the application performance
When AI is deployed in the production environment, we need to monitor the application performance. As shown below, we guide you monitor the accuracy and latency with xxx.
## 7. Profile the inference pipeline and optimize it with deep compression
To improve the performance of the inference pipeline in further, we usually profile the inference pipeline and optimize it with advanced deep compression. As shown below, we present the standard profiling process and accelerate the inference with xxx deep compression techniques.

## 8. Use advanced parallel computing techniques to improve resource utilization
However, the performance of the inference pipeline is still limited due to the limited resource utilization. Thus, we often use advanced parallel computing techniques to improve the resource utilization. As shown below, we present the standard parallel computing techniques and accelerate the inference with xxx parallel computing techniques.

## References

<!-- ### Environments -->