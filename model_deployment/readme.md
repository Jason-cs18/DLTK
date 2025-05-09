# Model deployment
_A hands-on tutorial for efficiently deploying deep learning models. The topic includes inference engine, model profiling, and inference server. The goal is to provide a comprehensive guide for deploying deep learning models in production environments._

## Table of contents
- [Model deployment](#model-deployment)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Accelerate model with inference engine](#accelerate-model-with-inference-engine)
    - [ONNX test](#onnx-test)
    - [TensorRT test](#tensorrt-test)
  - [Model profiling](#model-profiling)
    - [PyTorch Profile](#pytorch-profile)
    - [Nsight Systems](#nsight-systems)
  - [Deploy models on inference server](#deploy-models-on-inference-server)
    - [Deploy TensorRT model with NVIDIA Triton](#deploy-tensorrt-model-with-nvidia-triton)
    - [Send request to Triton](#send-request-to-triton)

## Installation
We recommend using docker for model deployment. The following instructions are for building a docker image for the DLTK model deployment.

```bash
docker build -t dltk:latest .
```

After building the image, you can run the container using the following command:

```bash
docker run --rm --runtime=nvidia -v /mnt/platform/code/DLTK/:/codebase --gpus all -it dltk:latest
```

## Accelerate model with inference engine

### ONNX test
Run ONNX inference
```bash
cd model_deployment
python speed_test.py
```
After running the above command, you will see the following output:

```bash
     Latency Results for ResNet50 (ms/batch)      
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Batch Size ┃ PyTorch GPU ┃ ONNX CPU ┃ ONNX GPU ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│     1      │    3.48     │   1.26   │   1.96   │
├────────────┼─────────────┼──────────┼──────────┤
│     8      │    3.45     │   3.55   │   2.58   │
├────────────┼─────────────┼──────────┼──────────┤
│     16     │    3.55     │   5.16   │   3.16   │
├────────────┼─────────────┼──────────┼──────────┤
│     32     │    3.85     │   9.16   │   3.20   │
├────────────┼─────────────┼──────────┼──────────┤
│     64     │    5.00     │  16.91   │   3.30   │
├────────────┼─────────────┼──────────┼──────────┤
│    128     │    8.07     │  32.46   │   3.89   │
└────────────┴─────────────┴──────────┴──────────┘
```

### TensorRT test
Export ONNX to TensorRT (tf32, fp16)
```bash
# tp32
/usr/src/tensorrt/bin/trtexec --onnx=resnet18_dynamic_batch.onnx --saveEngine=resnet50_engine_fp16.trt --minShapes=input_image:1x1x28x28 --optShapes=input_image:32x1x28x28 --maxShapes=input_image:128x1x28x28
# fp16
/usr/src/tensorrt/bin/trtexec --onnx=resnet18_dynamic_batch.onnx --saveEngine=resnet50_engine_fp16.trt --minShapes=input_image:1x1x28x28 --optShapes=input_image:32x1x28x28 --maxShapes=input_image:128x1x28x28 --fp16 
```

Run TensorRT inference
```bash
python tensorrt_test.py
```

After running the above command, you will see the following output:

```bash
# tf32
            Performance Results (ms/batch)            
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃  Batch Size  ┃ Avg Latency (ms) ┃ Throughput (FPS) ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│      1       │            2.971 │           336.56 │
│      8       │            2.992 │          2673.73 │
│      16      │            3.001 │          5330.94 │
│      32      │            3.039 │         10529.39 │
│      64      │            3.196 │         20026.25 │
│     128      │            3.818 │         33521.31 │
└──────────────┴──────────────────┴──────────────────┘
Explicitly deleting TensorRT engine object before script exit...

# fp16
            Performance Results (ms/batch)            
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃  Batch Size  ┃ Avg Latency (ms) ┃ Throughput (FPS) ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│      1       │            0.857 │          1167.05 │
│      8       │            0.816 │          9809.14 │
│      16      │            0.824 │         19411.92 │
│      32      │            0.834 │         38375.70 │
│      64      │            0.894 │         71587.56 │
│     128      │            1.074 │        119129.94 │
└──────────────┴──────────────────┴──────────────────┘
Explicitly deleting TensorRT engine object before script exit...           
```

## Model profiling

### PyTorch Profile

PyTorch profile is helpful to find the bottlenecks in your model. For example, the inference latency of ResNet50 on V100 GPU is `~208` ms and CPU operations take `~205` ms. Thus, this workload is memory bound where the GPU is waiting for the CPU to finish its operations. To optimize the model, we can use layer fusion, quantization, and other techniques to reduce the CPU operations.

```bash
python pytorch_profile.py
```

### Nsight Systems

Nsight Systems provides a timeline view of the model execution. It shows the GPU and CPU utilization, memory usage, and other metrics. You can use it to identify the bottlenecks in your model and optimize it.

Install Nsight Systems

```bash
cd /tmp && \
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2023_4_1_97/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
    apt-get install -y ./nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
    rm -rf /tmp/*
```

Test Nsight Systems on ResNet50

```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o nsight_report -f true -x true python nsys_profile.py
```

## Deploy models on inference server

### Deploy TensorRT model with NVIDIA Triton
```bash
# start a triton server with tensorrt model
docker run --gpus=all --rm --net=host \
-v /mnt/platform/code/DLTK/model_deployment/tensorrt_models:/models \
nvcr.io/nvidia/tritonserver:23.03-py3 tritonserver --model-repository=/models

# models directory structure
./tensorrt_models/
└── resnet50
    ├── 1
    │   └── model.plan # TensorRT engine file
    └── config.pbtxt

# If you want to use fp16 model, please use the outputIOFormats=fp16:chw and inputIOFormats=fp16:chw
/usr/src/tensorrt/bin/trtexec --onnx=resnet18_dynamic_batch.onnx \
--saveEngine=resnet50_engine_fp16.trt\
--minShapes=input_image:1x1x28x28 \
--optShapes=input_image:32x1x28x28 \
--maxShapes=input_image:128x1x28x28 \
--fp16 \
--inputIOFormats=fp16:chw \
--outputIOFormats=fp16:chw
```

### Send request to Triton
```bash
# install triton client
pip install nvidia-pyindex tritonclient[http]
# test an image
python triton_client.py
```