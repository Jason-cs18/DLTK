# syntax=docker/dockerfile:1

# Use a standard NVIDIA CUDA base image for Ubuntu 20.04 with CUDA 11.8 and cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# pre-installed libaries: CUDA 11.8, CUDNN 8.9.6

# Set environment variables for non-interactive installation and Python output buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Set default locale
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Ensure pip cache is not used during build to save space, and paths are updated
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Change apt sources to a mirror if desired (e.g., Aliyun for users in China)
# Ensure these RUN commands are robust
RUN apt-get update && apt-get install -y curl && \
    sed -i 's#http://archive.ubuntu.com/ubuntu/#http://mirrors.aliyun.com/ubuntu/#' /etc/apt/sources.list && \
    sed -i 's#http://security.ubuntu.com/ubuntu/#http://mirrors.aliyun.com/ubuntu/#' /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/*

# Add the NVIDIA repository for TensorRT packages
# Note: apt-key is deprecated. For future robustness, consider using gpg --dearmor with signed-by in sources.list.d
# However, keeping it for consistency with the original if it works.
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - && \
    curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    rm -rf /var/lib/apt/lists/*

# Update apt-get sources and install necessary system packages
# Install Python 3.9 from deadsnakes PPA
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    git wget curl vim && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
    # Install pip for Python 3.9
    # Usually python3.9-distutils allows 'python3.9 -m ensurepip' or get-pip.py
    # Or install python3.9-pip if directly available and reliable from PPA
    # Let's use get-pip.py for a clean pip install for python3.9
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
    # Install TensorRT Libraries (version 8.5.3-1)
    && apt-get install -y --no-install-recommends \
    libnvinfer8=8.5.3-1+cuda11.8 \
    libnvinfer-dev=8.5.3-1+cuda11.8 \
    libnvparsers8=8.5.3-1+cuda11.8 \
    libnvparsers-dev=8.5.3-1+cuda11.8 \
    libnvonnxparsers8=8.5.3-1+cuda11.8 \
    libnvonnxparsers-dev=8.5.3-1+cuda11.8 \
    libnvinfer-plugin8=8.5.3-1+cuda11.8 \
    libnvinfer-plugin-dev=8.5.3-1+cuda11.8 \
    # libnvinfer-samples was commented out in original, keeping it so
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as the default python and python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip for Python 3.9 and install general Python libraries
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install \
    pytorch-lightning \
    numpy pandas matplotlib scikit-learn tqdm Pillow opencv-python requests pyyaml

# Install TensorRT Python bindings via NVIDIA’s PyPI for Python 3.9
# RUN python3.9 -m pip install nvidia-pyindex && \
#     python3.9 -m pip install nvidia-tensorrt==8.4.3.1 --index-url https://pypi.ngc.nvidia.com

# Install PyTorch 2.2.2, TorchVision 0.17.2 compatible with CUDA 11.8 for Python 3.9
RUN python3.9 -m pip install \
    torch==2.2.2+cu118 \
    torchvision==0.17.2+cu118 \
    torchaudio==2.2.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install Torch-TensorRT 2.2.0 for Python 3.9
# Using --no-deps to bypass problematic 'tensorrt' dependency build if nvidia-tensorrt is already installed
# RUN python3.9 -m pip install --no-deps \
#     torch-tensorrt==2.2.0+cu118 \
#     --extra-index-url https://download.pytorch.org/whl/cu118

# Install timm rich lightning onnx onnxruntime-gpu
# Removed "some-package" from the original line
RUN python3.9 -m pip install \
    timm rich lightning onnx onnxruntime-gpu==1.16.1 numpy~=1.26.4 \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install ONNX optimization tools
RUN pip install onnx onnxconverter-common -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install additional libraries for model development
RUN pip install jsonargparse[signatures]>=4.27.7 tensorboard -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install tensorrt engine and build trtexec
RUN apt-get update && apt-get install -y libnvinfer-samples=8.5.3-1+cuda11.8 && \
    cd /usr/src/tensorrt/samples/trtexec && make

COPY . /codebase

RUN pip install /codebase/tensorrt-8.5.3.1-cp39-none-linux_x86_64.whl && \
    pip install pycuda -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    pip install numpy==1.23 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# use trtexec to convert onnx model to tensorrt engine
# /usr/src/tensorrt/bin/trtexec --onnx=xxx.onnx --saveEngine=xxx.trt

# Install TenrorRT Model Optimizer
# RUN pip install "nvidia-modelopt[all]" --extra-index-url https://pypi.nvidia.com -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install nsys
RUN cd /tmp && \
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2023_4_1_97/nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
    apt-get install -y ./nsight-systems-2023.4.1_2023.4.1.97-1_amd64.deb && \
    rm -rf /tmp/*

RUN pip install "ray[serve]" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Set the working directory inside the container
WORKDIR /codebase

# Copy your project code into the container
# Make sure your Dockerfile is at the root of your project
# COPY . /codebase

# Set the default command to run when the container starts
# 'bash' is useful for interactive exploration/experimentation within the environment
CMD ["bash"]