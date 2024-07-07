# MLSys Template
This repository is the official codebase for the [blog](https://jason-cs18.github.io/ml-engineering/ml_engineer.html).

## Table of contents
- [x] Quickstart ([Code](https://github.com/Jason-cs18/DLTK/tree/main/mlsys_template/quickstart))
    - [x] [Find a suitable model for your needs](https://jason-cs18.github.io/ml-engineering/model_selection.html)
    - [x] [Build a simple interative demo](https://jason-cs18.github.io/ml-engineering/web_demo.html)
- [ ] Let's make models fast and serve them with multiple requests
    - [ ] [Accelerate inference with ONNX and TensorRT](https://jason-cs18.github.io/ml-engineering/inference_engine.html)
    - [ ] [Serve inference endpoints on Ray Serve](https://jason-cs18.github.io/ml-engineering/inference_server.html)
- [ ] Let's train a model on your custom data
    - [ ] [Fine-tune DETR on your own dataset with Ray Train](https://jason-cs18.github.io/ml-engineering/detr_train.html)
    - [ ] [Hyperparameter tuning with Ray Tune](https://jason-cs18.github.io/ml-engineering/detr_tune.html)
- [ ] Monitor and debug deployed models
- [ ] Dive into generative AI
    - [ ] LLM and MLLM
    - [ ] Diffusion and its variants
    - [ ] CLIP and its variants
- [x] Engineering Tools
    - [x] [Web scraper](https://jason-cs18.github.io/ml-engineering/web_scraper.html)
    - [x] [Web development with FastAPI](https://jason-cs18.github.io/ml-engineering/fastapi.html)
    - [x] [Docker and docker-compose](https://jason-cs18.github.io/ml-engineering/docker.html)


## Setup
```bash
conda create -n mltemplate python=3.9
pip install -r requirements.txt
```