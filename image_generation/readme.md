# Text-to-image web applications
In this tutorial, we will guide you through the process of developing text-to-image web applications using popular libraries (`Transformers, Gradio, FastAPI`).


<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#setup">Setup</a></li>
    <li><a href="#file-structure">File structure</a></li>
    <li><a href="#benchmark-text-to-image-models-locally">Benchmark text-to-image models locally</a></li>
    <li><a href="#develop-text-to-image-web-application">Develop text-to-image web application</a></li>
  </ol>
</details>

## Setup
```bash
conda create -n t2i python=3.8
conda activate t2i
pip install -r requirements.txt
```

## File structure
```bash
.
├── [4.0K]  image_generation
│   ├── [4.0K]  benchmark
│   │   └── [3.8K]  inference.py
│   ├── [4.0K]  conf
│   │   ├── [ 171]  infer_config_onnx.yaml
│   │   └── [ 174]  infer_config_pytorch.yaml
│   ├── [4.0K]  docs
│   │   ├── [1.7K]  benchmark_infer.ipynb
│   │   └── [ 951]  run_web.ipynb
│   ├── [1.9K]  readme.md
│   ├── [5.4K]  requirements.txt
│   └── [4.0K]  web_devlop
│       ├── [4.0K]  flagged
│       ├── [ 12K]  main.py
│       ├── [1.8K]  predict.py
│       ├── [4.0K]  __pycache__
│       │   ├── [9.5K]  main.cpython-38.pyc
│       │   └── [1.1K]  predict.cpython-38.pyc
│       └── [4.0K]  templates
│           ├── [ 488]  index.html
│           ├── [ 510]  login.html
│           ├── [ 180]  private.html
│           └── [ 244]  _site_map.html
```

## Benchmark text-to-image models locally

1. Tutorial: `benchmark_infer.ipynb`
2. Reproduce the results of the tutorial: 
   1. `python -W ignore ./benchmark/inference.py pytorch`  
   2. `python -W ignore ./benchmark/inference.py onnx`

## Develop text-to-image web application

1. Tutorial: `run_web.ipynb`
2. Run the web application locally:
   1. `cd ./web_develop`
   2. `uvicorn main:app --reload`
   3. API documentation: `http://127.0.0.1:8000/docs`

## Web application demo
![web_demo](https://github.com/Jason-cs18/DLTK/blob/main/imgs/text2img.gif)