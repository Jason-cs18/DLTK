# Text-to-video web applications
In this tutorial, we will guide you through the process of developing text-to-video web applications using popular libraries (Transformers, Gradio, FastAPI).

## Setup
```bash
conda create -n t2v python=3.8
conda activate t2v
pip install -r requirements.txt
```

## File Structure
```bash
└── [4.0K]  video_generation
    ├── [4.0K]  benchmark
    │   └── [1.9K]  inference.py
    ├── [ 614]  readme.md
    └── [4.0K]  web_develop
        ├── [4.0K]  flagged
        ├── [ 12K]  main.py
        ├── [ 825]  predict.py
        ├── [4.0K]  __pycache__
        │   ├── [9.7K]  main.cpython-38.pyc
        │   └── [ 935]  predict.cpython-38.pyc
        └── [4.0K]  templates
            ├── [ 488]  index.html
            ├── [ 510]  login.html
            ├── [ 180]  private.html
            └── [ 244]  _site_map.html
```

## Benchmark text-to-video models locally
Benchmark on NVIDIA GTX 1070 (8G)
```bash
python -W ignore ./benchmark/inference.py
```

|Metric|Pytorch (engine)|
|:---:|:---:|
|Latency (s)|284|
|GPU Memory (G)|~6.5|

## Develop a web application

```bash
cd web_develop
uvicorn main:app --reload
```
