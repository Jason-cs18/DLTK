# Text-to-video web applications
In this tutorial, we will guide you through the process of developing text-to-video web applications using popular libraries (Transformers, Gradio, FastAPI).

## Setup
```bash
conda create -n t2v python=3.8
conda activate t2v
pip install -r requirements.txt
```

## File Structure
xxx

## Benchmark text-to-video models locally
Benchmark on NVIDIA GTX 1070 (8G)
```bash
python -W ignore ./benchmark/inference.py
```

|Metric|Pytorch (engine)|
|:---:|:---:|
|Latency (s)|284|
|GPU Memory (G)|~6.5|

## 
