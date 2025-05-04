# Model development

Using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) to efficiently train image classification models, enabling modular deep learning projects and easy experimentation through configurable settings.

## Installation
```bash
conda create -n dltk_model python=3.9 -y
conda activate dltk_model
pip install lightning
pip install timm
pip install tensorboard
pip install "jsonargparse[signatures]"
```

## File structure

```bash
.
├── configs # experiment configs
├── datasets.py # dataset module
├── models.py # lightning model module
├── main.py # profiler, training & evaluation
├── transformers_extension.ipynb # using lightning to train models from transformers 
└── readme.md # notes
```

## How to run experiments

Modify configs for your needs

```yaml
# ./config/resnet.yaml
model: LitResNet18 
trainer:
  max_epochs: 10
```

```yaml
# ./config/vit.yaml
model: LitVisionTransformer 
trainer:
  max_epochs: 5
```

Print training configs

```bash
python -W ignore main.py fit --config configs/resnet.yaml --print_config
```

Train and evaluate your model

```bash
# train a resnet18 on MNIST
python -W ignore main.py fit --config configs/resnet.yaml
# train a vision transformer on MNIST
python -W ignore main.py fit --config configs/vit.yaml
```

Test your model

```bash
python -W ignore main.py test --model=LitResNet18 --ckpt_path xxx
```

Visualize training logs

```bash
tensorboard --logdir .
```


## NLP examples
Because NLP community often uses [transformers](https://huggingface.co/docs/transformers/index) to share their models and datasets, we provide an [hands-on notebook (TBD)](https://github.com/Jason-cs18/DLTK/blob/main/model_development/transformers_extension.ipynb) to implement Lightning training logic with transformers.