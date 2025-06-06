# Model development

_Using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) to efficiently train image classification models, enabling modular deep learning projects and easy experimentation through configurable settings._

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
  default_root_dir: ./logs/resnet
```

```yaml
# ./config/vit.yaml
model: LitVisionTransformer 
trainer:
  max_epochs: 10
  default_root_dir: ./logs/vit
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
# test a pre-trained resnet18 on MNIST
python -W ignore main.py test --config configs/resnet.yaml --ckpt_path xxx
# after running the above command, you will see the following output
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test_acc            0.9890000224113464
        test_loss           0.03381314501166344
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# test a pre-trained vision transformer on MNIST
python -W ignore main.py test --config configs/vit.yaml --ckpt_path xxx
```

Visualize training logs

```bash
tensorboard --logdir .
```

## NLP examples

The NLP community frequently uses [Transformers](https://huggingface.co/docs/transformers/index) to share pre-trained models and datasets. To support this, we provide a [hands-on notebook](https://github.com/Jason-cs18/DLTK/blob/main/model_development/transformers_extension.ipynb) (TBD) that demonstrates how to implement Lightning training logic with Transformers.

### Useful links
1. Lightning-AI. "[Finetune Hugging Face BERT with PyTorch Lightning](https://lightning.ai/lightning-ai/studios/finetune-hugging-face-bert-with-pytorch-lightning)". Tech blog.