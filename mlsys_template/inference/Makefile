.PHONY: help run

export CUDA_LAUNCH_BLOCKING=1

help:
	@echo "make help: print all available commands"
	@echo "make onnx: measure DETR-onnx on a single image"
	@echo "make export: export all requirements to requirements.txt"
	@echo "make clean: remove all temp files"
	@echo "make build: install all dependencies"

clean:
	rm -rf __pycache__

build:
	pip install -r requirements.txt

export:
	pip freeze > requirements.txt

onnx:
	python onnx_infer.py

ray:
	serve run config.yaml



