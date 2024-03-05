# Anconda

```bash
conda create -n my_env python=3.10 # create a new conda environment named my_env with Python 3.10
pip freeze > requirements.txt # freeze the current package list and save it to requirements.txt
conda env export --no-builds | grep -v "prefix" > environment.yml # export the current environment to environment.yml
conda env create -f environment.yaml # create a new conda environment from environment.yml
pip install -r requirements.txt # install the packages listed in requirements.txt
```