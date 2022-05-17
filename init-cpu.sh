#!/bin/bash

# create conda env
conda create -n kgpip python=3.7 -y

conda activate kgpip

pip install -U pip

pip install torch==1.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl==0.5.3 -f https://data.dgl.ai/wheels/repo.html

# install requirements
pip install -r requirements.txt


python utils/patch_autosklearn_verbose.py
python utils/patch_dgl_verbose.py