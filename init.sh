#!/bin/bash

# create conda env
conda create -n kgpip python=3.7 -y

conda activate kgpip

pip install -U pip

# install requirements
pip install -r requirements.txt
