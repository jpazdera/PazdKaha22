#!/bin/bash
echo Please enter a name for your new Anaconda environment:
read ENV_NAME
conda create -n $ENV_NAME python=3 numpy scipy mkl cython ipykernel
source activate $ENV_NAME
python -m ipykernel install --user --name $ENV_NAME --display-name "$ENV_NAME"
python setup_cmr2.py install
