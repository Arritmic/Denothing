#!/bin/bash

#1) Create and activate environment
ENVS=$(conda info --envs | awk '{print $1}' )
if [[ $ENVS = *"Denothing"* ]]; then
   source ~/anaconda3/etc/profile.d/conda.sh
   conda activate Denothing
else
   echo "Creating a new conda environment for Denothing project..."
   conda env create -f environment.yml
   source ~/anaconda3/etc/profile.d/conda.sh
   conda activate Denothing
   #python setup.py install
   #exit
fi;

