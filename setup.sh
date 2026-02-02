#!/bin/bash

set -e
ENV_TYPE=${1:-gpu}

if [ "$ENV_TYPE" == "gpu" ]; then
    conda env create -f env-gpu.yaml
    eval "$(conda shell.bash hook)"
    conda activate vdat-ml
    echo "Complete. Now can run: conda activate vdat-ml"

# does not work right now
elif [ "$ENV_TYPE" == "cpu" ]; then
    conda env create -f env-cpu.yaml
    eval "$(conda shell.bash hook)"
    conda activate vdat-ml-cpu
    echo "Complete. Now can run: conda activate vdat-ml-cpu"
else
    echo "Usage: ./setup.sh [gpu|cpu]"
    exit 1
fi
