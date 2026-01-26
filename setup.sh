#!/bin/bash
# =============================================================================
# Setup Script for vdat-ml
# =============================================================================
#
# Usage:
#   ./setup.sh gpu    # GPU environment
#   ./setup.sh cpu    # CPU environment
#
# =============================================================================

set -e

ENV_TYPE=${1:-gpu}

if [ "$ENV_TYPE" == "gpu" ]; then
    ENV_NAME="vdat-ml"
    echo "=========================================="
    echo "Setting up GPU environment: $ENV_NAME"
    echo "=========================================="
    
    # Create conda env
    echo "[1/3] Creating conda environment..."
    conda env create -f env-gpu.yaml
    
    # Activate and install PyG
    echo "[2/3] Installing PyTorch Geometric..."
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
    pip install torch-geometric
    
    echo "[3/3] Verifying installation..."
    python -c "
import torch
import torch_geometric
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'PyG: {torch_geometric.__version__}')
print('✓ GPU environment ready!')
"

elif [ "$ENV_TYPE" == "cpu" ]; then
    ENV_NAME="vdat-ml-cpu"
    echo "=========================================="
    echo "Setting up CPU environment: $ENV_NAME"
    echo "=========================================="
    
    # Create conda env
    echo "[1/3] Creating conda environment..."
    conda env create -f env-cpu.yaml
    
    # Activate and install PyG
    echo "[2/3] Installing PyTorch Geometric..."
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
    pip install torch-geometric
    
    echo "[3/3] Verifying installation..."
    python -c "
import torch
import torch_geometric
print(f'PyTorch: {torch.__version__}')
print(f'PyG: {torch_geometric.__version__}')
print('✓ CPU environment ready!')
"

else
    echo "Usage: ./setup.sh [gpu|cpu]"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test:"
echo "  python src/train.py experiment=debug"
