#!/bin/bash

# Step 1: Create a new conda environment with Python 3.10
conda create -n liten python=3.10 -y

# Step 2: Activate the new environment
source activate liten

# Step 3: Install PyTorch 2.4.0, TorchVision 0.19.0, and Torchaudio 2.4.0 with CUDA 12.1 support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# Step 4: Install PyG and related libraries
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

pip install torch_geometric
pip install rdkit
pip install ase
pip install pyyaml
pip install scikit-learn
pip install easydict

echo "Environment setup complete!"
