import os
import sys
import time
import random
import logging
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from tqdm import tqdm
from ase.units import fs, Hartree, kcal, mol
from easydict import EasyDict

# Add project root to sys.path
sys.path.append('/home/suqun/model/LumiForce')

# Project imports
from model.foundation_model import LumiForce
from dataset.SPICE import TorchDataset

# Use high precision matmul if available
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_index = ''

# Load test dataset and dataloader
test_set = TorchDataset(data_path='/home/suqun/data/H2O_dataset/pbc_h2o.pt', index=test_index)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)


def load_model(model, ckpt_path, device='cpu', single=True):
    """Load model weights from checkpoint with optional key name handling."""
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get('model', state)

    if single:
        new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict

    model_state_dict = model.state_dict()
    counter = 0

    for k, v in model_state_dict.items():
        if k in new_state_dict and new_state_dict[k].shape == v.shape:
            model_state_dict[k] = new_state_dict[k]
            counter += 1

    if counter > 20:
        print('Checkpoint loaded successfully.')

    model.load_state_dict(model_state_dict, strict=True)
    return model


def evaluate(model, loader, device):
    """Evaluate the model and compute MAE and RMSE for energy and force."""
    model.eval()
    pred_energy, true_energy = [], []
    pred_force, true_force = [], []

    for data in loader:
        data = data.to(device)
        label = data.y.view(-1)
        dy = data.dy

        pred, pdy = model(data, mean=1, std=1, if_train=False)

        pred_energy.append(pred.detach().cpu().numpy())
        true_energy.append(label.detach().cpu().numpy())

        pred_force.append(pdy.detach().cpu().numpy())
        true_force.append(dy.detach().cpu().numpy())

    # Convert to arrays and units (kJ/mol)
    pred_energy = np.concatenate(pred_energy) * Hartree * 1000
    true_energy = np.concatenate(true_energy) * Hartree * 1000

    pred_force = np.concatenate(pred_force) * Hartree * 1000
    true_force = np.concatenate(true_force) * Hartree * 1000

    # Compute metrics
    energy_mae = np.mean(np.abs(pred_energy - true_energy))
    force_mae = np.mean(np.abs(pred_force - true_force))

    energy_rmse = np.sqrt(np.mean((pred_energy - true_energy) ** 2))
    force_rmse = np.sqrt(np.mean((pred_force - true_force) ** 2))

    return energy_mae, force_mae, energy_rmse, force_rmse


def main():
    model = LumiForce()
    ckpt_path = ""
    model = load_model(model, ckpt_path, device=device).to(device)

    energy_mae, force_mae, energy_rmse, force_rmse = evaluate(model, test_loader, device)

    print(f"Energy MAE:  {energy_mae:.6f} kJ/mol")
    print(f"Force MAE:   {force_mae:.6f} kJ/mol/Å")
    print(f"Energy RMSE: {energy_rmse:.6f} kJ/mol")
    print(f"Force RMSE:  {force_rmse:.6f} kJ/mol/Å")


if __name__ == "__main__":
    main()
