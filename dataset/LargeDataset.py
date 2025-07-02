import numpy as np
from rdkit import Chem
import os
import sys
import tempfile
import shutil
from torch_geometric.data import Batch, Data, Dataset  # , InMemoryDataset
from torch.utils.data import DataLoader
import logging
import warnings
import pickle
from ase.db import connect
import torch
from tqdm import tqdm
import lmdb
from dataset.utils import AtomicEnergiesBlock
from torch_scatter import scatter


class HLDataset(Dataset):
    def __init__(self, datapath, dataset_name):
        super(HLDataset, self).__init__()

        self.dataset_name = dataset_name
        self.datapath = datapath
        self.db = connect(os.path.join(self.datapath, f'{self.dataset_name}.db'))

        self.data_length = len(self.db)

        logging.info(f'data length: {self.data_length}')

    def len(self):
        return self.data_length

    def get(self, idx):
        db_row = self.db.get(idx + 1)

        z = torch.from_numpy(db_row.numbers.copy()).long()
        positions = torch.from_numpy(db_row.positions.copy()).float()
        y = torch.from_numpy(np.array(db_row.data["energy"]).copy()).float().squeeze(-1)
        forces = torch.from_numpy(np.array(db_row.data["forces"]).copy()).float()
        molecule_size = len(positions)
        data = Data(z=z, pos=positions, y=y.unsqueeze(-1), dy=forces, molecule_size=molecule_size)

        # logging.info(data)

        return data

def get_mean_std(data_loader):
    ys = torch.cat([batch.y for batch in data_loader])
    return ys.mean(), ys.std()

# def get_mean_std(dataset):
#     """
#     Args:
#         dataset: list-like of graph data (each has `z` and `y`)
#         atomic_energies_fn: instance of AtomicEnergiesBlock
#     Returns:
#         mean, std of per-atom interaction energy
#     """
#     atomic_energies_fn = AtomicEnergiesBlock()
#
#     avg_interaction_energies = []
#
#     for data in dataset:
#         z = data.z
#         y = data.y
#         logging.info(y)
#
#         node_e0 = atomic_energies_fn(z)  # [n_atoms]
#         base_energy = node_e0.sum()     # scalar
#         avg_interaction = (y - base_energy) / z.numel()
#         avg_interaction_energies.append(avg_interaction)
#
#     avg_interaction_energies = torch.stack(avg_interaction_energies)
#     atom_mean = avg_interaction_energies.mean()
#     atom_std = avg_interaction_energies.std()
#
#     return atom_mean, atom_std
