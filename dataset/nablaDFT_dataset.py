"""Module describes PyTorch Geometric interfaces for nablaDFT datasets"""

import logging
import os
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from ase.db import connect
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdmolops

logger = logging.getLogger(__name__)


class PyGNablaDFT(InMemoryDataset):
    """Pytorch Geometric interface for nablaDFT datasets.

    Based on `MD17 implementation <https://github.com/atomicarchitects/equiformer/blob/master/datasets/pyg/md17.py>`_.

    .. code-block:: python
        from nablaDFT.dataset import PyGNablaDFT

        dataset = PyGNablaDFT(
            datapath="./datasets/",
            dataset_name="dataset_train_tiny",
            split="train",
        )
        sample = dataset[0]

    .. note::
        If split parameter is 'train' or 'test' and dataset name are ones from nablaDFT splits
        (see nablaDFT/links/energy_databases.json), dataset will be downloaded automatically.

    Args:
        datapath (str): path to existing dataset directory or location for download.
        dataset_name (str): split name from links .json or filename of existing file from datapath directory.
        split (str): type of split, must be one of ['train', 'test', 'predict'].
        transform (Callable): callable data transform, called on every access to element.
        pre_transform (Callable): callable data transform, called on every element during process.
    """

    db_suffix = ".db"

    @property
    def raw_file_names(self) -> List[str]:
        return [(self.dataset_name + self.db_suffix)]

    @property
    def processed_file_names(self) -> str:
        return f"{self.dataset_name}_{self.split}.pt"

    def __init__(
        self,
        datapath: str = "database",
        dataset_name: str = "dataset_train_tiny",
        split: str = "train",
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.dataset_name = dataset_name
        self.datapath = datapath
        self.split = split
        self.data_all, self.slices_all = [], []
        self.offsets = [0]
        super(PyGNablaDFT, self).__init__(datapath, transform, pre_transform)

        self.data = torch.load(self.processed_paths[0])

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

    def process(self) -> None:
        db = connect(self.raw_paths[0])
        samples = []
        for db_row in tqdm(db.select(), total=len(db)):
            z = torch.from_numpy(db_row.numbers.copy()).long()
            # print(f'z: {len(z)}')
            positions = torch.from_numpy(db_row.positions.copy()).float()
            y = torch.from_numpy(np.array(db_row.data["energy"])).float()
            forces = torch.from_numpy(np.array(db_row.data["forces"])).float()
            molecule_size = len(positions)
            # print(f'molecule_size: {molecule_size}')
            samples.append(Data(z=z, pos=positions, y=y, dy=forces, molecule_size=molecule_size))

        # data, slices = self.collate(samples)
        # torch.save((data, slices), self.processed_paths[0])
        torch.save(samples, self.processed_paths[0])
        logger.info(f"Saved processed dataset: {self.processed_paths[0]}")


def get_mol(smiles):

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  

    formal_charge = Chem.GetFormalCharge(mol)

    return torch.tensor(formal_charge, dtype=torch.float)


def get_mean_std(data_loader):
    ys = torch.cat([batch.y for batch in data_loader])
    return ys.mean(), ys.std()
