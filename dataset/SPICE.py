import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from ase.units import Bohr
from torch_geometric.data import Data, InMemoryDataset


class HDF5ConformationDataset(Dataset):
    def __init__(self, hdf5_path, index=None):
        self.hdf5_path = hdf5_path
        self.file = None
        # self.file = h5py.File(self.hdf5_path, 'r')

        # 使用外部提供的 index，或构建全量 index
        if index is not None:
            self.index = index
        else:
            self.index = []
            for mol_id in self.file.keys():
                n_confs = self.file[mol_id]["conformations"].shape[0]
                for i in range(n_confs):
                    self.index.append((mol_id, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
        mol_id, conf_idx = self.index[idx]
        conf_idx = int(conf_idx)
        mol = self.file[mol_id]

        z = torch.tensor(mol["atomic_numbers"][()], dtype=torch.long)
        pos = torch.tensor(np.array(mol["conformations"])[conf_idx], dtype=torch.float32) * Bohr
        y = torch.tensor(np.array(mol["formation_energy"])[conf_idx], dtype=torch.float32)
        neg_dy = -torch.tensor(np.array(mol["dft_total_gradient"])[conf_idx], dtype=torch.float32) / Bohr

        data = Data(z=z, pos=pos, y=y, dy=neg_dy)

        return data

    def __del__(self):
        if hasattr(self, "file"):
            self.file.close()


class TorchDataset(Dataset):
    def __init__(self, data_path, index=None):
        self.data_path = data_path
        total = torch.load(self.data_path)
        self.data = [total[i] for i in index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TorchDataset2(Dataset):
    def __init__(self, data_path, index=None):
        self.data_path = data_path
        self.data_name = os.listdir(self.data_path)
        # self.data_name = [total[i] for i in index]

        # self.data = torch.load(self.data_path)

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, idx):
        abs_path = os.path.join(self.data_path, self.data_name[idx])
        data = torch.load(abs_path, weights_only=False)
        data.y = torch.tensor([data.y], dtype=torch.float) if data.y.dim() == 0 else data.y
        data.z = data.z.to(torch.long)
        data.edge_index = data.edge_index.to(torch.long)

        data.pos = data.pos.to(torch.float) 
        data.dy = data.dy.to(torch.float)  
        data.shifts = data.shifts.to(torch.float) 

        return data



