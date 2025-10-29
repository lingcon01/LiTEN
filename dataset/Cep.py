import h5py
import numpy as np
from torch_geometric.data import Batch, Data, Dataset  # , InMemoryDataset
import torch
from ase.data import atomic_numbers
from torch_geometric.data import DataLoader
from tqdm import tqdm
import os


class CepDataset(Dataset):
    def __init__(self, npz_dir):
        super(CepDataset, self).__init__()

        self.npz_dir = npz_dir
        self.npz_file = os.listdir(self.npz_dir)

    def get(self, idx):
        np_file = os.path.join(self.npz_dir, self.npz_file[idx])
        np_data = np.load(np_file, allow_pickle=True)
        z = torch.from_numpy(np.array(np_data['symbols']).copy()).long()
        positions = torch.from_numpy(np_data['coordinates'].copy()).float()
        y = torch.from_numpy(np.array(np_data['energies'].copy())).float().squeeze(-1)
        forces = torch.from_numpy(np_data['forces'].copy()).float()
        molecule_size = len(positions)
        data = Data(z=z, pos=positions, y=y.unsqueeze(-1), dy=forces, molecule_size=molecule_size)

        return data

    def len(self):
        return len(self.npz_file)


def get_split(dataset, train_ra, val_ra):
    size = len(dataset)
    print(size)
    idx = np.arange(size)
    random_state = np.random.RandomState(seed=2025)
    random_state.shuffle(idx)

    num_train = int(size * train_ra)
    num_val = int(size * val_ra)

    train_idx = idx[:num_train]
    val_idx = idx[num_train:num_train + num_val]
    train_set = dataset[train_idx]
    val_set = dataset[val_idx]

    return train_set, val_set


def get_mean_std(data_loader):
    ys = torch.cat([batch.y for batch in tqdm(data_loader)])
    return ys.mean(), ys.std()
