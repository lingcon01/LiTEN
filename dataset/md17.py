import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from tqdm import tqdm
import pickle as pkl
import os
from rdkit import Chem
from rdkit.Chem import SDMolSupplier


from sklearn.model_selection import train_test_split


class rMD17(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = dict(
        aspirin="rmd17_aspirin.npz",
        benzene="rmd17_benzene.npz",
        ethanol="rmd17_ethanol.npz",
        malonaldehyde="rmd17_malonaldehyde.npz",
        naphthalene="rmd17_naphthalene.npz",
        salicylic="rmd17_salicylic.npz",
        toluene="rmd17_toluene.npz",
        uracil="rmd17_uracil.npz",
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(rMD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(rMD17.available_molecules)
        self.molecules = dataset_arg.split(",")

        if len(self.molecules) > 1:
            print(
                "rMD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(rMD17, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )

    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        data = super(rMD17, self).get(idx - self.offsets[data_idx])
        if self.transform:
            return self.transform(data)
        else:
            return data

    @property
    def raw_file_names(self):
        return [rMD17.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f"rmd17-{mol}.pt" for mol in self.molecules]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(rMD17.raw_url + file_name, self.raw_dir)

    def process(self):
        for path in self.raw_paths:
            one_hot = []
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["nuclear_charges"]).long()
            for z_i in z:
                one_hot.append(np.array(one_of_k_encoding_unk(z_i, [6, 7, 8, 16, 9, 14, 15, 17, 35, 53, 5, 1, 0])))
            one_hot = torch.tensor(one_hot, dtype=torch.long)
            positions = torch.from_numpy(data_npz["coords"]).float()
            energies = torch.from_numpy(data_npz["energies"]).unsqueeze(-1).float()
            forces = torch.from_numpy(data_npz["forces"]).float()

            samples = []
            for pos, y, dy in zip(positions, energies, forces):
                molecule_size = len(pos)
                samples.append(
                    Data(z=z, node_attr=one_hot, pos=pos, y=y.unsqueeze(1), dy=dy, molecule_size=molecule_size))

            data, slices = self.collate(samples)
            torch.save((data, slices), self.processed_paths[0])


class MD17(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = dict(
        aspirin="md17_aspirin.npz",
        benzene="md17_benzene2017.npz",
        ethanol="md17_ethanol.npz",
        malonaldehyde="md17_malonaldehyde.npz",
        naphthalene="md17_naphthalene.npz",
        salicylic="md17_salicylic.npz",
        toluene="md17_toluene.npz",
        uracil="md17_uracil.npz",
        md22_AT_AT_CG_CG="md22_AT-AT-CG-CG.npz",
        md22_Ac_Ala3_NHMe='md22_Ac-Ala3-NHMe.npz',
        md22_AT_AT='md22_AT-AT.npz',
        md22_buckyball_catcher='md22_buckyball-catcher.npz',
        md22_DHA='md22_DHA.npz',
        md22_dw_nanotube='md22_dw_nanotube.npz',
        md22_stachyose='md22_stachyose.npz',
        raspirin="rmd17_aspirin.npz",
        rbenzene="rmd17_benzene.npz",
        rethanol="rmd17_ethanol.npz",
        rmalonaldehyde="rmd17_malonaldehyde.npz",
        rnaphthalene="rmd17_naphthalene.npz",
        rsalicylic="rmd17_salicylic.npz",
        rtoluene="rmd17_toluene.npz",
        ruracil="rmd17_uracil.npz",
        razobenzene="rmd17_azobenzene.npz",
        rparacetamol="rmd17_paracetamol.npz"
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(MD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(MD17.available_molecules)
        self.molecules = dataset_arg.split(",")

        self.base_energy = {1: -13.663181292231226, 6: -1029.2809654211628, 7: -1484.1187695035828,
                            8: -2042.0330099956639}

        if len(self.molecules) > 1:
            print(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(MD17, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []

        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )

    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        data = super(MD17, self).get(idx - self.offsets[data_idx])
        if self.transform:
            return self.transform(data)
        else:
            return data

    @property
    def raw_file_names(self):
        return [MD17.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f"md17-{mol}.pt" for mol in self.molecules]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(MD17.raw_url + file_name, self.raw_dir)

    def process(self):
        for path in self.raw_paths:

            one_hot = []
            atom_energy = []
            print(path)
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["z"]).long()
            # z = torch.from_numpy(data_npz["nuclear_charges"]).long()
            # for z_i in z:
            #     atom_energy.append(self.base_energy[int(z_i)])
            #     one_hot.append(np.array(one_of_k_encoding_unk(z_i, [6, 7, 8, 16, 9, 14, 15, 17, 35, 53, 5, 1, 0])))

            # base_energy = sum(atom_energy) * 1000 # meV
            # one_hot = torch.tensor(one_hot).float()
            positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()  # kcal.mol-1
            forces = torch.from_numpy(data_npz["F"]).float()
            # positions = torch.from_numpy(data_npz["coords"]).float()
            # energies = torch.from_numpy(data_npz["energies"]).unsqueeze(-1).float()
            # forces = torch.from_numpy(data_npz["forces"]).float()
            # edge_index, bond_types = smiles_to_edge_index(path='/home/suqun/model/LumiForce/data/aspirin.sdf')
            # edge_num = edge_index.size()[1]

            if energies.dim() == 1:
                energies = energies.unsqueeze(-1)  # 保证形状为 [N, 1]

            samples = []
            for pos, y, dy in zip(positions, energies, forces):
                molecule_size = len(pos)
                samples.append(
                    Data(z=z, pos=pos, y=y, dy=dy, molecule_size=molecule_size))

            data, slices = self.collate(samples)
            torch.save((data, slices), self.processed_paths[0])


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smiles_to_edge_index(path):
    """
    从 SMILES 字符串中提取共价键连接，并生成 edge_index。

    参数:
        smiles (str): SMILES 字符串。

    返回:
        edge_index (torch.Tensor): 形状为 (2, num_edges) 的边索引张量。
    """
    # 将 SMILES 字符串转换为 RDKit 分子对象
    supplier = SDMolSupplier(path, removeHs=False)
    mol = supplier[0]
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # 获取原子数量
    num_atoms = mol.GetNumAtoms()
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # 获取键信息
    edges_src = []
    edges_end = []
    bond_types = []
    for bond in mol.GetBonds():
        # 获取键的起始和结束原子索引
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bond_types.append(bond_type)
        # 添加双向边（因为图是无向的）
        edges_src.extend([start_atom, end_atom])
        edges_end.extend([end_atom, start_atom])
        # edges_src.append(start_atom)
        # edges_end.append(end_atom)

    num_atoms = mol.GetNumAtoms()  # 获取分子中原子的总数
    for atom_idx in range(num_atoms):
        edges_src.append(atom_idx)
        edges_end.append(atom_idx)
        bond_types.append(Chem.BondType.SINGLE)  # 自环的键类型可以设为单键，或者根据需求自定义

    # 转换为 PyTorch 的 edge_index
    edge_index = torch.tensor([edges_src, edges_end], dtype=torch.long)()
    bond_types = torch.tensor(bond_types, dtype=torch.float)

    return edge_index, bond_types


def stratified_split_by_energy(dataset, num_train, num_val, num_test, num_bins=10, seed=2025):
    # 获取所有样本的 energy 值
    energies = np.array([data.y for data in dataset])

    # 按 energy 值的分位数分为 10 等分（等频分桶）
    bins = np.quantile(energies, np.linspace(0, 1, num_bins + 1))
    # 为了处理重复边界，稍微微调一下 bins
    bins = np.unique(bins)  # 去除重复边界，否则 np.digitize 会出问题
    energy_labels = np.digitize(energies, bins, right=False)

    # 初始化
    train_idx, val_idx, test_idx = [], [], []
    random_state = np.random.RandomState(seed=seed)

    for idx, label in enumerate(np.unique(energy_labels)):
        idxs = np.where(energy_labels == label)[0]
        random_state.shuffle(idxs)
        n_total = len(idxs)

        if idx == (len(np.unique(energy_labels)) - 2):
            n_train = num_train - len(train_idx)
            n_val = num_val - len(val_idx)
        else:
            n_train = int(num_train * n_total / len(dataset))
            n_val = int(num_val * n_total / len(dataset))

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val: -1])

    # 再次打乱
    random_state.shuffle(train_idx)
    random_state.shuffle(val_idx)
    random_state.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def get_dataloaders(dataset, num_train, num_val, num_test, batch_size, test_batch_size, num_workers, idx_dir):
    size = len(dataset)
    print(size)
    idx = np.arange(size)
    random_state = np.random.RandomState(seed=2025)
    random_state.shuffle(idx)

    train_idx = idx[:num_train]
    val_idx = idx[num_train:num_train + num_val]
    test_idx = idx[num_train + num_val: -1]
    # split_path = os.path.join(idx_dir, 'splits.npz')
    #
    # if os.path.exists(split_path):
    #     print("Loading index split from:", split_path)
    #     splits = np.load(split_path)
    #     train_idx = splits['train_idx']
    #     val_idx = splits['val_idx']
    #     test_idx = splits['test_idx']
    # else:
    #     train_idx, val_idx, test_idx = stratified_split_by_energy(dataset, num_train, num_val, num_test, num_bins=10, seed=2025)
    #     np.savez(split_path, train_idx=np.array(train_idx), val_idx=np.array(val_idx), test_idx=np.array(test_idx))
    train_set = dataset[train_idx]
    val_set = dataset[val_idx]
    test_set = dataset[test_idx]
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }


def get_mean_std(data_loader):
    ys = torch.cat([batch.y for batch in data_loader['train']])
    return ys.mean(), ys.std()
# def get_mean_std(dataset):
#     ys = torch.cat([data.y for data in dataset])
#     mean, std = ys.mean(), ys.std()
#     atom_inter_energy = torch.cat([(data.y - mean)/len(data.z) for data in dataset])
#     force_list = torch.cat([data.dy for data in dataset])
#     atom_mean = atom_inter_energy.mean()
#     atom_std = torch.sqrt(torch.mean(torch.square(force_list))).item()
#
#     return mean, std, atom_mean, atom_std
