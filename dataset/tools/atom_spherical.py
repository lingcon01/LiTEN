import numpy as np
import scipy.special as sp
import logging
from rdkit import Chem

# base feature: scaler
# interact feature: spd

# 孤对电子和edge的相互作用, edge和edge的相互作用

# 考虑是否使用双向图

# 每个原子和周围相连的原子构建一个多面体，mji = self.conv_tp(
#             node_feats[sender], edge_attrs, tp_weights
#         )  # [n_edges, irreps]

# 上述全用共价键进行获取edge_index

# 格式使用 [主量子数，价电子数] eg. O原子：[2, 2, 2, 2, 0, 0, 0, 0, 0, 0] 1x0e + 1x0e + 1x1o + 1x2e ➡️ 128x0e + 128x1o + 128x2e

# 价电子用于和edge_sh进行交互, 在获得mij之后，将以该原子为中心的所有mij进行融合交互，用投影的方式两两乘积，输出node_features（获取每个原子的边索引即可）

# 原子序数: [8] 作为base_features, 用于自交互，将该特征和价电子获得的interaction特征node_features进行交互，输出base_features

# 叠加态的基函数

# 用这个替代径向基函数

# 计算径向波函数 R_nl(r)
# def radial_wave_function(n, l, r, alpha):
#     """计算径向波函数 R_nl(r)"""
#     # N_{n,l} 规范化常数
#     norm_factor = np.sqrt(
#         (2 * alpha) ** (2 * l + 3) * (np.math.factorial(n - l - 1)) / (2 * n * np.math.factorial(n + l)))
#     # 径向波函数
#     return norm_factor * r ** l * np.exp(-alpha * r)

# 长程相互作用，主要使用H键的互作，获取edge_index



# 更新后的 α 值表
ALPHA_VALUES = {
    1: [1.000, None, None, None, None],  # H
    2: [2.000, None, None, None, None],  # He
    3: [1.500, 1.200, None, None, None],  # Li
    4: [1.200, 1.000, None, None, None],  # Be
    5: [1.000, 0.850, 0.900, None, None],  # B
    6: [0.850, 0.650, 0.700, None, None],  # C
    7: [0.650, 0.450, 0.500, None, None],  # N
    8: [0.500, 0.350, 0.400, None, None],  # O
    9: [0.400, 0.250, 0.300, None, None],  # F
    10: [0.300, 0.200, 0.250, None, None],  # Ne
    11: [0.800, 0.600, 0.650, 0.600, 0.700],  # Na
    12: [0.650, 0.450, 0.500, 0.450, 0.500],  # Mg
    13: [0.650, 0.450, 0.500, 0.450, 0.500],  # Al
    14: [0.500, 0.350, 0.400, 0.350, 0.400],  # Si
    15: [0.400, 0.250, 0.300, 0.250, 0.300],  # P
    16: [0.350, 0.200, 0.250, 0.200, 0.250],  # S
    17: [0.250, 0.150, 0.200, 0.150, 0.200],  # Cl
    35: [0.250, 0.150, 0.200, 0.150, 0.200],  # Br
    53: [0.250, 0.150, 0.200, 0.150, 0.200],  # I
}

# 元素的电子配置 (电子数与轨道类型)
ELECTRON_CONFIG = {
    1: {"orbitals": ["1s"], "electrons": [1]},  # H
    5: {"orbitals": ["1s", "2s", "2p"], "electrons": [2, 2, 1]},  # B
    6: {"orbitals": ["1s", "2s", "2p"], "electrons": [2, 2, 2]},  # C
    7: {"orbitals": ["1s", "2s", "2p"], "electrons": [2, 2, 3]},  # N
    8: {"orbitals": ["1s", "2s", "2p"], "electrons": [2, 2, 4]},  # O
    9: {"orbitals": ["1s", "2s", "2p"], "electrons": [2, 2, 5]},  # F
    14: {"orbitals": ["1s", "2s", "2p", "3s", "3p"], "electrons": [2, 2, 6, 2, 2]},  # Si
    15: {"orbitals": ["1s", "2s", "2p", "3s", "3p"], "electrons": [2, 2, 6, 2, 3]},  # P
    16: {"orbitals": ["1s", "2s", "2p", "3s", "3p"], "electrons": [2, 2, 6, 2, 4]},  # S
    17: {"orbitals": ["1s", "2s", "2p", "3s", "3p"], "electrons": [2, 2, 6, 2, 5]},  # Cl
    35: {"orbitals": ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p"], "electrons": [2, 2, 6, 2, 6, 10, 2, 5]},  # Br
    53: {"orbitals": ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d", "5s", "5p"], "electrons": [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 5]}  # I
}

# 最大轨道数目 (s=1, p=3, d=5, f=7)
ORBITAL_DIMENSIONS = {
    "s": 1,
    "p": 3,
    "d": 5,
    "f": 7
}

# 计算球谐函数 Y_lm(theta, phi)
def spherical_harmonics(l, m, theta, phi):
    """计算球谐函数 Y_lm(theta, phi)"""
    return sp.sph_harm(m, l, phi, theta).real


# 计算径向波函数 R_nl(r)
def radial_wave_function(n, l, r, alpha):
    """计算径向波函数 R_nl(r)"""
    # N_{n,l} 规范化常数
    norm_factor = np.sqrt(
        (2 * alpha) ** (2 * l + 3) * (np.math.factorial(n - l - 1)) / (2 * n * np.math.factorial(n + l)))
    # 径向波函数
    return norm_factor * r ** l * np.exp(-alpha * r)


# 获取单个原子的轨道表示
def get_atom_orbital_representation(atom, r, theta, phi):
    """获取单个原子的轨道填充信息和球谐展开"""
    atomic_number = atom.GetAtomicNum()
    if atomic_number not in ALPHA_VALUES:
        logging.error(f"{atomic_number} is not in ALPHA_VALUES")
        return None

    # 获取电子排布信息
    config = ELECTRON_CONFIG[atomic_number]
    orbitals = config["orbitals"]
    electrons = config["electrons"]
    alphas = ALPHA_VALUES[atomic_number]

    # 创建统一维度的特征向量 (最大维度是 16)
    feature_vector = np.zeros(16)

    # 为每个轨道类型生成径向和角向部分
    idx = 0  # 用于填充特征向量
    for i, orbital in enumerate(orbitals):
        n_e = electrons[i]  # 电子数

        # 解析轨道类型并分配相应的维度
        if "s" in orbital:
            l = 0
            orbital_dim = ORBITAL_DIMENSIONS["s"]
            alpha = alphas[0] if alphas[0] is not None else 0.5  # 默认使用第一个α值
        elif "p" in orbital:
            l = 1
            orbital_dim = ORBITAL_DIMENSIONS["p"]
            alpha = alphas[1] if alphas[1] is not None else 0.4  # 默认使用第二个α值
        elif "d" in orbital:
            l = 2
            orbital_dim = ORBITAL_DIMENSIONS["d"]
            alpha = alphas[2] if alphas[2] is not None else 0.3  # 默认使用第三个α值
        elif "f" in orbital:
            l = 3
            orbital_dim = ORBITAL_DIMENSIONS["f"]
            alpha = alphas[3] if alphas[3] is not None else 0.2  # 默认使用第四个α值
        else:
            continue  # 如果轨道类型不支持，跳过

        # 计算径向部分和球谐展开并加权
        for m in range(-l, l + 1):
            if idx < 16:
                R_nl = radial_wave_function(i + 1, l, r, alpha)  # 计算径向波函数
                feature_vector[idx] = R_nl * spherical_harmonics(l, m, theta, phi) * n_e
                idx += 1
            else:
                break

    return feature_vector


# 对整个分子进行轨道特征表示
def molecular_orbital_representation(smiles):
    """对整个分子进行轨道特征建模，生成原子特征向量"""
    mol = Chem.MolFromSmiles(smiles)
    atom_orbitals = []

    r = 1.0  # 假设的半径值
    theta, phi = np.pi / 4, np.pi / 3  # 假设的角度

    for atom in mol.GetAtoms():
        atom_orbitals.append(get_atom_orbital_representation(atom, r, theta, phi))

    return atom_orbitals


# 测试示例
smiles = "O=C=O"  # CO₂
orbital_repr = molecular_orbital_representation(smiles)
print(orbital_repr)
