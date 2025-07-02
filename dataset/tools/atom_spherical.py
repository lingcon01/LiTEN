import numpy as np
import scipy.special as sp
import logging
from rdkit import Chem

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

ORBITAL_DIMENSIONS = {
    "s": 1,
    "p": 3,
    "d": 5,
    "f": 7
}

def spherical_harmonics(l, m, theta, phi):
    return sp.sph_harm(m, l, phi, theta).real

def radial_wave_function(n, l, r, alpha):
    norm_factor = np.sqrt(
        (2 * alpha) ** (2 * l + 3) * (np.math.factorial(n - l - 1)) / (2 * n * np.math.factorial(n + l)))
    return norm_factor * r ** l * np.exp(-alpha * r)

def get_atom_orbital_representation(atom, r, theta, phi):
    atomic_number = atom.GetAtomicNum()
    if atomic_number not in ALPHA_VALUES:
        logging.error(f"{atomic_number} is not in ALPHA_VALUES")
        return None

    config = ELECTRON_CONFIG[atomic_number]
    orbitals = config["orbitals"]
    electrons = config["electrons"]
    alphas = ALPHA_VALUES[atomic_number]

    feature_vector = np.zeros(16)

    idx = 0  
    for i, orbital in enumerate(orbitals):
        n_e = electrons[i] 
        if "s" in orbital:
            l = 0
            orbital_dim = ORBITAL_DIMENSIONS["s"]
            alpha = alphas[0] if alphas[0] is not None else 0.5 
        elif "p" in orbital:
            l = 1
            orbital_dim = ORBITAL_DIMENSIONS["p"]
            alpha = alphas[1] if alphas[1] is not None else 0.4  
        elif "d" in orbital:
            l = 2
            orbital_dim = ORBITAL_DIMENSIONS["d"]
            alpha = alphas[2] if alphas[2] is not None else 0.3  
        elif "f" in orbital:
            l = 3
            orbital_dim = ORBITAL_DIMENSIONS["f"]
            alpha = alphas[3] if alphas[3] is not None else 0.2  
        else:
            continue 

        for m in range(-l, l + 1):
            if idx < 16:
                R_nl = radial_wave_function(i + 1, l, r, alpha)  
                feature_vector[idx] = R_nl * spherical_harmonics(l, m, theta, phi) * n_e
                idx += 1
            else:
                break

    return feature_vector


def molecular_orbital_representation(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_orbitals = []

    r = 1.0  
    theta, phi = np.pi / 4, np.pi / 3 

    for atom in mol.GetAtoms():
        atom_orbitals.append(get_atom_orbital_representation(atom, r, theta, phi))

    return atom_orbitals

smiles = "O=C=O"  # CO₂
orbital_repr = molecular_orbital_representation(smiles)
print(orbital_repr)
