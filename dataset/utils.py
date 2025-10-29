import torch
import torch.nn as nn
from typing import Dict, Union

try:
    torch_compile = torch.compile  # PyTorch 2.x
except AttributeError:
    torch_compile = torch.jit.script  # fallback for older versions

@torch_compile
class AtomicEnergiesBlock(nn.Module):
    def __init__(self, z_charge_energy: Dict[int, Dict[int, float]] = None):
        super().__init__()

        if z_charge_energy is None:
            z_charge_energy = {
                5: {
                    -1: -0.9069259227409529,
                    0: -0.9067014997308661,
                    1: -0.8955912585096613
                },
                35: {
                    -1: -94.6090157909654,
                    0: -94.604404905443
                },
                6: {
                    -1: -1.3932418998827225,
                    0: -1.3916987987479127,
                    1: -1.3759382381507295
                },
                20: {
                    2: -24.88112570877957
                },
                17: {
                    -1: -16.912265570591728,
                    0: -16.907185460268183
                },
                9: {
                    -1: -3.672819229197178,
                    0: -3.6682253180170643
                },
                1: {
                    -1: -0.018479295573135273,
                    0: -0.01832997834502815,
                    1: 0.0
                },
                53: {
                    -1: -10.950741509446582,
                    0: -10.94636015721096
                },
                19: {
                    1: -22.03781924636417
                },
                3: {
                    1: -0.26775029926726546
                },
                12: {
                    2: -7.322184135640635
                },
                7: {
                    -1: -2.0063839463975473,
                    0: -2.0071399826210213,
                    1: -1.9867330882457677
                },
                11: {
                    1: -5.958752782203725
                },
                8: {
                    -1: -2.763314563434649,
                    0: -2.7611998823447985,
                    1: -2.741634050870444
                },
                15: {
                    0: -12.540610523725608,
                    1: -12.526828346635187
                },
                16: {
                    -1: -14.638878157050365,
                    0: -14.635829845173457,
                    1: -14.622592033424905
                },
                14: {
                    -1: -10.636087279353365,
                    0: -10.63456948425163,
                    1: -10.62379944847741
                }
            }

        self.z_max = max(z_charge_energy.keys())
        self.charge_min = min(min(qs) for qs in z_charge_energy.values())
        self.charge_max = max(max(qs) for qs in z_charge_energy.values())
        self.charge_offset = -self.charge_min

        # 创建查找表
        energy_table = torch.full(
            (self.z_max + 1, self.charge_max - self.charge_min + 1),
            float('nan'),
            dtype=torch.get_default_dtype()
        )
        for z, charges in z_charge_energy.items():
            for q, e in charges.items():
                energy_table[z, q + self.charge_offset] = e
        self.register_buffer("energy_table", energy_table)

    def forward(self, z: torch.Tensor, charge: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Args:
            z: [N] atomic numbers
            charge: [N] atomic charges (optional, default to 0)
        Returns:
            energies: [N] atomic ground state energies
        """
        if charge is None:
            charge = torch.zeros_like(z)

        charge_idx = charge + self.charge_offset
        energies = self.energy_table[z, charge_idx]

        if torch.isnan(energies).any():
            raise ValueError("Some Z/charge combinations are not in the energy table.")

        return energies

