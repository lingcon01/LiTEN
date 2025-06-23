import math
import os
from argparse import Namespace
from collections import defaultdict

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import logging
from torch_geometric.nn import global_mean_pool
from model.MBI_blocks.TQA import *
from model.MBI_blocks.utils import *


class LiTEN(nn.Module):

    def __init__(self, num_heads=8, num_layers=6, hidden_channels=256, num_rbf=32,
                 cutoff=5.0, max_neighbors=32, max_z=100, vec_norm=True, atomic_inter_scale=None,
                 atomic_inter_shift=None):
        super(LiTEN, self).__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.activation = nn.SiLU()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.drop_rate = [0.0, 0.0, 0.1, 0.05, 0.05]
        self.atomic_inter_scale = atomic_inter_scale
        self.atomic_inter_shift = atomic_inter_shift

        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.connect = Edge_Connect(cutoff, max_neighbors=max_neighbors, loop=True)
        self.radial_fn = ExpNormalSmearing(cutoff, num_rbf)

        self.Interaction = nn.ModuleList()

        for idx in range(num_layers):
            if vec_norm:
                vecnorm = idx > 0
            else:
                vecnorm = False
            layer = MBI(num_heads=num_heads, hidden_channels=hidden_channels, activation=self.activation,
                        cutoff=cutoff, vecnorm=vecnorm)

            self.Interaction.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.ShiftedSoft = ShiftedSoftplus()

        self.edge_embedding = nn.Linear(num_rbf, hidden_channels)

        self.readout_energy = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            self.activation,
            nn.Linear(hidden_channels // 2, 1),
        )

        self.readout_force = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            self.activation,
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.radial_fn.reset_parameters()
        for layer in self.Interaction:
            layer.reset_parameters()

        self.out_norm.reset_parameters()

    def forward(self, atoms_batch, mean, std, if_train):

        z, pos, batch = atoms_batch.z, atoms_batch.pos, atoms_batch.batch

        pos.requires_grad_(True)

        # Embedding Layers
        node_scalar = self.embedding(z)

        edge_index, dist, edge_vector = self.connect(pos, batch)
        edge_feats = self.radial_fn(dist)
        node_vector = torch.zeros(node_scalar.size(0), 3, node_scalar.size(1), device=node_scalar.device)
        edge_feats = self.edge_embedding(edge_feats)

        for idx, interaction in enumerate(self.Interaction):
            update_edge = idx < len(self.Interaction) - 1
            node_scalar, node_vector, edge_feats = interaction(
                dist=dist,
                node_scalar=node_scalar,
                node_vector=node_vector,
                edge_index=edge_index,
                edge_feats=edge_feats,
                edge_vector=edge_vector,
                update_edge=update_edge,
            )

        node_scalar = self.out_norm(node_scalar)

        node_energy = self.readout_energy(node_scalar)

        node_energy = node_energy * std

        energy = scatter(src=node_energy, index=batch, dim=0, reduce="sum")

        energy = energy + mean
        energy = energy.squeeze(-1)

        # calculate_forces:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]

        forces = -torch.autograd.grad(
            outputs=[energy],  # [n_graphs, ]
            inputs=[pos],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=if_train,  # Make sure the graph is not destroyed during training
            allow_unused=True,  # For complete dissociation turn to true
            create_graph=if_train,  # Create graph for second derivative
        )[0]  # [n_nodes, 3]

        return energy, forces
