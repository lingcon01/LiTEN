from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_scatter import scatter
from typing import List, Optional, Tuple
from torch_geometric.nn.inits import glorot
from .utils import *
import logging


class MBI(nn.Module):
    def __init__(self, num_heads, hidden_channels, activation, cutoff, vecnorm):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.eps = 1e-12

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_norm = vecnorm

        self.act = activation
        self.alpha = torch.nn.Parameter(torch.Tensor(1, self.num_heads, self.head_dim))

        self.cutoff = CosineCutoff(cutoff)

        self.vec_linear = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.cross_linear = nn.Linear(hidden_channels, hidden_channels, bias=False)
        # self.vs_linear = nn.Linear(hidden_channels * 2, hidden_channels, bias=False)

        self.node_linear = nn.Linear(hidden_channels, hidden_channels)
        self.edge_linear = nn.Linear(hidden_channels, hidden_channels)
        self.part_linear1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.part_linear2 = nn.Linear(hidden_channels, hidden_channels * 3)

        self.f_linear = nn.Linear(hidden_channels, hidden_channels)

        self.vecnorm = VecNorm(hidden_channels, norm_type="norm")

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        glorot(self.alpha)
        nn.init.xavier_uniform_(self.node_linear.weight)
        self.node_linear.bias.data.zero_()
        nn.init.xavier_uniform_(self.edge_linear.weight)
        self.edge_linear.bias.data.zero_()
        nn.init.xavier_uniform_(self.part_linear1.weight)
        self.part_linear1.bias.data.zero_()
        nn.init.xavier_uniform_(self.part_linear2.weight)
        self.part_linear2.bias.data.zero_()
        nn.init.xavier_uniform_(self.cross_linear.weight)
        nn.init.xavier_uniform_(self.vec_linear.weight)
        nn.init.xavier_uniform_(self.f_linear.weight)
        self.f_linear.bias.data.zero_()

    def edge_update(self, node_vector, edge_index, edge_vector, edge_feats):
        i, j = edge_index[0], edge_index[1]

        node_vector = self.cross_linear(node_vector)

        # vec_cross = torch.cross(node_vector[i], node_vector[j], dim=1)
        vec_cross_i = torch.cross(node_vector[i], edge_vector.unsqueeze(-1), dim=1)
        vec_cross_j = torch.cross(node_vector[j], edge_vector.unsqueeze(-1), dim=1)
        sum_phi = torch.sum(vec_cross_i * vec_cross_j, dim=1)

        delta_edge_feats = self.act(self.f_linear(edge_feats)) * sum_phi

        return delta_edge_feats

    def message(self, node_scalar, node_vector, edge_index, dist, edge_vector, edge_feats):
        i, j = edge_index[0], edge_index[1]
        edge_feats = self.act(self.edge_linear(edge_feats)).reshape(-1, self.num_heads, self.head_dim)
        node_scalar = self.node_linear(node_scalar).reshape(-1, self.num_heads, self.head_dim)
        n_nodes = len(node_scalar)
        attn = (node_scalar[i] + node_scalar[j] + edge_feats)
        attn = self.act(attn) * self.alpha
        attn = attn.sum(dim=-1) * self.cutoff(dist).unsqueeze(1)

        node_scalar = node_scalar[j] * edge_feats
        node_scalar = (node_scalar * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        node_sca1, node_sca2 = torch.split(self.act(self.part_linear1(node_scalar)), self.hidden_channels, dim=1)
        node_vector = node_vector[j] * node_sca1.unsqueeze(1) + node_sca2.unsqueeze(
            1) * edge_vector.unsqueeze(2)

        node_scalar = scatter(node_scalar, i, dim=0, reduce="sum", dim_size=n_nodes)
        node_vector = scatter(node_vector, i, dim=0, reduce="sum", dim_size=n_nodes)

        return node_scalar, node_vector, attn

    def node_update(self, node_scalar, node_vector):

        node_vec1, node_vec2 = torch.split(self.vec_linear(node_vector), self.hidden_channels, dim=-1)
        # vec_tri = torch.einsum('BiK,BiK->BK', node_vec1, node_vec2)
        vec_tri = torch.sum(node_vec1 * node_vec2, dim=1)

        norm_vec = torch.sqrt(torch.sum(node_vec2 ** 2, dim=-2) + 1e-8)
        vec_qua = norm_vec ** 3

        node_scalar = self.part_linear2(node_scalar)

        node_sca1, node_sca2, node_sca3 = torch.split(node_scalar, self.hidden_channels, dim=1)

        delta_scalar = (vec_qua + vec_tri) * node_sca1 + node_sca2
        # delta_scalar = vec_tri * node_sca1 + node_sca2
        delta_vector = node_vec1 * node_sca3.unsqueeze(1)

        return delta_scalar, delta_vector

    def forward(self, node_scalar, node_vector, edge_index, dist, edge_feats, edge_vector,
                update_edge: bool = True):

        scalar_out = self.layernorm(node_scalar)

        if self.vec_norm:
            node_vector = vecnorm(node_vector, self.eps)

        scalar_out, vector_out, attn_out = self.message \
            (scalar_out, node_vector, edge_index, dist, edge_vector, edge_feats)

        if update_edge:
            delta_edge_feats = self.edge_update(node_vector, edge_index, edge_vector, edge_feats)

            edge_feats = edge_feats + delta_edge_feats

        node_scalar = node_scalar + scalar_out
        node_vector = node_vector + vector_out

        delta_scalar, delta_vector = self.node_update(node_scalar, node_vector)

        node_scalar = node_scalar + delta_scalar
        node_vector = node_vector + delta_vector

        return node_scalar, node_vector, edge_feats


@torch.jit.script
def vecnorm(vec: Tensor, eps: float) -> Tensor:
    dist = torch.norm(vec, dim=1, keepdim=True)

    dist = dist.clamp(min=eps)
    radial = vec / dist

    min_val = torch.min(dist, dim=-1, keepdim=True)[0]
    max_val = torch.max(dist, dim=-1, keepdim=True)[0]
    dist_norm = (dist - min_val) / (max_val - min_val).clamp(min=eps)

    norm_vec = dist_norm * radial

    return norm_vec
