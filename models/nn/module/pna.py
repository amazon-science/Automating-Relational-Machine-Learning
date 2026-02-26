"""
    Adapted from the dgl-based implementation from https://github.com/awslabs/multi-table-benchmark/blob/main/dbinfer/solutions/gnn/pna.py
"""
import logging
from typing import Tuple, Dict, Optional, List, Any, Union
from enum import Enum

import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

# PyG imports
from torch_geometric.nn.aggr import (
    MeanAggregation, MaxAggregation, MinAggregation, StdAggregation, SumAggregation, Aggregation
)
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.utils import degree
from torch_geometric.typing import Adj, OptTensor, PairTensor



class AggregatorEnum(str, Enum):
    mean = "mean"
    max = "max"
    min = "min"
    std = "std"
    var = "var" # Note: PyG's StdAggregation can compute var implicitly.
    sum = "sum"

class ScalarEnum(str, Enum):
    identity = "identity"
    amplification = "amplification"
    attenuation = "attenuation"


# Mapping from string to PyG Aggregation class
AGGREGATOR_MAP = {
    'mean': MeanAggregation(),
    'max': MaxAggregation(),
    'min': MinAggregation(),
    'sum': SumAggregation(),
    'std': StdAggregation(),
    # 'var' can be derived from 'std', so we use StdAggregation and square the result.
    'var': StdAggregation(),
}


class PyG_PNAConvTower(nn.Module):
    """A single PNA tower in PNA layers, adapted for PyTorch Geometric."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        aggregators: List[AggregatorEnum],
        scalers: List[ScalarEnum],
        delta: float = 1.0,
        dropout: float = 0.0,
    ):
        super(PyG_PNAConvTower, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.aggregators = aggregators
        self.scalers = scalers
        self.delta = delta

        # Create a list of PyG aggregation modules
        pyg_aggrs = []
        for agg in aggregators:
            if agg.value not in AGGREGATOR_MAP:
                raise ValueError(f"Unknown aggregator: {agg.value}")
            pyg_aggrs.append(AGGREGATOR_MAP[agg.value])

        self.multi_aggr = MultiAggregation(pyg_aggrs)

        self.scaler_funcs = [
            getattr(self, f"_scale_{scaler.value}") for scaler in scalers
        ]

        self.fc_self = nn.Linear(in_size, in_size)
        
        # Calculate the size of the concatenated aggregated features
        # For 'var', StdAggregation is used, so the output dim is the same
        mixer_in_size = len(aggregators) * len(scalers) * in_size
        
        self.U = nn.Linear(mixer_in_size, out_size)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.fc_self.reset_parameters()
        self.U.reset_parameters()
        self.multi_aggr.reset_parameters()

    def forward(self, x_src: Tensor, edge_index: Adj, D: Tensor, edge_attr: OptTensor = None):
        """
        Compute the forward pass of a single tower in PNA convolution layer.
        
        Args:
            x_src (Tensor): Source node features.
            edge_index (Adj): Edge connectivity.
            D (Tensor): Node degrees for scaling.
            edge_attr (OptTensor, optional): Edge features.
        """
        # Isolate target nodes for aggregation
        row, col = edge_index
        num_nodes = D.size(0) # Number of destination nodes

        x = self.fc_self(x_src)

        # Aggregate node features
        # Shape: [num_dst_nodes, len(aggregators) * in_size]
        h = self.multi_aggr(x[row], index=col, dim_size=num_nodes)
        
        # Handle 'var' if requested
        if 'var' in self.aggregators:
            var_idx = self.aggregators.index('var')
            start, end = var_idx * self.in_size, (var_idx + 1) * self.in_size
            h[:, start:end] = h[:, start:end].pow(2) # std -> var

        # Apply scalers to node feature aggregations
        h_list = [scaler_fn(h, D, self.delta) for scaler_fn in self.scaler_funcs]
        
        h_cat = torch.cat(h_list, dim=1)
        h_out = self.U(h_cat)
        return self.dropout(h_out)

    def _scale_identity(self, h, D=None, delta=1.0):
        return h

    def _scale_amplification(self, h, D, delta):
        # Add epsilon to prevent log(0) for isolated nodes
        return h * (torch.log(D + 1) / delta)

    def _scale_attenuation(self, h, D, delta):
        # Add epsilon to avoid division by zero for isolated nodes
        return h * (delta / torch.log(D + 1))


class PyG_PNAConv(nn.Module):

    def __init__(
        self,
        in_size: Union[int, Tuple[int, int]],
        out_size: int,
        has_bias: bool = True,
        negative_slope: float = 0.01,
        tower_dropout: float = 0.0,
        num_towers: int = 1,
        aggregators: List[AggregatorEnum] = [
            AggregatorEnum.mean,
            AggregatorEnum.max,
            AggregatorEnum.min,
            AggregatorEnum.std,
        ],
        scalers: List[ScalarEnum] = [ScalarEnum.identity],
        delta: int = 1
    ):
        super(PyG_PNAConv, self).__init__()
        if isinstance(in_size, int):
            in_size = (in_size, in_size)
        self._in_size_src, self._in_size_dst = in_size
        self._out_size = out_size
        self.num_towers = num_towers
        
        if self._in_size_src % self.num_towers != 0:
            self.num_towers = 1
        
        if out_size % self.num_towers != 0:
            self.num_towers = 1

        self.tower_in_size = self._in_size_src // self.num_towers
        self.tower_out_size = self._out_size // self.num_towers

        self.towers = nn.ModuleList([
            PyG_PNAConvTower(
                self.tower_in_size,
                self.tower_out_size,
                aggregators,
                scalers,
                delta,
                dropout=tower_dropout,
            ) for _ in range(self.num_towers)
        ])

        self.mixing_layer = nn.Sequential(
            nn.Linear(out_size, out_size, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

        self.fc_dst = nn.Linear(
            self._in_size_dst, out_size, bias=has_bias
        )
    
    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        for tower in self.towers:
            tower.reset_parameters()
        
        # Reset mixing layer parameters
        for layer in self.mixing_layer:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # Reset fc_dst parameters
        self.fc_dst.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None):
        if isinstance(x, Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x

        # Handle the case of graphs without edges
        if edge_index.numel() == 0:
            return self.fc_dst(x_dst)

        # Calculate degree for scalers.
        # D has shape [num_dst_nodes, 1] for broadcasting.
        D = degree(edge_index[1], num_nodes=x_dst.size(0)).view(-1, 1).float()

        # Message Passing through towers
        h_towers = []
        for i, tower in enumerate(self.towers):
            x_src_tower = x_src[:, i * self.tower_in_size : (i + 1) * self.tower_in_size]
            h_towers.append(tower(x_src_tower, edge_index, D, edge_attr))
        
        h_cat = torch.cat(h_towers, dim=1)
        h_out = self.mixing_layer(h_cat)
        
        # Skip connection
        rst = self.fc_dst(x_dst) + h_out

        return rst
