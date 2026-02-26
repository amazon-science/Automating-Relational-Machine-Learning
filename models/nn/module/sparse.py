from .base import BaseMPModule
from torch_geometric.nn.conv import SAGEConv, HGTConv, HeteroConv
from torch_geometric.typing import Metadata, NodeType, EdgeType, Tensor
from typing import Dict, Tuple
from torch import Tensor
from models.nn.module.pna import PyG_PNAConv

import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Union

class AutoSageConv(BaseMPModule):
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: int,
                 aggr: str,
                 num_heads: int,
                 dropout: float,
                 metadata: Metadata,
                 **kwargs):
        super().__init__(input_channels, hidden_channels, aggr, num_heads, dropout, metadata, **kwargs)
        self.conv = SAGEConv(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            aggr=self.aggr
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: Tuple[Tensor, Tensor], edge_index: Tensor):
        return self.conv(x, edge_index)


class AutoPNAConv(BaseMPModule):
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: int,
                 aggr: str,
                 num_heads: int,
                 dropout: float,
                 metadata: Metadata,
                 **kwargs):
        super().__init__(input_channels, hidden_channels, aggr, num_heads, dropout, metadata, **kwargs)
        self.conv = PyG_PNAConv(
            in_size = (self.input_channels, self.input_channels),
            out_size = self.hidden_channels,
            has_bias = True,
            negative_slope=0.01,
            tower_dropout=0, 
            num_towers=self.num_heads)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: Tuple[Tensor, Tensor], edge_index: Tensor):
        return self.conv(x, edge_index)

class AutoHGTConv(BaseMPModule):
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: int,
                 aggr: str,
                 num_heads: int,
                 dropout: float,
                 metadata: Metadata,
                 **kwargs):
        super().__init__(input_channels, hidden_channels, aggr, num_heads, dropout, metadata, **kwargs)
        self.conv = HGTConv(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            metadata=self.metadata, 
            heads=self.num_heads)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv.reset_parameters()
    
    def forward(self, x_dict: Dict[NodeType, Tensor], edge_index_dict: Dict[EdgeType, Tensor]):
        return self.conv(x_dict, edge_index_dict)


# class SimplifiedRMPNNConv(MessagePassing):
#     """
#     A simplified version of RMPNNConv that does not use edge attributes.
#     It computes the mean of neighboring node features for a single relation type.
#     The `hiddim` argument is kept for API consistency but is not used in this layer.
#     """
#     def __init__(self, hiddim: int):
#         super().__init__(aggr='mean')

#     def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], edge_index: Tensor) -> Tensor:
#         # The forward pass no longer accepts or uses edge_attr.
#         return self.propagate(edge_index, x=x)

#     def message(self, x_j: Tensor) -> Tensor:
#         # The message is simply the feature vector of the neighboring node.
#         return x_j


# class AutoGriffinConv(BaseMPModule):
#     def __init__(self, 
#                  input_channels: int,
#                  hidden_channels: int,
#                  aggr: str,
#                  num_heads: int,
#                  dropout: float,
#                  metadata: Metadata,
#                  **kwargs):
#         super().__init__(input_channels, hidden_channels, aggr, num_heads, dropout, metadata, **kwargs)
#         node_types = self.metadata[0]
#         self.gatelin = nn.ModuleDict({
#                 ntype: nn.Sequential(nn.Linear(self.hidden_channels, int(self.hidden_channels**0.5)), nn.SiLU(True), nn.Linear(int(self.hidden_channels**0.5), 1, False))
#                 for ntype in node_types
#             })
#         self.forward_mpnn = HeteroConv(
#             {
#                 edge_type: SimplifiedRMPNNConv(self.hidden_channels)
#                 for edge_type in self.metadata[1]
#             },
#             aggr='sum'
#         )
    
#     def forward(self, x_dict: Dict[NodeType, Tensor], edge_index_dict: Dict[EdgeType, Tensor]):
#         mp_out = self.forward_mpnn(x_dict, edge_index_dict)
#         for ntype in mp_out.keys():
#             gate_val = self.gatelin[ntype](x_dict[ntype])
#             mp_out[ntype] = gate_val * mp_out[ntype]
#         return mp_out
        
        
        
        

        