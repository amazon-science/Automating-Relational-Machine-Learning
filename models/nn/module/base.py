import torch
from torch_geometric.typing import Metadata


class BaseMPModule(torch.nn.Module):
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: int,
                 aggr: str,
                 num_heads: int,
                 dropout: float,
                 metadata: Metadata,
                 **kwargs):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.aggr = aggr
        self.num_heads = num_heads
        self.dropout = dropout
        self.metadata = metadata

        