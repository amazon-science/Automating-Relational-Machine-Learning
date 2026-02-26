from typing import Tuple, Dict, Optional, List, Any, Union
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import registry


# Adapted from https://github.com/pyg-team/pytorch-frame/blob/master/torch_frame/nn/models/resnet.py
class FCResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = "layer_norm",
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        self.norm1: Optional[nn.Module]
        self.norm2: Optional[nn.Module]
        if normalization == "batchnorm":
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        elif normalization == "layernorm":
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            self.norm1 = self.norm2 = None

        self.shortcut: Optional[nn.Module]
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin1(x)
        if self.norm1:
            out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.lin2(out)
        if self.norm2:
            out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.shortcut:
            x = self.shortcut(x)

        out = out + x
        return out


class ResNetTabNNConfig(pydantic.BaseModel):
    hid_size: Optional[int] = 256
    num_layers: Optional[int] = 4
    normalization: Optional[str] = "layer_norm"
    dropout: Optional[float] = 0.2


@registry.tabnn
class ResNetTabNN(nn.Module):
    name = "resnet"
    config_class = ResNetTabNNConfig
    def __init__(
        self,
        config: ResNetTabNNConfig,
        num_fields: int,
        field_size: int,
        out_size: int,
    ):
        super().__init__()
        self.config = config

        if config.num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")

        in_dim = num_fields * field_size

        # Backbone
        self.backbone = nn.ModuleList()
        # First block: maps input dim to hidden dim
        self.backbone.append(FCResidualBlock(
            in_channels=in_dim,
            out_channels=config.hid_size,
            normalization=config.normalization,
            dropout_prob=config.dropout,
        ))

        # Subsequent blocks: operate on hidden dim
        for _ in range(config.num_layers - 1):
            self.backbone.append(FCResidualBlock(
                in_channels=config.hid_size,
                out_channels=config.hid_size,
                normalization=config.normalization,
                dropout_prob=config.dropout,
            ))

        # Output layer
        self.fc_out = nn.Sequential(
            nn.LayerNorm(config.hid_size),
            nn.ReLU(),
            nn.Linear(config.hid_size, out_size),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Flatten input tensor if it's not already 2D
        if X.dim() > 2:
            X = X.view(X.shape[0], -1)

        for block in self.backbone:
            X = block(X)

        X = self.fc_out(X)
        return X