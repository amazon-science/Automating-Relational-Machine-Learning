from .base import BaseMPModule
from torch_geometric.typing import Metadata
from .relgnn import RelGNNConv
from typing import Tuple
import torch
from typing import Optional, Dict, List
import warnings
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import NodeType
from torch_geometric.utils.hetero import check_add_self_loops
from torch import Tensor

def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out

class AutoRelGNNConv(BaseMPModule):
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: int,
                 aggr: str,
                 num_heads: int,
                 dropout: float,
                 metadata: Metadata,
                 **kwargs):
        super().__init__(input_channels, hidden_channels, aggr, num_heads, dropout, metadata, **kwargs)
        self.conv = RelGNNConv(
            attn_type=metadata[-1],
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            heads=self.num_heads,
            aggr=self.aggr)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv.reset_parameters()
    
    def forward(self, x: Tuple, edge_index: Tuple):
        return self.conv(x, edge_index)

class AutoRelGNNHeteroConv(torch.nn.Module):
    r"""
    Adapted from https://github.com/snap-stanford/RelGNN
    """
    def __init__(
        self,
        convs,
        aggr: Optional[str] = "sum",
        simplified_MP: Optional[bool] = False,
    ):
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = {key[0] for key in convs.keys()}
        dst_node_types = {key[-1] for key in convs.keys()}
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behavior.")

        self.convs = ModuleDict(convs)
        self.aggr = aggr
        self.simplified_MP = simplified_MP
        self.reset_parameters()
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        x_dict, 
        edge_index_dict,
    ) -> Dict[NodeType, Tensor]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.
        """
        out_dict: Dict[str, List[Tensor]] = {}
        def update(out_dict, dst, out):
            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)

        for edge_type_info, conv in self.convs.items():
            attn_type = edge_type_info[0]

            if attn_type == 'dim-dim':
                src, rel, dst = edge_type_info[1:]
                x = (
                        x_dict.get(src, None),
                        x_dict.get(dst, None),
                    )
                edge_index = edge_index_dict[(src, rel, dst)]

                out = conv(x, edge_index)

                if self.simplified_MP and out is None:
                    continue

                update(out_dict, dst, out)
                        
            elif attn_type == 'dim-fact-dim':
                edge_attn, edge_aggr = edge_type_info[1:4], edge_type_info[4:]
                src_attn, _, dst = edge_attn
                src_aggr = edge_aggr[0]
                x = (
                        x_dict[src_aggr],
                        x_dict[src_attn],
                        x_dict[dst],
                    )
                edge_index = (
                        edge_index_dict[edge_attn],
                        edge_index_dict[edge_aggr],
                    )
                out = conv(x, edge_index)
                
                if self.simplified_MP and out is None:
                    continue

                out_dst, out_src_attn = out
                update(out_dict, dst, out_dst)
                update(out_dict, src_attn, out_src_attn)            


        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        if self.simplified_MP:
            for key, value in x_dict.items():
                if key not in out_dict:
                    out_dict[key] = value

        return out_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'