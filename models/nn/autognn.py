import torch 
from models.nn.module.high import AutoRelGNNConv, AutoRelGNNHeteroConv
from torch import Tensor
from torch_geometric.nn import HeteroConv, LayerNorm, BatchNorm
from models.nn.module.sparse import AutoSageConv, AutoPNAConv, AutoHGTConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.data import HeteroData
from torch.nn import ModuleDict, Embedding
from typing import List, Dict, Any
from .encoder import HeteroEncoder, DEFAULT_STYPE_ENCODER_DICT, HeteroTemporalEncoder, TIME_FREE_ENCODER_DICT
from torch_frame.data.stats import StatType
from torch_frame.nn import FTTransformer, ResNet
from torch_geometric.nn import MLP
from torch_geometric.typing import Metadata
from torch_geometric.nn import HeteroJumpingKnowledge
from utils.graph import get_atomic_routes
from models.nn.module.rhs_embedding import RHSEmbedding
from contextgnn.utils import RHSEmbeddingMode
from collections import defaultdict
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from models.nn.encoder import ZeroEncoder

def load_encoder_model_from_str(model_str: str):
    if model_str.lower() == "fttransformer":
        return FTTransformer
    elif model_str.lower() == "resnet":
        return ResNet
    else:
        raise ValueError(f"Invalid model string: {model_str}")

def get_norm_type_from_str(config: dict):
    if config['norm'].lower() == "layernorm":
        return LayerNorm(config['hidden_channels'], mode='node')
    elif config['norm'].lower() == "batchnorm":
        return BatchNorm(config['hidden_channels'])
    elif config['norm'].lower() == "none":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid norm layer: {config['norm']}")


class AutoGNN(torch.nn.Module):
    """
        Generate a GNN architecture based on the selected architecture config file.
        This only deals with message-passing related auto-search
    """
    def __init__(self, src_table: NodeType, 
                 dst_table: NodeType, gnn_config: dict, 
                 edge_types: List[EdgeType], node_types: List[NodeType],
                 data_metadata: Metadata):
        super().__init__()
        self.gnn_config = gnn_config
        self.skip_connection = gnn_config.get('skip_connection', False)
        self.jumping_knowledge = gnn_config.get('jk', False)
        target_edge_type = [
            edge_type for edge_type in edge_types
            if (edge_type[0] == src_table and edge_type[1] == dst_table) or 
            (edge_type[0] == dst_table and edge_type[1] == src_table)
        ]
        self.configure_gnn(node_types, target_edge_type, gnn_config, edge_types, data_metadata)
        self.configure_norm_layer(gnn_config, node_types)
        self.dropout = torch.nn.Dropout(p=gnn_config['dropout'])
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            for node_type, norm_layer in norm.items():
                if hasattr(norm_layer, 'reset_parameters'):
                    norm_layer.reset_parameters()
        
    def extract_gnn_params(self, gnn_config: dict):
        """
        Extract the parameters for the GNN layer
        Use a unified interface:
        Params 1: input_channels, 
        Params 2: hidden_channels,
        Params 3: (optional) aggr,
        Params 4: (optional) num_heads 
        Params 5: (optional) dropout 
        Args:
            gnn_config: The configuration for the GNN layer
        Returns:
            A dictionary of parameters for the GNN layer
        """
        params = {
            'input_channels': gnn_config['hidden_channels'],
            'hidden_channels': gnn_config['hidden_channels'],
            'aggr': gnn_config['aggregation'],
            'num_heads': gnn_config['num_heads'],
            'dropout': gnn_config['dropout'],
            'metadata': gnn_config['metadata']
        }

        return params

    def configure_norm_layer(self, gnn_config: dict, node_types: List[NodeType]):
        self.norms = torch.nn.ModuleList()
        for _ in range(gnn_config['num_layers']):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                get_norm_type = get_norm_type_from_str(gnn_config)
                norm_dict[node_type] = get_norm_type
            self.norms.append(norm_dict)

    def configure_gnn(self, node_types: List[NodeType], target_edge_type: List[EdgeType], 
                      gnn_config: dict, edge_types: List[EdgeType],
                      data_metadata: Metadata):
        self.convs = torch.nn.ModuleList()
        mapped_result = self.return_gnn_layer(gnn_config['mpnn_type'])
        for _ in range(gnn_config['num_layers']):
            if len(mapped_result) == 2:
                wrapper, layer = mapped_result 
                if wrapper is None:
                    gnn_layer = layer(
                        **self.extract_gnn_params(gnn_config)
                    )
                else:
                    if gnn_config['mpnn_type'] == 'relgnn':
                        params = self.extract_gnn_params(gnn_config)
                        hetero_dict = {}
                        atomic_routes = get_atomic_routes(edge_types)
                        for edge_type in atomic_routes:
                            type_metadata = data_metadata + (edge_type[0], )
                            params['metadata'] = type_metadata
                            hetero_dict[edge_type] = layer(**params)
                        gnn_layer = wrapper(
                            hetero_dict, 
                            aggr = 'sum'
                        )
                    else:
                        ## sparse gnn case
                        gnn_layer = wrapper(
                            {
                                edge_type: layer(
                                    **self.extract_gnn_params(gnn_config)
                                ) for edge_type in edge_types
                            }, 
                            aggr = 'sum'
                        )
            else:
                wrapper, layer, second_layer = mapped_result 
                gnn_layer = wrapper(
                    {
                        edge_type: (
                            layer(
                                **self.extract_gnn_params(gnn_config)
                            )
                            if edge_type in target_edge_type
                            else second_layer(
                                **self.extract_gnn_params(gnn_config)
                            )
                        )
                        for edge_type in edge_types
                    },
                    aggr='sum'
                )
            self.convs.append(gnn_layer)
        if self.jumping_knowledge:
            self.jk = HeteroJumpingKnowledge(mode='cat', types=node_types)
            self.jk_lin = torch.nn.Linear(gnn_config['hidden_channels'] * gnn_config['num_layers'], gnn_config['hidden_channels'])

    def return_gnn_layer(self, mpnn_choice: str):
        mapping = {
            'sage': (HeteroConv, AutoSageConv), 
            'hgt': (None, AutoHGTConv), 
            'pna': (HeteroConv, AutoPNAConv), 
            'relgnn': (AutoRelGNNHeteroConv, AutoRelGNNConv)
        }
        return mapping[mpnn_choice]
    
        

    def forward(self, x_dict: Dict[NodeType, Tensor], edge_index_dict: Dict[EdgeType, Tensor]):
        if self.jumping_knowledge:
            x_dict_list = defaultdict(list)
        if self.skip_connection:
            input_x_dict = {node_type: x.clone() for node_type, x in x_dict.items()}
        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict.keys():
                x_dict[node_type] = norm[node_type](x_dict[node_type])
            for node_type in x_dict.keys():
                if self.skip_connection:
                    x_dict[node_type] = self.dropout(x_dict[node_type]) + input_x_dict[node_type]
                else:
                    x_dict[node_type] = self.dropout(x_dict[node_type])
            x_dict = {
                node_type: x_dict[node_type].relu() for node_type in x_dict.keys()
            }
            if self.jumping_knowledge:
                for node_type in x_dict.keys():
                    x_dict_list[node_type].append(x_dict[node_type])
        if self.jumping_knowledge:
            x_dict = self.jk(x_dict_list)
            x_dict = {node_type: self.jk_lin(x_dict[node_type]) for node_type in x_dict.keys()}
        return x_dict






class AutoFullPredictor(torch.nn.Module):
    """
        Auto Module for GNN-based prediction.
        Parts:
        1. Feature encoding
        2. Message-passing (handled by AutoGNN)
        3. Prediction 
    """
    def __init__(self, node_to_col_names_dict: Dict[NodeType, List[str]], 
                 time_types: List[str],
                 num_nodes_dict: Dict[NodeType, int],
                 data_metadata: Metadata,
                 edge_types: List[EdgeType],
                 node_types: List[NodeType],
                 out_channels: int,
                 col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                 model_config: dict, 
                 src_table: NodeType,
                 dst_table: NodeType,
                 strategy: str, 
                 time_free: bool = False,
                 zero_init: bool = False,
                 **kwargs):
        """
            kwargs is used to eliminate unnecessary parameters
        """
        super().__init__()

        model_config['channels'] = model_config['gnn_config']['hidden_channels']
        self.zero_init = zero_init
        if zero_init:
            self.feature_encoder = ZeroEncoder(
                channels=model_config['channels']
            )
        else:
            self.feature_encoder = HeteroEncoder(
                channels=model_config['channels'],
                node_to_col_names_dict=node_to_col_names_dict,
                node_to_col_stats=col_stats_dict,
                stype_encoder_cls_kwargs=TIME_FREE_ENCODER_DICT if time_free else DEFAULT_STYPE_ENCODER_DICT,
                torch_frame_model_cls=load_encoder_model_from_str(model_config['torch_frame_model_cls']),
                torch_frame_model_kwargs={
                    "channels": model_config['channels'],
                    "num_layers": model_config["encoder_num_layers"]
                },
                time_free = time_free
            )
        self.time_free = time_free
        if not time_free:
            self.temporal_encoder = HeteroTemporalEncoder(
                node_types=time_types,
                channels=model_config['channels'],
        )
        else:
            self.temporal_encoder = None
        model_config['gnn_config']['input_channels'] = model_config['channels']
        model_config['gnn_config']['num_layers'] = model_config['sampler_config']['num_layers']
        model_config['gnn_config']['metadata'] = data_metadata
        submodule_config = {
            'gnn_config': model_config['gnn_config'],
            'src_table': src_table,
            'dst_table': dst_table,
            'edge_types': edge_types,
            'node_types': node_types,
            'data_metadata': data_metadata
        }
        self.model_config = model_config
        self.configure_auto_structural_feature(num_nodes_dict, src_table, dst_table, strategy)
        self.out_channels = out_channels
        self.gnn = AutoGNN(**submodule_config)
        if model_config['gnn_config']['post_sf'] == 'ncn':
            self.head_type = 'ncn'
        elif model_config['sampler_config']['loader_type'] == 'link':
            self.head_type = 'mlp'
            self.out_channels = model_config['channels']
        elif model_config['gnn_config']['post_sf'] == 'shallow':
            self.head_type = 'shallowrhs'
        elif model_config['gnn_config']['post_sf'] == 'contextgnn':
            self.head_type = 'contextgnn'
        elif model_config['gnn_config']['post_sf'] == 'identity':
            self.head_type = 'identity'
        else:
            self.head_type = 'mlp'
        self.configure_head(self.head_type, num_nodes_dict=num_nodes_dict, model_config=model_config, 
                            col_stats_dict=col_stats_dict, col_names_dict=node_to_col_names_dict,
                            stype_encoder_dict=DEFAULT_STYPE_ENCODER_DICT, src_table=src_table, dst_table=dst_table)
        self.reset_parameters()

    def reset_parameters(self):
        if not self.zero_init:
            self.feature_encoder.reset_parameters()
        if self.temporal_encoder is not None:
            self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        if self.head_type == 'mlp' or self.head_type == 'contextgnn':
            self.head.reset_parameters()
        if self.head_type == 'shallowrhs':
            self.rhs_embedding.reset_parameters()
            self.lhs_projector.reset_parameters()
        elif self.head_type == 'contextgnn':
            self.rhs_embedding.reset_parameters()
            self.lhs_projector.reset_parameters()
            self.lin_offset_embgnn.reset_parameters()
            self.lin_offset_idgnn.reset_parameters()
        

    def configure_auto_structural_feature(self, num_nodes_dict: Dict[NodeType, int], src_table: NodeType, dst_table: NodeType, strategy: str):
        if strategy == 'src_dst':
            self.pre_embedding = ModuleDict(
                {
                    node_type: Embedding(num_nodes_dict[node_type], self.model_config['gnn_config']['hidden_channels'])
                    for node_type in [src_table, dst_table]
                }
            )
            torch.nn.init.orthogonal_(self.pre_embedding[src_table].weight)
            torch.nn.init.orthogonal_(self.pre_embedding[dst_table].weight)
        elif strategy == 'zero_learn':
            self.pre_embedding = torch.nn.Embedding(1, self.model_config['gnn_config']['hidden_channels'])
            # torch.nn.init.orthogonal_(self.pre_embedding.weight)
            self.pre_embedding.reset_parameters()
        elif strategy == 'zero_one':
            self.pre_embedding = torch.ones(1, self.model_config['gnn_config']['hidden_channels'])
        elif strategy == 'none':
            self.pre_embedding = None
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        
          
    def apply_auto_structural_feature(self, x_dict: Dict[NodeType, Tensor], data: HeteroData, src_table: NodeType, dst_table: NodeType, strategy: str) -> Dict[NodeType, Tensor]:
        if strategy == 'none':
            pass 
        elif strategy == 'src_dst':
            for node_type, embedding in self.pre_embedding.items():
                x_dict[node_type] += embedding(data[node_type].n_id)
        elif strategy == 'zero_learn':
            seed_time = data[src_table].seed_time
            x_dict[src_table][:seed_time.size(0)] += self.pre_embedding.weight
        elif strategy == 'zero_one':
            seed_time = data[src_table].seed_time
            self.pre_embedding = self.pre_embedding.to(seed_time.device)
            x_dict[src_table][:seed_time.size(0)] += self.pre_embedding
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        return x_dict 
    
    def configure_rhs_embedding(self, num_nodes_dict, model_config, col_stats_dict, col_names_dict,
                                stype_encoder_dict, which_table: NodeType):
        if self.head_type not in ['shallowrhs', 'contextgnn']:
            self.rhs_embedding = None
            return
        self.rhs_embedding = RHSEmbedding(
            emb_mode=RHSEmbeddingMode.LOOKUP,
            embedding_dim=model_config['gnn_config']['hidden_channels'],
            num_nodes=num_nodes_dict[which_table],
            col_stats=col_stats_dict,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict
        )
        
    def configure_head(self, head_type: str, num_nodes_dict, model_config, col_stats_dict, col_names_dict,
                                stype_encoder_dict, src_table: NodeType, dst_table: NodeType):
        if head_type == 'mlp':
            self.head = MLP(
                in_channels=self.model_config['gnn_config']['hidden_channels'],
                out_channels = self.out_channels, 
                num_layers = 1, 
                norm_layer = self.model_config['gnn_config']['norm']
            )
        elif head_type == 'ncn':
            self.head = torch.nn.Identity()
        elif head_type == 'identity':
            self.head = torch.nn.Identity()
        elif head_type == 'shallowrhs':
            self.configure_rhs_embedding(num_nodes_dict, model_config, col_stats_dict, col_names_dict,
                                stype_encoder_dict, dst_table)
            self.lhs_projector = torch.nn.Linear(
                in_features=model_config['gnn_config']['hidden_channels'],
                out_features=model_config['gnn_config']['hidden_channels']
            )
        elif head_type == 'contextgnn':
            self.configure_rhs_embedding(num_nodes_dict, model_config, col_stats_dict, col_names_dict,
                                stype_encoder_dict, dst_table)
            self.lhs_projector = torch.nn.Linear(
                in_features=model_config['gnn_config']['hidden_channels'],
                out_features=model_config['gnn_config']['hidden_channels']
            )
            self.lin_offset_embgnn = torch.nn.Linear(
                in_features=model_config['gnn_config']['hidden_channels'],
                out_features=1
            )
            self.lin_offset_idgnn = torch.nn.Linear(
                in_features=model_config['gnn_config']['hidden_channels'],
                out_features=1
            )
            self.head = MLP(
                in_channels=self.model_config['gnn_config']['hidden_channels'],
                out_channels = 1, 
                num_layers = 1, 
                norm_layer = self.model_config['gnn_config']['norm']
            )
        else:
            raise ValueError(f"Invalid head type: {head_type}")

    def construct_logits(self, lhs_embedding_projected, lhs_embedding,
                         rhs_gnn_embedding, rhs_embedding, lhs_idgnn_batch,
                         rhs_idgnn_index):
        embgnn_logits = lhs_embedding_projected @ rhs_embedding.t(
        )  # batch_size, num_rhs_nodes

        # Model the importance of embedding-GNN prediction for each lhs node
        embgnn_offset_logits = self.lin_offset_embgnn(
            lhs_embedding_projected).flatten()
        embgnn_logits += embgnn_offset_logits.view(-1, 1)

        # Calculate idgnn logits
        idgnn_logits = self.head(
            rhs_gnn_embedding).flatten()  # num_sampled_rhs
        # Because we are only doing 2 hop, we are not really sampling info from
        # lhs therefore, we need to incorporate this information using
        # lhs_embedding[lhs_idgnn_batch] * rhs_gnn_embedding
        idgnn_logits += (
            lhs_embedding[lhs_idgnn_batch] *  # num_sampled_rhs, channel
            rhs_gnn_embedding).sum(
                dim=-1).flatten()  # num_sampled_rhs, channel

        # Model the importance of ID-GNN prediction for each lhs node
        idgnn_offset_logits = self.lin_offset_idgnn(
            lhs_embedding_projected).flatten()
        idgnn_logits = idgnn_logits + idgnn_offset_logits[lhs_idgnn_batch]

        embgnn_logits[lhs_idgnn_batch, rhs_idgnn_index] = idgnn_logits
        return embgnn_logits 


    def forward(self, 
                batch: HeteroData,
                entity_table: NodeType,
                dst_table: NodeType,
                read_out_table: str = 'dst', 
                **kwargs):
        seed_time = batch[entity_table].seed_time
        x_dict = self.feature_encoder(batch.tf_dict)
        x_dict = self.apply_auto_structural_feature(x_dict, batch, entity_table, dst_table, 
                                                    self.model_config['gnn_config']['pre_sf'])
        if not self.time_free:
            rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                                  batch.batch_dict)
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time
        
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )
        if self.head_type == 'mlp':
            if read_out_table == 'dst':
                return self.head(x_dict[dst_table])
            elif read_out_table == 'src':
                return self.head(x_dict[entity_table])
            else:
                raise ValueError(f"Invalid read out table: {read_out_table}")
        elif self.head_type == 'ncn':
            raise NotImplementedError("NCN head type is not yet implemented")
        elif self.head_type == 'identity':
            return x_dict[dst_table if read_out_table == 'dst' else entity_table] 
        elif self.head_type == 'shallowrhs':
            lhs_embedding = self.lhs_projector(x_dict[entity_table][:seed_time.size(0)])
            rhs_embedding = self.rhs_embedding()
            return lhs_embedding @ rhs_embedding.t()
        elif self.head_type == 'contextgnn':
            lhs_embedding = x_dict[entity_table][:seed_time.size(0)]
            lhs_embedding_projected = self.lhs_projector(lhs_embedding)
            rhs_gnn_embedding = x_dict[dst_table]  # num_sampled_rhs, channel
            rhs_idgnn_index = batch.n_id_dict[dst_table]  # num_sampled_rhs
            lhs_idgnn_batch = batch.batch_dict[dst_table]  # batch_size
            rhs_embedding = self.rhs_embedding()
            embgnn_logits = self.construct_logits(
                lhs_embedding_projected=lhs_embedding_projected,
                lhs_embedding=lhs_embedding,
                rhs_gnn_embedding=rhs_gnn_embedding,
                rhs_embedding=rhs_embedding,
                lhs_idgnn_batch=lhs_idgnn_batch,
                rhs_idgnn_index=rhs_idgnn_index
            )
            return embgnn_logits
        else:
            raise ValueError(f"Invalid head type: {self.head_type}")