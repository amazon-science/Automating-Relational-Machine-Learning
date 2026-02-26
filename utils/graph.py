import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from data.relbench import DatasetwithTransform
from torch_frame.data.stats import StatType
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import sort_edge_index
from relbench.base import Database
from relbench.modeling.utils import remove_pkey_fkey, to_unix_time
from loguru import logger
# from models.autog.agent import load_dbb_dataset_from_cfg_path_no_name
from relbench.base import Table
from collections import defaultdict
from relbench.datasets import Dataset
from relbench.datasets import get_dataset_names
from data.relbench import load_relbench_data, load_relbench_task
from pathlib import Path
from torch_frame import stype, TensorFrame
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Optional, Tuple, List

def retrieve_meta_graph_from_constructed_data(data, task, channels, labeling_trick='zo', task_type='node'):
    """Retrieve the meta graph from the constructed data.

    Args:
        data: the constructed HeteroData
        task: the task object
        channels: the number of channels
        labeling_trick: 'zo' labels source node as 1; 'zod' labels source
            and destination as 1; others as 0
        task_type: 'node' or 'link'
    """
    node_type_to_id = {node_type: i for i, node_type in enumerate(data.node_types)}
    adj_matrix = torch.zeros(len(data.node_types), len(data.node_types))
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        src_id = node_type_to_id[src_type]
        dst_id = node_type_to_id[dst_type]
        adj_matrix[src_id, dst_id] = 1
    edge_index = torch.nonzero(adj_matrix).t()
    x = torch.zeros(len(data.node_types), channels)
    meta_data = Data(x=x, edge_index=edge_index, edge_attr=None)
    if task_type == 'node':
        src_entity_table = task.entity_table
        dst_entity_table = task.entity_table
    elif task_type == 'link':
        src_entity_table = task.src_entity_table
        dst_entity_table = task.dst_entity_table
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Choose 'node' or 'link'.")
    if labeling_trick == 'zo':
        source_entity_index = node_type_to_id[src_entity_table]
        meta_data.x[source_entity_index] = torch.ones(channels)
    elif labeling_trick == 'zod':
        source_entity_index = node_type_to_id[src_entity_table]
        dst_entity_index = node_type_to_id[dst_entity_table]
        meta_data.x[source_entity_index] = torch.ones(channels)
        meta_data.x[dst_entity_index] = torch.ones(channels)
    return meta_data, node_type_to_id


class CustomDataset(Dataset):
    def get_db(self):
        db_path = os.path.join(self.cache_dir, 'db')
        os.makedirs(db_path, exist_ok=True)
        if self.cache_dir and Path(db_path).exists() and any(Path(db_path).iterdir()):
            db = Database.load(db_path)
        else:
            raise ValueError(f"Database not found at {db_path}")
        return db

def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
    transform: Optional[str] = 'none',
    dataset_name: Optional[str] = None
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
    foreign key relationships, together with the column stats of each table.

    Args:
        db: A database object containing a set of tables.
        col_to_stype_dict: Column to stype for
            each table.
        text_embedder_cfg: Text embedder config.
        transform: The transformation to apply to the dataset.
        dataset_name: The name of the dataset.
        cache_dir: A directory for storing materialized tensor
            frames. If specified, we will either cache the file or use the
            cached file. If not specified, we will not use cached file and
            re-process everything from scratch without saving the cache.

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature.
    """
    data = HeteroData()
    col_stats_dict = dict()
    new_col_to_stype_dict = dict()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None and table.pkey_col != '':
            assert (df[table.pkey_col].values == np.arange(len(df))).all()
        
        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        # Determine postfix for LLM encoder if needed
        llm_postfix = ""
        if (
            text_embedder_cfg is not None 
            and getattr(text_embedder_cfg, "encoder_name", None) == "llm"
        ):
            llm_postfix = "_llm"
        transformed_name = f"{dataset_name}_{transform}" if transform != "" else dataset_name
        if cache_dir is not None:
            out_dir = os.path.join(cache_dir, f"{transformed_name}{llm_postfix}")
            path = os.path.join(out_dir, f"{table_name}.pt")
        else:
            path = None
        logger.info("Load data with transform: {}".format(transform))
        dataset = DatasetwithTransform(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
            dataset_name=dataset_name
        ).materialize(path=path, transform=transform)

        data[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats
        new_col_to_stype_dict[table_name] = dataset.col_to_stype

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()

    return data, col_stats_dict, new_col_to_stype_dict


def normalize_numeric_f(
    tf_dict: Dict[str, TensorFrame],
    skew_threshold: float = 0.99
) -> Dict[str, TensorFrame]:
    """
    Intelligently normalizes numerical columns within multiple TensorFrames.

    This function iterates through specified numerical columns in each TensorFrame,
    checks their distribution's skewness, and applies an appropriate
    normalization strategy:
    - For highly skewed data: Imputes missing values and applies a Quantile
      Transformer.
    - For data that is not highly skewed: Imputes missing values and applies
      a standard Z-score scaler.
    - Multi-dimensional columns (ndim > 1) are skipped.

    Args:
        tf_dict: A dictionary where keys are table names and values are
                 TensorFrame objects.
        skew_threshold: The absolute skewness value above which a column is
                        considered "highly skewed". Defaults to 1.0.

    Returns:
        The dictionary of TensorFrames with numerical columns normalized in-place.
    """
    logger.info("Starting numerical normalization for TensorFrames...")

    # Iterate over each table's TensorFrame in the dictionary
    for table_name, tf in tf_dict.items():
        logger.info(f"  Processing table: '{table_name}'")

        # Check if there are any numerical features to process
        if stype.numerical not in tf.feat_dict:
            continue

        numerical_data_tensor = tf.feat_dict[stype.numerical]

        # Iterate over each numerical column's name and data
        for i in range(numerical_data_tensor.shape[1]):
            col_data = numerical_data_tensor[:, i]
            # --- Logic from NormNumericTransform starts here ---

            # Use pandas Series for convenient stats calculation and NaN handling
            col_series = pd.Series(col_data).dropna()

            # Ensure we have data to process after dropping NaNs
            if col_series.empty:
                # logger.info(f"    - Skipping column as it contains only NaN values.")
                continue

            # 2. Analyze data distribution by calculating skewness
            skew_score = col_series.skew()
            if pd.isna(skew_score):
                skew_score = 0 # Treat columns with no variance as not skewed

            pipeline: Pipeline

            # 3. Choose the normalization strategy based on skewness
            if abs(skew_score) > skew_threshold:
                # Data is skewed: use a robust imputer and a quantile transformer
                # logger.info(f"    - Column is skewed (skewness: {skew_score:.2f}). Applying QuantileTransformer.")
                pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(col_series) // 10, 1000), 10)))
                ])
            else:
                # Data is not skewed: use a standard imputer and scaler
                #logger.info(f"    - Column is not highly skewed (skewness: {skew_score:.2f}). Applying StandardScaler.")
                pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])

            # Reshape data for scikit-learn, fit the pipeline, and transform the column
            data_to_transform = col_data.reshape(-1, 1)
            transformed_data = pipeline.fit_transform(data_to_transform)

            # Update the data in the dictionary with the normalized data
            numerical_data_tensor[:, i] = torch.from_numpy(transformed_data.flatten()).float()
        tf_dict[table_name].feat_dict[stype.numerical] = numerical_data_tensor
    logger.info("Numerical normalization complete.")
    return tf_dict


def return_embed_feature_dict(db: Database, col_to_stype_dict: Dict[str, Dict[str, stype]], 
                         text_embedder_cfg: Optional[TextEmbedderConfig] = None,
                         cache_dir: Optional[str] = None,
                         transform: Optional[str] = 'none',
                         dataset_name: Optional[str] = None, 
                         normalize_numeric: bool = True, 
                         drop_text: bool = False) -> Dict:
    output_dict = dict()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None and table.pkey_col != '':
            assert (df[table.pkey_col].values == np.arange(len(df))).all()
        
        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, dataset_name, f"{table_name}.pt")
        )
        if cache_dir is not None:
            os.makedirs(os.path.join(cache_dir, dataset_name), exist_ok=True)
        logger.info("Load data with transform: {}".format(transform))
        dataset = DatasetwithTransform(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
            dataset_name=dataset_name
        ).materialize(path=path, transform=transform)

        if drop_text:
            if stype.embedding in dataset.tensor_frame.feat_dict:
                del dataset.tensor_frame.feat_dict[stype.embedding]
            if stype.embedding in dataset.tensor_frame.col_names_dict:
                del dataset.tensor_frame.col_names_dict[stype.embedding]
        
        if dataset.tensor_frame.num_rows == 0:  # Add constant feature in case df is empty:
            col_to_stype["__const__"] = stype.numerical
            col_to_stype_dict[table_name]["__const__"] = stype.numerical
            dataset.tensor_frame.feat_dict[stype.numerical] = torch.ones(1, len(table.df))

        output_dict[table_name] = dataset.tensor_frame
        # col_stats_dict[table_name] = dataset.col_stats
    
    if normalize_numeric:
        output_dict = normalize_numeric_f(output_dict)

    return output_dict, col_to_stype_dict


def make_pkey_fkey_graph_r2ne_for_heuristics(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
    transform: Optional[str] = 'none',
    dataset_name: Optional[str] = None,
    entity_table: Optional[str] = None,
    skip_time: bool = False,
) -> tuple[HeteroData, dict]:
    """
    PLAN A:
      1) Create a node type for *every* table (even 2-FK 'edge-tables' and multi-FK tables).
         If a table has no usable features OR is not the entity table, attach a constant feature.
      2) For every table, create structural FK→PK edges (f2p_* and rev_f2p_*).
      3) For tables with >2 FKs that include the entity_table among their PK targets, also create
         star-shaped 'multi_*' edges from entity_table to each other referenced table (+ reverse).
      4) For tables with exactly 2 FKs, additionally create relation edges between the two endpoint
         PK tables (rel_{table} and rev_rel_{table}), aligning edge features with masks.

    This design guarantees no edge points to a non-existing node type.

    Returns:
        data: PyG HeteroData
        col_stats_dict: per-(node/edge)-TF column stats
    """
    data = HeteroData()
    col_stats_dict: Dict[str, Any] = {}

    if cache_dir is not None and dataset_name is not None:
        os.makedirs(os.path.join(cache_dir, f'{dataset_name}_r2ne'), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, f'{dataset_name}_r2ne_heuristics'), exist_ok=True)

    # ---------- 0) Classify tables for extra edge projections ----------
    role: Dict[str, set] = {}
    for tname, tbl in db.table_dict.items():
        num_fk = len(tbl.fkey_col_to_pkey_table)
        r = set()
        if num_fk == 2:
            r.add('edge')  # will additionally emit relation edges
        if num_fk > 2 and (entity_table in tbl.fkey_col_to_pkey_table.values()):
            r.add('multi-edge')  # will additionally emit multi-star edges
        role[tname] = r
    logger.info(f"Table roles for extra projections: {role}")

    # ---------- 1) Create a node for every table ----------
    for table_name, table in db.table_dict.items():
        # unify possibly empty PK names
        if table.pkey_col == '':
            table.pkey_col = None

        df = table.df
        col_to_stype = dict(col_to_stype_dict.get(table_name, {}))  # copy to avoid side-effects

        # PK sanity: if present, must be 0..N-1
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all(), \
                f"Primary key of {table_name} must be 0..N-1"

        # strip PK/FK columns from features
        remove_pkey_fkey(col_to_stype, table)

        ## we only need structure, so no need to materialize features
        col_to_stype = {"__const__": stype.numerical}
        df_node = pd.DataFrame({"__const__": np.ones(len(df))})


        path = (
            None if cache_dir is None or dataset_name is None
            else os.path.join(cache_dir, f'{dataset_name}_r2ne', f"{table_name}_node.pt")
        )
        dataset = DatasetwithTransform(
            df=df_node,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
            dataset_name=dataset_name
        ).materialize(path=path, transform=transform)

        data[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # optional node timestamp (skipped when only structure is needed)
        if not skip_time and table.time_col is not None:
            data[table_name].time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    # ---------- 2) Structural FK→PK edges for ALL tables ----------
    for table_name, table in db.table_dict.items():
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            # source index is row-id of the FK table (0..len-1); select rows with non-null FK
            pkey_series = table.df[fkey_name]
            mask_np = (~pkey_series.isna()).values
            fkey_index = torch.arange(len(pkey_series))[torch.from_numpy(mask_np)]
            pkey_index = torch.from_numpy(pkey_series[mask_np].astype(int).values)

            assert (pkey_index < len(db.table_dict[pkey_table_name].df)).all(), \
                f"Dangling FK {table_name}.{fkey_name} → {pkey_table_name}"

            e = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            re = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)

            data[e].edge_index = sort_edge_index(torch.stack([fkey_index, pkey_index]))
            data[re].edge_index = sort_edge_index(torch.stack([pkey_index, fkey_index]))

            # Optional: project a timestamp from source rows (aligned with fkey_index)
            if table.time_col is not None:
                edge_time = torch.from_numpy(to_unix_time(table.df[table.time_col][mask_np]))
                data[e].time = edge_time
                data[re].time = edge_time

    # ---------- 3) Multi-edge star from entity_table ----------
    if entity_table is not None:
        for table_name, table in db.table_dict.items():
            if 'multi-edge' not in role[table_name]:
                continue

            df_me = table.df
            # locate the FK column that points to the entity table
            entity_fkey = None
            others: List[Tuple[str, str]] = []
            for fk, tgt in table.fkey_col_to_pkey_table.items():
                if tgt == entity_table:
                    entity_fkey = fk
                else:
                    others.append((fk, tgt))
            if entity_fkey is None:
                logger.warning(f"[multi-edge] {table_name} has no FK to entity '{entity_table}'")
                continue

            ent_vals = df_me[entity_fkey]
            for other_fk, other_tbl in others:
                oth_vals = df_me[other_fk]
                mask_np = (~ent_vals.isna() & ~oth_vals.isna()).values

                src = torch.from_numpy(ent_vals[mask_np].astype(int).values)  # entity ids
                dst = torch.from_numpy(oth_vals[mask_np].astype(int).values)  # other-table ids

                assert (src < len(db.table_dict[entity_table].df)).all()
                assert (dst < len(db.table_dict[other_tbl].df)).all()

                e = (entity_table, f"multi_{table_name}_{other_fk}", other_tbl)
                re = (other_tbl, f"rev_multi_{table_name}_{other_fk}", entity_table)

                data[e].edge_index = sort_edge_index(torch.stack([src, dst]))
                data[re].edge_index = sort_edge_index(torch.stack([dst, src]))

                # Edge features: by default constant-1 aligned to mask; or keep selected cols if desired
                col_to_stype = dict(col_to_stype_dict.get(table_name, {}))
                remove_pkey_fkey(col_to_stype, table)
                col_to_stype = {"__const__": stype.numerical}
                df_feat = pd.DataFrame({"__const__": np.ones(len(df_me))})


                path = (
                    None if cache_dir is None or dataset_name is None
                    else os.path.join(cache_dir, f'{dataset_name}_r2ne', f"{table_name}_multi_edge_{other_fk}.pt")
                )
                dataset = DatasetwithTransform(
                    df=df_feat[mask_np],
                    col_to_stype=col_to_stype,
                    col_to_text_embedder_cfg=text_embedder_cfg,
                    dataset_name=dataset_name
                ).materialize(path=path, transform=transform)

                data[e].tf = dataset.tensor_frame
                data[re].tf = dataset.tensor_frame
                col_stats_dict[f'multi_{table_name}_{other_fk}'] = dataset.col_stats
                col_stats_dict[f'rev_multi_{table_name}_{other_fk}'] = dataset.col_stats

                if table.time_col is not None:
                    edge_time = torch.from_numpy(to_unix_time(df_me[table.time_col][mask_np]))
                    data[e].time = edge_time
                    data[re].time = edge_time
                    assert len(edge_time) == data[e].edge_index.shape[1]

    # ---------- 4) Relation edges for 2-FK tables ----------
    for table_name, table in db.table_dict.items():
        if 'edge' not in role[table_name]:
            continue

        df_e = table.df
        fks = list(table.fkey_col_to_pkey_table.keys())
        fkey1_name, fkey2_name = fks[0], fks[1]
        src_tbl = table.fkey_col_to_pkey_table[fkey1_name]
        dst_tbl = table.fkey_col_to_pkey_table[fkey2_name]

        src_series = df_e[fkey1_name]
        dst_series = df_e[fkey2_name]
        mask_np = (~src_series.isna() & ~dst_series.isna()).values

        src_ids = torch.from_numpy(src_series[mask_np].astype(int).values)
        dst_ids = torch.from_numpy(dst_series[mask_np].astype(int).values)

        assert (src_ids < len(db.table_dict[src_tbl].df)).all()
        assert (dst_ids < len(db.table_dict[dst_tbl].df)).all()

        e = (src_tbl, f"rel_{table_name}", dst_tbl)
        re = (dst_tbl, f"rev_rel_{table_name}", src_tbl)

        data[e].edge_index = sort_edge_index(torch.stack([src_ids, dst_ids]))
        data[re].edge_index = sort_edge_index(torch.stack([dst_ids, src_ids]))

        # Edge features: default constant-1 unless you specifically want features from this table
        col_to_stype = dict(col_to_stype_dict.get(table_name, {}))
        remove_pkey_fkey(col_to_stype, table)

        col_to_stype = {"__const__": stype.numerical}
        df_feat = pd.DataFrame({"__const__": np.ones(len(df_e))})

        path = (
            None if cache_dir is None or dataset_name is None
            else os.path.join(cache_dir, f'{dataset_name}_r2ne_heuristics', f"{table_name}_edge.pt")
        )
        dataset = DatasetwithTransform(
            df=df_feat[mask_np],
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
            dataset_name=dataset_name
        ).materialize(path=path, transform=transform)

        data[e].tf = dataset.tensor_frame
        data[re].tf = dataset.tensor_frame
        col_stats_dict[f"rel_{table_name}"] = dataset.col_stats
        col_stats_dict[f"rev_rel_{table_name}"] = dataset.col_stats

        if table.time_col is not None:
            edge_time = torch.from_numpy(to_unix_time(df_e[table.time_col][mask_np]))
            data[e].time = edge_time
            data[re].time = edge_time
            assert len(edge_time) == data[e].edge_index.shape[1]

    # ---------- final checks ----------
    data.validate()
    return data, col_stats_dict

    

def make_pkey_fkey_graph_with_edge(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
    transform: Optional[str] = 'none',
    dataset_name: Optional[str] = None,
) -> tuple[HeteroData, dict]:
    """
    Constructs a heterogeneous graph from a database following the r2ne method.

    This method models tables as either nodes or edges based on their schema:
    - Tables with a primary key or a number of foreign keys not equal to two
      are represented as nodes.
    - Tables with exactly two foreign keys and no primary key are represented as
      edges connecting the two referenced tables.

    Args:
        db: The Database object containing the tables.
        col_to_stype_dict: A dictionary mapping table names to their column
            semantic types.
        text_embedder_cfg: Configuration for the text embedder.
        cache_dir: Directory to cache materialized tensor frames.
        transform: The transformation to apply to the dataset.
        dataset_name: The name of the dataset.

    Returns:
        A tuple containing the constructed HeteroData object and a dictionary
        of column statistics.
    """
    data = HeteroData()
    col_stats_dict = {}
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    # 1. Determine the role of each table (node or edge)
    table_domain = {}
    for table_name, table in db.table_dict.items():
        num_fkey = len(table.fkey_col_to_pkey_table)
        no_pkey = table.pkey_col is None or table.pkey_col == ""
        if num_fkey == 2 and no_pkey:
            table_domain[table_name] = 'edge'
        else:
            table_domain[table_name] = 'node'
    
    logger.info(f"Table domain mapping: {table_domain}")

    os.makedirs(os.path.join(cache_dir, f'{dataset_name}_r2ne'), exist_ok=True)

    # 2. Create nodes and materialize features for node tables
    edge_table_dfs = {} # Store dataframes for tables that will become edges
    for table_name, table in db.table_dict.items():
        ## unify null pkey 
        if table.pkey_col == '': table.pkey_col = None
        if table_domain[table_name] == 'node':
            logger.info(f"Processing '{table_name}' as a node table.")
            df = table.df
            col_to_stype = col_to_stype_dict[table_name]

            # Primary key check
            if table.pkey_col is not None:
                assert (df[table.pkey_col].values == np.arange(len(df))).all()

            remove_pkey_fkey(col_to_stype, table)

            if len(col_to_stype) == 0:
                col_to_stype = {"__const__": stype.numerical}
                df = pd.DataFrame({"__const__": np.ones(len(table.df))})

            path = (
                None if cache_dir is None
                else os.path.join(cache_dir, f'{dataset_name}_r2ne', f"{table_name}_node.pt")
            )

            dataset = DatasetwithTransform(
                df=df,
                col_to_stype=col_to_stype,
                col_to_text_embedder_cfg=text_embedder_cfg,
                dataset_name=dataset_name
            ).materialize(path=path, transform=transform)

            data[table_name].tf = dataset.tensor_frame
            col_stats_dict[table_name] = dataset.col_stats

            if table.time_col is not None:
                data[table_name].time = torch.from_numpy(to_unix_time(table.df[table.time_col]))
        else: # table is an 'edge' table
             edge_table_dfs[table_name] = table.df


    # 3. Create edges from both node and edge tables
    for table_name, table in db.table_dict.items():
        if table_domain[table_name] == 'node':
            # Create edges from foreign key relationships in node tables
            for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
                pkey_index = table.df[fkey_name]
                mask = ~pkey_index.isna()
                fkey_index = torch.arange(len(pkey_index))[torch.from_numpy(mask.values)]
                pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
                
                assert (pkey_index < len(db.table_dict[pkey_table_name].df)).all()

                edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
                rev_edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)

                # These edges do not have features from a relation table.
                # PyG allows edges without attributes. We can add a default
                # timestamp if needed, similar to the original 4dbinfer logic.
                data[edge_type].edge_index = sort_edge_index(torch.stack([fkey_index, pkey_index]))
                data[rev_edge_type].edge_index = sort_edge_index(torch.stack([pkey_index, fkey_index]))
                
                # # Optionally add a default timestamp feature
                # if table.time_col is not None:
                #      edge_time = torch.from_numpy(to_unix_time(table.df[table.time_col]))[fkey_index]
                #      data[edge_type].edge_time = edge_time
                #      data[rev_edge_type].edge_time = edge_time


        else: # table is an 'edge' table
            logger.info(f"Processing '{table_name}' as an edge table.")
            df = edge_table_dfs[table_name]
            fkeys = list(table.fkey_col_to_pkey_table.keys())
            fkey1_name, fkey2_name = fkeys[0], fkeys[1]

            src_table_name = table.fkey_col_to_pkey_table[fkey1_name]
            dst_table_name = table.fkey_col_to_pkey_table[fkey2_name]

            src_fkey = df[fkey1_name]
            dst_fkey = df[fkey2_name]

            mask = ~src_fkey.isna() & ~dst_fkey.isna()
            src_fkey = torch.from_numpy(src_fkey[mask].astype(int).values)
            dst_fkey = torch.from_numpy(dst_fkey[mask].astype(int).values)

            assert (src_fkey < len(db.table_dict[src_table_name].df)).all()
            assert (dst_fkey < len(db.table_dict[dst_table_name].df)).all()

            edge_type = (src_table_name, f'rel_{table_name}', dst_table_name)
            rev_edge_type = (dst_table_name, f'rev_rel_{table_name}', src_table_name)

            data[edge_type].edge_index = sort_edge_index(torch.stack([src_fkey, dst_fkey]))
            data[rev_edge_type].edge_index = sort_edge_index(torch.stack([dst_fkey, src_fkey]))

            # data[edge_type].edge_attr = torch.arange(len(src_fkey))

            # Handle edge features for these relation-table-derived edges
            col_to_stype = col_to_stype_dict[table_name]
            remove_pkey_fkey(col_to_stype, table)
            
            if len(col_to_stype) > 0:
                path = (
                    None if cache_dir is None
                    else os.path.join(cache_dir, f'{dataset_name}_r2ne', f"{table_name}_edge.pt")
                )

                dataset = DatasetwithTransform(
                    df=df[mask], # Use masked df to align features with edges
                    col_to_stype=col_to_stype,
                    col_to_text_embedder_cfg=text_embedder_cfg,
                    dataset_name=dataset_name
                ).materialize(path=path, transform=transform)
                
                # Assign edge attributes
                data[edge_type].tf = dataset.tensor_frame
                data[rev_edge_type].tf = dataset.tensor_frame # Use same features for reverse
                col_stats_dict[f'rel_{table_name}'] = dataset.col_stats
                col_stats_dict[f'rev_rel_{table_name}'] = dataset.col_stats
            
            if table.time_col is not None:
                data[edge_type].time = torch.from_numpy(to_unix_time(table.df[table.time_col][mask]))
                data[rev_edge_type].time = torch.from_numpy(to_unix_time(table.df[table.time_col][mask]))
                assert len(data[edge_type].time) == len(data[edge_type].edge_index[0])


    data.validate()
    ## if not edge_attr, set it to 0
    # for edge_type in data.edge_types:
    #     if not hasattr(data[edge_type], 'edge_attr'):
    #         data[edge_type].edge_attr = torch.zeros(data[edge_type].edge_index.shape[1])
    
    return data, col_stats_dict


def filter_nulls(arr):
    """
    Filters out various types of null values from an array-like object.

    This function is designed to handle Python's `None`, NumPy's `np.nan`, 
    and pandas' `pd.NA`. It leverages the power of pandas for robust null detection,
    which is generally safer than checking for each null type individually.

    Args:
    arr: An array-like object (e.g., list, NumPy array, pandas Series) that may
            contain mixed data types and various null representations.

    Returns:
    A new NumPy array containing only the non-null elements.
    """
    # pandas.notna() is a comprehensive function that returns a boolean mask, 
    # with `True` for non-null values and `False` for any null-like value.
    # We convert the input to a pandas Series to reliably use this function.
    series = pd.Series(arr)

    # We use the boolean mask to select only the non-null elements.
    # .to_numpy() ensures the final output is a NumPy array.
    return series[pd.notna(series)].to_numpy()

def generate_adaptive_null(arr):
    """
    Generates a type-appropriate 'null' or empty value based on the array's dtype.

    For example, it returns "" for string arrays, np.nan for float arrays, and 0
    for integer arrays.

    Args:
    arr: An array-like object (e.g., list, NumPy array, pandas Series).

    Returns:
    A null/empty value suitable for the array's data type.
    """
    # Convert to pandas Series to handle various inputs and infer a consistent dtype
    series = pd.Series(arr)
    dtype = series.dtype

    if np.issubdtype(dtype, np.floating):
        return 0.
    elif np.issubdtype(dtype, np.integer):
        return 0
    # For object dtype, pandas often uses it for strings. An empty string is a safe bet.
    elif np.issubdtype(dtype, np.object_) or np.issubdtype(dtype, np.str_):
        return ""
    else:
    # A generic fallback for other types (boolean, datetime, etc.)
        return None

def extract_dummy(rdb_data):
    pk_to_fk_list = defaultdict(list)
    for relationship in rdb_data.metadata.relationships:
        fk_tbl, fk_col = relationship.fk.table, relationship.fk.column
        pk_tbl, pk_col = relationship.pk.table, relationship.pk.column
        pk_to_fk_list[(pk_tbl, pk_col)].append((fk_tbl, fk_col))
    dummy_tbl_to_pk = {}
    for (pk_tbl, pk_col), fk_list in pk_to_fk_list.items():
        if pk_tbl not in rdb_data.tables:
            if pk_tbl in dummy_tbl_to_pk:
                other_pk_col = dummy_tbl_to_pk[pk_tbl]
                raise ValueError(
                    f"Dummy table {pk_tbl} can only have one PK column but got"
                    f" '{pk_col}' and '{other_pk_col}'."
                )
            logger.info(f"Add dummy table '{pk_tbl}' with PK column '{pk_col}'.")
            
            pk_data = np.unique(filter_nulls(np.concatenate([
                rdb_data.tables[fk_tbl][fk_col]
                for fk_tbl, fk_col in fk_list])))
            ## add a null value to the pk data
            # pk_data = np.concatenate([pk_data, [generate_adaptive_null(pk_data)]])
            pk_id = np.arange(len(pk_data))
            rdb_data.tables[pk_tbl] = {pk_col : pk_data, f'{pk_col}ID': pk_id}
            ## change the original column in fk table to fk value instead of original value
            mapping_data_to_id = {data: i for i, data in enumerate(pk_data)}
            for fk_tbl, fk_col in fk_list:
                fk_values = rdb_data.tables[fk_tbl][fk_col]
                adaptive_null = None
                rdb_data.tables[fk_tbl][fk_col] = np.array([mapping_data_to_id.get(x, adaptive_null) for x in fk_values])
            dummy_tbl_to_pk[pk_tbl] = f'{pk_col}ID'
    return rdb_data, dummy_tbl_to_pk

# def convert_4dbinfer_to_rdb(dbb_path):
#     """
#     Description:
#     Convert the 4dbinfer database to the rdb database
#     Parameters:
#     dbb_path: the path to the 4dbinfer database
#     Returns:
#     The rdb database
#     """
#     dbb = load_dbb_dataset_from_cfg_path_no_name(dbb_path)
#     plot_rdb_dataset_schema(dbb, dbb_path)
#     table_dict = {}
#     old_db_meta = dbb.metadata
#     ## convert storage mode into dict for easier query
#     db_meta = {
#         table.name: {
#             column.name: column for column in table.columns
#         } for table in old_db_meta.tables 
#     }
#     rdb_data, dummy_tbl_to_pk = extract_dummy(dbb)
#     for table_name, table in rdb_data.tables.items():
#         ## check the dtype of each column
#         table_df = pd.DataFrame(table)
#         if table_name not in db_meta:
#             ## in these cases, the table is a dummy table
#             primary_key_col = dummy_tbl_to_pk[table_name]
#             fkey_col_to_pkey_table = {}
#             time_col = None
#         else:
#             primary_key_col = ""
#             for col_name in table_df.columns:
#                 if db_meta[table_name][col_name].dtype == 'primary_key':
#                     primary_key_col = col_name
#                     break
#             fkey_col_to_pkey_table = {}
#             for col_name in table_df.columns:
#                 if db_meta[table_name][col_name].dtype == 'foreign_key':
#                     fkey_col_to_pkey_table[col_name] = db_meta[table_name][col_name].link_to.split('.')[0]
#             time_col = None
#             for col_name in table_df.columns:
#                 if db_meta[table_name][col_name].dtype == 'timestamp':
#                     time_col = col_name
#                     break
#         table_dict[table_name] = Table(table_df, pkey_col=primary_key_col, fkey_col_to_pkey_table=fkey_col_to_pkey_table, time_col=time_col)
#     return Database(table_dict)


def get_timestamp(dataset, split, timedelta, num_eval_timestamps):
    db = dataset.get_db(upto_test_timestamp=split != "test")

    if split == "train":
        start = dataset.val_timestamp - timedelta
        end = db.min_timestamp
        freq = -timedelta

    elif split == "val":
        if dataset.val_timestamp + timedelta > db.max_timestamp:
            raise RuntimeError(
                "val timestamp + timedelta is larger than max timestamp! "
                "This would cause val labels to be generated with "
                "insufficient aggregation time."
            )

        start = dataset.val_timestamp
        end = min(
            dataset.val_timestamp
            + timedelta * (num_eval_timestamps - 1),
            dataset.test_timestamp - timedelta,
        )
        freq = timedelta

    elif split == "test":
        if dataset.test_timestamp + timedelta > db.max_timestamp:
            raise RuntimeError(
                "test timestamp + timedelta is larger than max timestamp! "
                "This would cause test labels to be generated with "
                "insufficient aggregation time."
            )

        start = dataset.test_timestamp
        end = min(
            dataset.test_timestamp
            + timedelta * (num_eval_timestamps - 1),
            db.max_timestamp - timedelta,
        )
        freq = timedelta

    timestamps = pd.date_range(start=start, end=end, freq=freq)
    timestamp_df = pd.DataFrame({'timestamp': timestamps})
    return timestamp_df


def convert_schema_to_pyg_data(schema_path):
    pass 






def load_relbench_to_mtask(cfg:dict):
    datasets = {}
    pretrain_data = cfg['pretrain']
    eval_data = cfg['eval']
    available_datasets = get_dataset_names()
    for dataset in pretrain_data['datasets']:       
        dataset_name = dataset['name']
        task_name = dataset['task']
        if dataset.get('path') is not None:
            logger.info(f"Loading custom version dataset {dataset_name} from {dataset['path']}")
            dataset = CustomDataset(dataset['path'])
        else:
            dataset = load_relbench_data(dataset_name)
        task = load_relbench_task(dataset_name, task_name)
        dataset.name = dataset_name
        task.name = task_name
        logger.info(f"Loading pretrain dataset {dataset_name} with task {task_name}")
        if datasets.get(dataset_name) is None:
            datasets[dataset_name] = {}
            datasets[dataset_name]['db'] = dataset
            datasets[dataset_name]['tasks'] = [(task, 'train')]
        else:
            datasets[dataset_name]['tasks'].append((task, 'train'))
    for dataset in eval_data['datasets']:
        dataset_name = dataset['name']
        task_name = dataset['task']
        if dataset.get('path') is not None:
            logger.info(f"Loading custom version dataset {dataset_name} from {dataset['path']}")
            dataset = CustomDataset(dataset['path'])
        else:
            dataset = load_relbench_data(dataset_name)
        task = load_relbench_task(dataset_name, task_name)
        dataset.name = dataset_name
        task.name = task_name
        logger.info(f"Loading eval dataset {dataset_name} with task {task_name}")
        if datasets.get(dataset_name) is None:
            datasets[dataset_name] = {}
            datasets[dataset_name]['db'] = dataset
            datasets[dataset_name]['tasks'] = [(task, 'eval')]
        else:
            datasets[dataset_name]['tasks'].append((task, 'eval'))
    return datasets



def get_atomic_routes(edge_type_list):
    """
        Atomic routes retrieval from edge type list, 
        as shown in https://arxiv.org/abs/2502.06784
        Code is adapted from https://github.com/snap-stanford/RelGNN
    """
    src_to_tuples = defaultdict(list)
    for src, rel, dst in edge_type_list:
        if rel.startswith('f2p'):
            if src == dst:
                src = src + '--' + rel
            src_to_tuples[src].append((src, rel, dst))
    atomic_routes_list = []
    get_rev_edge = lambda edge: (edge[2], 'rev_' + edge[1], edge[0])
    for src, tuples in src_to_tuples.items():
        if '--' in src:
            src = src.split('--')[0]
        if len(tuples) == 1:
            _, rel, dst = tuples[0]
            edge = (src, rel, dst)
            atomic_routes_list.append(('dim-dim',) + edge)
            atomic_routes_list.append(('dim-dim',) + get_rev_edge(edge))
        else:
            for _, rel_q, dst_q in tuples:
                for _, rel_v, dst_v in tuples:
                    if rel_q != rel_v:
                        edge_q = (src, rel_q, dst_q)
                        edge_v = (src, rel_v, dst_v)                   
                        atomic_routes_list.append(('dim-fact-dim',) + edge_q + get_rev_edge(edge_v))
    return atomic_routes_list