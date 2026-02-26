from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import duckdb
import numpy as np
import pandas as pd
import psutil
import torch
import torch_frame
from loguru import logger
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.loader import NeighborLoader
from typing_extensions import deprecated

from data.relbench import get_task_meta
from utils.graph import retrieve_meta_graph_from_constructed_data
from models.embedder import GloveTextEmbedding, SBTextEmbedding
from relbench.base import TaskType
from relbench.modeling.graph import (
    get_link_train_table_input,
    get_node_train_table_input,
    transform_regression_to_binary_classification,
)
from relbench.modeling.loader import SparseTensor
from relbench.modeling.utils import remove_pkey_fkey
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.typing import Metric

from utils.graph import (
    CustomDataset,
    make_pkey_fkey_graph,
    make_pkey_fkey_graph_r2ne_for_heuristics,
    make_pkey_fkey_graph_with_edge,
    return_embed_feature_dict,
)
from dbinfer.datetime_utils import dt2ts
from utils.misc import dict_to_filename, get_formatted_size_kb, lift_and_remove_nested_keys
from utils.type import (
    convert_llm_output_to_stype_proposal,
    get_stype_dict,
    seed_everything,
)

from .relbench import load_relbench_data, load_relbench_task
from .relbench_loader import LinkNeighborLoader
    

def print_memory_usage() -> None:
    """Log the current process RSS in megabytes for visibility during heavy loaders."""
    process = psutil.Process(os.getpid())
    logger.debug(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def _register_loader_spec(loader: Any, factory: Any, data: Any, kwargs: dict[str, Any]) -> None:
    """Attach the constructor hook Lightning needs to re-instantiate loaders."""
    if loader is None:
        return
    loader.__pl_spec__ = {
        "factory": factory,
        "data": data,
        "kwargs": kwargs,
    }

class MultiTabularDataset:
    def __init__(self, database, tasks, cache_dir, pre_transform = None, text_encoder = "glove",
     val_sample_size = 5000, test_sample_size = 5000, train_sample_size: int = -1,
     train_sample_seed: int | None = None, col_to_stype_mode = 'heuristic',
     graph_construct = 'r2n', sample_with_edge_ft = False, embed_device = "cpu", autog=False, autocomplete_full=True, text_pca_dim: int | None = None, normalize_target: bool = False, rfm = False, transductive = True):
        """
            Wrapper class for an RDB dataset with multiple tasks
        """
        self.database = database
        self.tasks = tasks
        self._autocomplete_task = next((task for task in self.tasks if getattr(task, "auto", False)), None)
        self.has_autocomplete = self._autocomplete_task is not None
        self.task_name_to_id_dict = {task.name: i for i, task in enumerate(self.tasks)}
        self.cache_dir = cache_dir
        ## this is an important update which can makes stypes more easily altered by users
        ## (the stype heuristic doesn't work well sometimes)
        self.stype_cache_dir = 'stypes'
        self.pre_transform = pre_transform
        self.text_encoder = text_encoder
        self.val_sample_size = val_sample_size
        self.test_sample_size = test_sample_size
        self.train_sample_size = train_sample_size
        self.train_sample_seed = 42 if train_sample_seed is None else train_sample_seed
        # legacy attribute used by deprecated APIs
        self.sample_size = train_sample_size
        self.col_to_stype_mode = col_to_stype_mode
        self.graph_construct = graph_construct
        self.sample_with_edge_ft = sample_with_edge_ft
        self.embed_device = embed_device
        self.autog = autog
        self.autocomplete_full = autocomplete_full or self.has_autocomplete
        self.text_pca_dim = text_pca_dim
        self.normalize_target = normalize_target
        self.scaler_paths = {}  # Store scaler paths for each task
        ## initialize col_to_stype_dict
        self.col_to_stype = get_stype_dict(self.database, self.stype_cache_dir, 
                                               f'{self.database.name}_autog' if self.autog else self.database.name)
        self.rfm = rfm
        self.transductive = transductive
        self._train_sample_indices: dict[int, np.ndarray] = {}
    # ------------------------------------------------------------------
    # Lightweight metadata accessors
    # ------------------------------------------------------------------
    def get_col_stats_dict(self):
        """Return cached column statistics produced during database materialization."""
        _, col_stats_dict, _ = self.get_db()
        return col_stats_dict

    def get_node_to_col_names_dict(self):
        """Map each node type to its feature column names."""
        data, _, _ = self.get_db()
        return {node_type: data[node_type].tf.col_names_dict for node_type in data.node_types}

    def get_time_types(self):
        """List node types that contain a `time` feature."""
        data, _, _ = self.get_db()
        return [node_type for node_type in data.node_types if "time" in data[node_type]]
    
    def get_num_nodes_dict(self):
        """Count nodes per node type in the materialized graph."""
        data, _, _ = self.get_db()
        return {node_type: data[node_type].num_nodes for node_type in data.node_types}
    
    def get_node_edge_types(self):
        """Return the heterogenous node and edge type tuples."""
        data, _, _ = self.get_db()
        return data.node_types, data.edge_types
    
    def get_data_metadata(self):
        """Expose PyG metadata tuple consumed by downstream models."""
        data, _, _ = self.get_db()
        return data.metadata()
        
    def get_task_meta(self, task_idx, channels):
        """Build task-level metadata, including link-pred settings when needed."""
        task = self.tasks[task_idx]
        if task.task_type == TaskType.LINK_PREDICTION or task.task_type == "link_prediction":
            link_args = SimpleNamespace(
                out_channels = channels,
                loss_fn = binary_cross_entropy_with_logits,
                tune_metric = "link_prediction_map",
                higher_is_better = True
            )
        else:
            link_args = None
        train_table = self._maybe_sample_train_table(task.get_table('train'), task_idx)
        task_meta = get_task_meta(task, link_args, train_table=train_table, transductive=self.transductive)
        return task_meta

    def get_scaler(self, task_idx):
        """Load and return the scaler for a task if it exists."""
        if task_idx in self.scaler_paths:
            scaler_path = self.scaler_paths[task_idx]
            if os.path.exists(scaler_path):
                import joblib
                return joblib.load(scaler_path)
        return None
    
    def get_col_to_stype_dict(self, dfs_mode = False):
        """Infer semantic column types, optionally via heuristics or LLM proposals."""
        if self.col_to_stype_mode == 'heuristic':
            col_to_stype_dict = get_stype_dict(self.database, self.stype_cache_dir, 
                                               f'{self.database.name}_autog' if self.autog else self.database.name)
        elif self.col_to_stype_mode == 'llm':
            llm_type_path = os.path.join('output', 'data', self.database.name, 'type.txt')
            assert os.path.exists(llm_type_path)
            col_to_stype_dict = convert_llm_output_to_stype_proposal(llm_type_path)
        else:
            raise ValueError(f"Unsupported col_to_stype_mode: {self.col_to_stype_mode}")
        ## augment key column
        db = self.database.get_db()
        for table_name, table in db.table_dict.items():
            if table.pkey_col is not None and table.pkey_col != "":
                col_to_stype_dict[table_name][table.pkey_col] = "primary_key"
            if table.fkey_col_to_pkey_table is not None and table.fkey_col_to_pkey_table != {}:
                for col_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
                    if col_name is not None and col_name != "":
                        col_to_stype_dict[table_name][col_name] = "foreign_key"
        return col_to_stype_dict

    # ------------------------------------------------------------------
    # Time feature engineering utilities
    # ------------------------------------------------------------------
    @staticmethod
    def featurize_timestamp(df, timestamp_col='timestamp'):
        """
        Extracts engineered time features from a timestamp column for TabPFN-TS.

        Args:
            df (pd.DataFrame): Input DataFrame with a timestamp column.
            timestamp_col (str): Name of the timestamp column.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        # Ensure timestamp column is datetime
        ts = pd.to_datetime(df[timestamp_col])
        
        # Year (integer, not cyclical)
        year = ts.dt.year

        # Hour of day (cyclical)
        hour = ts.dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Day of week (cyclical, Monday=0)
        dow = ts.dt.dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # Day of month (cyclical)
        dom = ts.dt.day
        dom_sin = np.sin(2 * np.pi * (dom - 1) / 31)
        dom_cos = np.cos(2 * np.pi * (dom - 1) / 31)

        # Day of year (cyclical)
        doy = ts.dt.dayofyear
        doy_sin = np.sin(2 * np.pi * (doy - 1) / 366)
        doy_cos = np.cos(2 * np.pi * (doy - 1) / 366)

        # Week of year (cyclical)
        woy = ts.dt.isocalendar().week.astype(int)
        woy_sin = np.sin(2 * np.pi * (woy - 1) / 53)
        woy_cos = np.cos(2 * np.pi * (woy - 1) / 53)

        # Month of year (cyclical)
        moy = ts.dt.month
        moy_sin = np.sin(2 * np.pi * (moy - 1) / 12)
        moy_cos = np.cos(2 * np.pi * (moy - 1) / 12)

        # Running index
        index = np.arange(len(df))

        # Assemble features
        features = {
            'year': year,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'dow_sin': dow_sin,
            'dow_cos': dow_cos,
            'dom_sin': dom_sin,
            'dom_cos': dom_cos,
            'doy_sin': doy_sin,
            'doy_cos': doy_cos,
            'woy_sin': woy_sin,
            'woy_cos': woy_cos,
            'moy_sin': moy_sin,
            'moy_cos': moy_cos
        }

        return features

    @lru_cache(maxsize=1)
    def get_embed_feature_dict(self, drop_text = True, normalize_numeric = True):
        """
        This function is mainly used for the compability with DFS and relbench
        """
        f_dict, _ =  return_embed_feature_dict(self.database.get_db(), self.get_col_to_stype_dict(), 
                TextEmbedderConfig(
                    text_embedder=self.select_text_encoder()(device=self.embed_device), batch_size=4096
                ), self.cache_dir, self.pre_transform, f'{self.database.name}_autog' if self.autog 
                else self.database.name, normalize_numeric=normalize_numeric, 
                drop_text=drop_text)
        db_tables = deepcopy(self.database.get_db().table_dict)
        col_to_stype_dict = deepcopy(self.get_col_to_stype_dict())
        for table_name, table in db_tables.items():
            table_df = pd.DataFrame()
            for col_name in table.df.columns:
                try:
                    if col_name in col_to_stype_dict[table_name] and \
                        str(col_to_stype_dict[table_name][col_name]) == "timestamp":
                        # here we need to further differ datetime and timestamp formats
                        # for example, 'time' in rel-f1 is a timestamp instead of a datetime object
                        try:
                            table.df[col_name].to_numpy().astype('datetime64[ns]')
                            table_df[col_name] = table.df[col_name]
                            col_to_stype_dict[table_name][col_name] = "datetime"
                            # table_df[f'feature_{col_name}'] = dt2ts(table.df[col_name]).astype('float32')
                            # col_to_stype_dict[table_name][f'feature_{col_name}'] = "numerical"
                            add_features = self.featurize_timestamp(table.df, col_name)
                            for k, v in add_features.items():
                                table_df[f'add_feature_{col_name}_{k}'] = v
                                col_to_stype_dict[table_name][f'add_feature_{col_name}_{k}'] = "numerical"
                        except: 
                            # table_df[col_name] = table.df[col_name]
                            # col_to_stype_dict[table_name][col_name] = "timestamp"
                            table_df[f'feature_{col_name}'] = dt2ts(table.df[col_name]).astype('float32')
                            col_to_stype_dict[table_name][f'feature_{col_name}'] = "numerical"
                        ## generate fine-grained time features 
                    elif col_name in f_dict[table_name]._col_to_stype_idx:
                        if str(col_to_stype_dict[table_name][col_name]) != "text_embedded":
                            table_df[col_name] = f_dict[table_name].get_col_feat(col_name).flatten().numpy()
                        else:
                            # check the ratio of unique values 
                            unique_ratio = len(table.df[col_name].unique()) / len(table.df[col_name])
                            # if smaller than 0.3, we consider it as a categorical column
                            if unique_ratio < 0.3:
                                col_to_stype_dict[table_name][col_name] = "categorical" 
                                table_df[col_name] = table.df[col_name].astype('category')
                            else:
                                if not drop_text:
                                    col_to_stype_dict[table_name][col_name] = "text_embedded"
                                    table_df[col_name] = table.df[col_name]
                    elif col_name == table.pkey_col or col_name in table.fkey_col_to_pkey_table:
                        table_df[col_name] = table.df[col_name]
                except:
                    continue
            db_tables[table_name].df = table_df
        return db_tables, col_to_stype_dict

    # ------------------------------------------------------------------
    # Data materialisation utilities
    # ------------------------------------------------------------------
    def create_duckdb_env(self, persist_path: str = ':memory:', task_idx: int = 0):
        """Project the relational database into DuckDB; useful for ad-hoc queries."""
        # 1) Open (or create) a DuckDB connection
        conn = duckdb.connect(database=persist_path)

        table_names = []
        # 2) Materialize each table.df into DuckDB under its table_name
        db = self.database.get_db()
        
        # First pass: Create tables without constraints
        for table_name, table in db.table_dict.items():
            df = table.df
            conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM df
            """)
            table_names.append(table_name)
        
        # Second pass: Recreate tables with constraints
        for table_name, table in db.table_dict.items():
            df = table.df
            table_obj = db.table_dict[table_name]
            pk_col = table_obj.pkey_col
            fk_cols = table_obj.fkey_col_to_pkey_table
            
            print(f"Table: {table_name}")
            print(f"Primary Key: {pk_col}")
            print(f"Foreign Keys: {fk_cols}")
            
            # Build table-level constraints
            constraints = []
            
            # Add primary key constraint
            if pk_col:
                constraints.append(f"PRIMARY KEY ({pk_col})")
            
            # Add foreign key constraints
            for fk_col, ref_table in fk_cols.items():
                ref_table_obj = db.table_dict.get(ref_table)
                if ref_table_obj and ref_table_obj.pkey_col:
                    ref_pk_col = ref_table_obj.pkey_col
                    constraints.append(f"FOREIGN KEY ({fk_col}) REFERENCES {ref_table}({ref_pk_col})")
            
            # Create table (DuckDB has limited constraint support, so we'll create without constraints)
            try:
                # Create table without constraints first
                conn.execute(f"""
                    CREATE OR REPLACE TABLE {table_name} AS
                    SELECT * FROM df
                """)
                
                print(f"Created table {table_name}")
                if pk_col:
                    print(f"  - Primary key column: {pk_col} (constraint not enforced)")
                for fk_col, ref_table in fk_cols.items():
                    ref_table_obj = db.table_dict.get(ref_table)
                    if ref_table_obj and ref_table_obj.pkey_col:
                        print(f"  - Foreign key column: {fk_col} -> {ref_table}.{ref_table_obj.pkey_col} (constraint not enforced)")
                        
            except Exception as e:
                print(f"Could not create table {table_name}: {e}")
        # 3) Return the live connection
        sampled_train_table = self._maybe_sample_train_table(self.tasks[task_idx].get_table('train'), task_idx)
        train_task_table = sampled_train_table.df
        conn.execute(f"""
            CREATE OR REPLACE TABLE train_task_table AS
            SELECT * FROM train_task_table
        """)
        val_task_table = self.tasks[task_idx].get_table('val').df
        conn.execute(f"""
            CREATE OR REPLACE TABLE val_task_table AS
            SELECT * FROM val_task_table
        """)
        test_task_table = self.tasks[task_idx].get_table('test').df
        conn.execute(f"""
            CREATE OR REPLACE TABLE test_task_table AS
            SELECT * FROM test_task_table
        """)
        return conn, table_names

    
    @lru_cache(maxsize=3)
    def get_db(self, task_idx = 0):
        """Construct the cached graph representation for the current database."""
        col_to_stype_dict = self.col_to_stype
        autocomplete_task = self._autocomplete_task
        if autocomplete_task is not None:
            # Autocomplete tasks operate on the full database snapshot (no test cutoff)
            db_instance = self.database.get_db(upto_test_timestamp=False if self.autocomplete_full else True)
        else:
            db_instance = self.database.get_db()

        if autocomplete_task is not None:
            new_col_to_stype_dict = deepcopy(col_to_stype_dict)
            entity_table = autocomplete_task.entity_table
            target_column = autocomplete_task.target_col
            # Remove target column features if present to avoid leakage
            if (
                entity_table in new_col_to_stype_dict
                and target_column in new_col_to_stype_dict[entity_table]
            ):
                del new_col_to_stype_dict[entity_table][target_column]
            for table_name, column_name in getattr(autocomplete_task, "remove_columns", []):
                if table_name in new_col_to_stype_dict and column_name in new_col_to_stype_dict[table_name]:
                    del new_col_to_stype_dict[table_name][column_name]
            data, col_stats_dict, new_col_to_stype_dict = make_pkey_fkey_graph(
                db_instance,
                col_to_stype_dict=new_col_to_stype_dict,
                text_embedder_cfg=TextEmbedderConfig(
                    text_embedder=self.select_text_encoder()(device=self.embed_device), batch_size=4096
                ),
                transform=self.pre_transform,
                cache_dir=self.cache_dir,
                dataset_name=f'{self.database.name}/{autocomplete_task.name}'
            )
            self.col_to_stype = new_col_to_stype_dict
            return data, col_stats_dict, new_col_to_stype_dict
        if self.graph_construct == 'r2n':
            logger.info(f"Constructing r2n graph for {self.database.name}")
            data, col_stats_dict, new_col_to_stype_dict = make_pkey_fkey_graph(
                db_instance,
                col_to_stype_dict=col_to_stype_dict,
                text_embedder_cfg=TextEmbedderConfig(
                    text_embedder=self.select_text_encoder()(device=self.embed_device), batch_size=4096, encoder_name=self.text_encoder
                ),
                transform=self.pre_transform,
                cache_dir=self.cache_dir,
                dataset_name=f'{self.database.name}_autog' if self.autog else self.database.name
            )
            self.col_to_stype = new_col_to_stype_dict
        elif self.graph_construct == 'r2ne':
            logger.info(f"Constructing r2ne graph for {self.database.name}")
            data, col_stats_dict = make_pkey_fkey_graph_with_edge(
                db_instance,
                col_to_stype_dict=col_to_stype_dict,
                text_embedder_cfg=TextEmbedderConfig(
                    text_embedder=self.select_text_encoder()(device=self.embed_device), batch_size=4096
                ),
                transform=self.pre_transform,
                cache_dir=self.cache_dir,
                dataset_name=f'{self.database.name}_autog' if self.autog else self.database.name
            )
        elif self.graph_construct == 'r2ne_for_heuristics':
            logger.info(f"Constructing r2ne_for_heuristics graph for {self.database.name}")
            ## task_idx is only needed for this
            task = self.tasks[task_idx]
            data, col_stats_dict = make_pkey_fkey_graph_r2ne_for_heuristics(
                db_instance,
                col_to_stype_dict=col_to_stype_dict,
                text_embedder_cfg=None,  # constant features don't need text embedding
                transform=self.pre_transform,
                cache_dir=self.cache_dir,
                dataset_name=f'{self.database.name}_autog' if self.autog else self.database.name,
                entity_table=task.entity_table,
                skip_time=True,  # homophily only needs graph structure
            )
        else:
            raise ValueError(f"Unsupported graph_construct: {self.graph_construct}")
        
        return data, col_stats_dict, col_to_stype_dict

    # ------------------------------------------------------------------
    # Tabular sampling helpers
    # ------------------------------------------------------------------
    def construct_compressed_task_table(self, table, idx, sample_train = -1):
        """Optionally subsample validation rows and reuse cached parquet artifacts."""
        # Apply sampling if specified
        # table = task.get_table('val', mask_input_cols=False)
        if os.path.exists(f"{self.cache_dir}/{self.database.name}/{self.tasks[idx].name}_compressed_table_{self.val_sample_size}_new.parquet"):
            needed_df = pd.read_parquet(f"{self.cache_dir}/{self.database.name}/{self.tasks[idx].name}_compressed_table_{self.val_sample_size}_new.parquet")
            table.df = needed_df
            return table
        else:
            if self.val_sample_size > 0 and self.val_sample_size < len(table.df):
                logger.debug(f"Sampling {self.val_sample_size} rows for validation")
                sampled_idx = np.random.permutation(len(table.df))[: self.val_sample_size]
                table.df = table.df.iloc[sampled_idx]
                table.df = table.df.reset_index(drop=True)
                table.df.to_parquet(f"{self.cache_dir}/{self.database.name}/{self.tasks[idx].name}_compressed_table_{self.val_sample_size}_new.parquet")
        return table

    def _maybe_sample_train_table(self, table, task_idx: int):
        """Subsample the training table deterministically when requested."""
        if self.train_sample_size is None or self.train_sample_size <= 0:
            return table
        num_rows = len(table)
        if num_rows <= 0 or self.train_sample_size >= num_rows:
            return table
        subset_size = min(self.train_sample_size, num_rows)
        cached_idx = self._train_sample_indices.get(task_idx)
        if (
            cached_idx is None
            or len(cached_idx) != subset_size
            or (len(cached_idx) > 0 and cached_idx.max() >= num_rows)
        ):
            base_seed = self.train_sample_seed if self.train_sample_seed is not None else 42
            rng = np.random.default_rng(base_seed + task_idx)
            cached_idx = np.sort(rng.choice(num_rows, size=subset_size, replace=False))
            self._train_sample_indices[task_idx] = cached_idx
        table.df = table.df.iloc[cached_idx].reset_index(drop=True)
        return table

    def clear_cached_graph(self) -> None:
        """Clear the cached graph representation produced by get_db."""
        cache_clear = getattr(self.get_db, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
    
    def get_transformed_regression_bc_table(self, task_idx, task_table, model_config):
        """Convert regression targets into binary classification dataset for experimentation."""
        model_config = lift_and_remove_nested_keys(model_config)
        task = self.tasks[task_idx]
        loader_args = SimpleNamespace(
                batch_size=model_config["batch_size"],
                num_neighbors=model_config["num_neighbors"],
                num_workers=model_config["num_workers"],
                temporal_strategy=model_config["temporal"],
                num_layers=model_config["num_layers"],
                eval_batch_size=model_config["eval_batch_size"]
            )
        table = task_table
        # table = self.construct_compressed_task_table(table, task_idx)
        sampled_idx = np.random.permutation(len(table))[: self.val_sample_size]
        table = table.iloc[sampled_idx]
        table = table.reset_index()
        table_input = transform_regression_to_binary_classification(table=table, task=task)
        table_input_nodes = table_input.nodes
        table_input_time = table_input.time
        table_input_transform = table_input.transform
        data, col_stats_dict, col_to_stype_dict = self.get_db()
        val_loader = NeighborLoader(
            data,
            num_neighbors=[int(loader_args.num_neighbors / 2**i) for i in range(loader_args.num_layers)],
            time_attr="time",
            input_nodes=table_input_nodes,
            input_time=table_input_time,
            transform=table_input_transform,
            batch_size=loader_args.batch_size,
            subgraph_type="bidirectional",
            temporal_strategy=loader_args.temporal_strategy,
            shuffle=False,
            num_workers=loader_args.num_workers,
            persistent_workers=loader_args.num_workers > 0,
        )
        return val_loader


    # ------------------------------------------------------------------
    # Loader / task initialisation
    # ------------------------------------------------------------------
    def initialize_data(
        self,
        task_idx: int,
        rank: int,
        world_size: int,
        loader_type: str,
        model_config: dict,
        preserve_test_targets: bool = False,
        enable_lightning_distributed: bool = False,  # noqa: ARG002 - kept for backward compatibility
        distributed_eval: bool = False,
    ):
        """Build data loaders and task metadata for a given task index."""
        # Normalize Hydra/OmegaConf input into a plain dict before accessing fields.
        model_config = lift_and_remove_nested_keys(model_config)
        dataset_name = f'{self.database.name}_autog' if self.autog else self.database.name
        data_cache_dir = f"{self.cache_dir}/{dataset_name}/materialized"
        if self.pre_transform is not None:
            self.data_cache_dir = f"{data_cache_dir}_{self.pre_transform}"
        else:
            self.data_cache_dir = data_cache_dir
        # Surface memory usage for debugging large graphs.
        print_memory_usage()

        task = self.tasks[task_idx]

        # ------------------------------------------------------------------
        # Pre-fetch raw task tables *before* graph preprocessing.
        #
        # Task.get_table() ultimately calls into the RelBench dataset, which
        # in turn uses dataset.get_db(upto_test_timestamp=split != "test").
        # If we were to preprocess the graph first, those calls would see the
        # transformed database and the labels would be computed on the
        # modified view. By materialising the tables upfront we guarantee that
        # label generation always relies on the untouched database snapshot.
        # ------------------------------------------------------------------
        train_table = self._maybe_sample_train_table(task.get_table('train'), task_idx)
        val_table = task.get_table('val')
        test_table_full = task.get_table('test', mask_input_cols=False)
        # For most node tasks we still mask input columns during loader
        # construction, but certain evaluation paths request the unmasked
        # version explicitly. Fetch it lazily only when required to avoid an
        # extra round-trip to the dataset.
        test_table_masked = test_table_full if preserve_test_targets else task.get_table('test')

        task_meta = self.get_task_meta(task_idx, model_config["channels"])

        # With labels cached, build the (potentially preprocessed) graph used
        # for feature extraction and sampling.
        data, col_stats_dict, col_to_stype_dict = self.get_db()
        self.col_meta = {node_type: data[node_type].tf.col_names_dict for node_type in data.node_types}

        logger.debug("Memory before generating meta graphs")
        print_memory_usage()
        # self.task_metas.append(task_meta)
        if loader_type == 'node':
            meta_graph = retrieve_meta_graph_from_constructed_data(data, task, 
                                                                    model_config["channels"], 
                                                                    task_type=task_meta['task_type'])
            logger.debug("Memory after generating meta graphs")
            print_memory_usage()
            loader_args = SimpleNamespace(
                batch_size=model_config["batch_size"],
                num_neighbors=model_config["num_neighbors"],
                num_workers=model_config["num_workers"],
                temporal_strategy=model_config["temporal"],
                num_layers=model_config["num_layers"],
                eval_batch_size=model_config["eval_batch_size"]
            )
            loader_dict = {}
            dst_nodes_dict = {}
            num_dst_nodes_dict = {}
            sampled_val_table = None
            ## if not sampled then just None

            ## only support compressing validation set
            if task_meta['task_type'] == 'node':
                if str(task.task_type).lower() == 'regression':
                    clamp_min, clamp_max = np.percentile(train_table.df[task.target_col].to_numpy(), [2, 98])
                    task_meta['clamp_min'] = clamp_min
                    task_meta['clamp_max'] = clamp_max

            # Setup scaler path for regression tasks
            scaler_path = None
            if self.normalize_target and str(task.task_type).lower() == 'regression':
                scaler_dir = os.path.join(self.cache_dir, 'scalers')
                os.makedirs(scaler_dir, exist_ok=True)
                dataset_name = f'{self.database.name}_autog' if self.autog else self.database.name
                scaler_path = os.path.join(scaler_dir, f'{dataset_name}_{task.name}_scaler.pkl')
                self.scaler_paths[task_idx] = scaler_path

            # Derive layer-wise neighborhood fanout (Lightning expects it on the loader).
            neighbor_sizes = [int(loader_args.num_neighbors / 2**i) for i in range(loader_args.num_layers)]
            # Cache the exact table objects we will feed into each split loader.
            table_cache: Dict[str, Any] = {
                "train": train_table,
                "val": val_table,
                "test": test_table_masked,
            }
            for split in ["train", "val", "test"]:
                if split == 'val':
                    table = self.construct_compressed_task_table(table_cache[split], task_idx)
                    sampled_val_table = table
                else:
                    table = table_cache[split]
                if task_meta['task_type'] == 'node':
                    is_train = (split == "train")
                    table_input = get_node_train_table_input(
                        table=table,
                        task=task,
                        normalize_target=self.normalize_target,
                        is_train_scaler=is_train,
                        scaler_path=scaler_path
                    )
                    table_input_nodes = table_input.nodes
                    table_input_time = table_input.time
                    table_input_transform = table_input.transform

                    # Apply distributed sampling for test split when enabled
                    if split == "test" and distributed_eval and world_size > 1:
                        # Partition the test nodes across GPUs
                        num_samples = len(table_input_nodes)
                        samples_per_rank = (num_samples + world_size - 1) // world_size
                        start_idx = rank * samples_per_rank
                        end_idx = min(start_idx + samples_per_rank, num_samples)

                        # Slice the input nodes and time for this rank
                        table_input_nodes = table_input_nodes[start_idx:end_idx]
                        if table_input_time is not None:
                            table_input_time = table_input_time[start_idx:end_idx]

                        logger.info(
                            f"Rank {rank}/{world_size}: Processing test samples "
                            f"[{start_idx}:{end_idx}] out of {num_samples} total"
                        )
                    loader_kwargs = {
                        "num_neighbors": neighbor_sizes,
                        "time_attr": "time",
                        "input_nodes": table_input_nodes,
                        "input_time": table_input_time,
                        "transform": table_input_transform,
                        "batch_size": loader_args.batch_size,
                        "subgraph_type": "bidirectional",
                        "temporal_strategy": loader_args.temporal_strategy,
                        "shuffle": split == "train",
                        "num_workers": loader_args.num_workers,
                        "persistent_workers": loader_args.num_workers > 0,
                    }
                    loader = NeighborLoader(data, **loader_kwargs)
                    _register_loader_spec(loader, NeighborLoader, data, loader_kwargs)
                    loader_dict[split] = loader
                elif task_meta['task_type'] == 'link':
                    table_input = get_link_train_table_input(table, task)
                    dst_nodes_dict[split] = table_input.dst_nodes
                    num_dst_nodes_dict[split] = table_input.num_dst_nodes

                    src_nodes = table_input.src_nodes
                    src_time = table_input.src_time

                    # Apply distributed sampling for test split when enabled
                    if split == "test" and distributed_eval and world_size > 1:
                        # Partition the test source nodes across GPUs
                        num_samples = len(src_nodes)
                        samples_per_rank = (num_samples + world_size - 1) // world_size
                        start_idx = rank * samples_per_rank
                        end_idx = min(start_idx + samples_per_rank, num_samples)

                        # Slice the source nodes and time for this rank
                        src_nodes = src_nodes[start_idx:end_idx]
                        if src_time is not None:
                            src_time = src_time[start_idx:end_idx]

                        logger.info(
                            f"Rank {rank}/{world_size}: Processing test link samples "
                            f"[{start_idx}:{end_idx}] out of {num_samples} total"
                        )

                    loader_kwargs = {
                        "num_neighbors": neighbor_sizes,
                        "time_attr": "time",
                        "input_nodes": src_nodes,
                        "input_time": src_time,
                        "subgraph_type": "bidirectional",
                        "batch_size": loader_args.batch_size,
                        "temporal_strategy": loader_args.temporal_strategy,
                        "shuffle": split == "train",
                        "num_workers": loader_args.num_workers,
                        "persistent_workers": loader_args.num_workers > 0,
                    }
                    loader = NeighborLoader(data, **loader_kwargs)
                    _register_loader_spec(loader, NeighborLoader, data, loader_kwargs)
                    loader_dict[split] = loader
                print_memory_usage()
                    
                # Store additional information for link prediction
            if dst_nodes_dict.get('train', None) is not None:
                train_sparse_tensors = SparseTensor(dst_nodes_dict['train'][1], device='cpu')
            else:
                train_sparse_tensors = None
            return loader_dict, num_dst_nodes_dict, task_meta, meta_graph, train_sparse_tensors, sampled_val_table
        elif loader_type == 'link':
            loader_args = SimpleNamespace(
                batch_size=model_config["batch_size"],
                eval_batch_size=model_config["eval_batch_size"],
                num_neighbors=model_config["num_neighbors"],
                num_workers=model_config["num_workers"],
                temporal_strategy=model_config["temporal"],
                num_layers=model_config["num_layers"],
                share_same_time=model_config["share_same_time"],
                val_timestamp=self.database.val_timestamp,
                test_timestamp=self.database.test_timestamp,
                num_negative_samples=model_config.get("num_negative_samples", 1)
            )
            table_input = get_link_train_table_input(train_table, task)
            sampled_val_table = self.construct_compressed_task_table(val_table, task_idx)
            test_res_task = test_table_full
            train_loader, eval_loaders_dict = self.get_link_loader(data, task,
                                                                   [int(loader_args.num_neighbors / 2**i) for i in range(loader_args.num_layers)],
                                                                   table_input, sampled_val_table, test_res_task,
                                                                   loader_args.share_same_time, loader_args.batch_size,
                                                                   loader_args.eval_batch_size,
                                                                   loader_args.temporal_strategy, loader_args.num_workers,
                                                                   loader_args.val_timestamp, loader_args.test_timestamp,
                                                                   loader_args.num_negative_samples,
                                                                   rank, world_size, distributed_eval)
            loader_dict = {}
            loader_dict['train'] = train_loader
            loader_dict['val'] = eval_loaders_dict['val']
            loader_dict['test'] = eval_loaders_dict['test']
            ## return some dummy values to keep the same interface
            return loader_dict, {}, task_meta, None, None, sampled_val_table
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")


    # ------------------------------------------------------------------
    # Legacy helpers and compatibility shims
    # ------------------------------------------------------------------
    def get_sparse_tensor(self, task_idx, mmap=True):
        """Load cached link-prediction sparse tensors when available."""
        if self.tasks[task_idx].task_type == TaskType.LINK_PREDICTION or self.tasks[task_idx].task_type == "link_prediction":
            sparse_tensor_path = f"{self.cache_dir}/{self.database.name}/{self.tasks[task_idx].name}_sparse_tensor.pt"
            return torch.load(sparse_tensor_path, map_location='cpu', mmap=mmap)
        else:
            return None

    @deprecated("This method has known signature issues and is not currently used")
    def get_graph_loaders(self, task_idx, split, rank, world_size):
        """Instantiate a neighbor loader for heuristics code paths that shard manually."""
        ## if single gpu, set rank to 0
        if not hasattr(self, 'loaders') or self.loaders is None:
            raise RuntimeError("loaders not initialized; call setup_loaders() first")
        loader_config = self.loaders[task_idx][split][rank]
        data, _, _ = self.get_db(task_idx)
        loader = NeighborLoader(**loader_config, data=data)
        return loader

    @deprecated("This method has known signature issues and is not currently used")
    def get_res_task(self, task_idx, split):
        """Load cached evaluation tensors for the requested split."""
        if not hasattr(self, 'res_task_path') or self.res_task_path is None:
            raise RuntimeError("res_task_path not initialized")
        return torch.load(self.res_task_path[task_idx][split], map_location='cpu')

    def get_exp_input_nodes(self, task_idx, split, sample_train = -1):
        """Return canonical input nodes/time pairs used by experiment scripts."""
        task = self.tasks[task_idx]

        # Get the task table for the specified split
        table = self.construct_compressed_task_table(task, task_idx)
        
        # Determine task type and get appropriate input nodes
        if task.task_type == TaskType.LINK_PREDICTION or task.task_type == "link_prediction":
            # For link prediction tasks
            table_input = get_link_train_table_input(table, task)
            input_nodes = table_input.src_nodes
            input_time = table_input.src_time if hasattr(table_input, 'src_time') else None
            input_transform = None
        else:
            # For node classification/regression tasks
            table_input = get_node_train_table_input(table=table, task=task)
            input_nodes = table_input.nodes
            input_time = table_input.time if hasattr(table_input, 'time') else None
            input_transform = table_input.transform if hasattr(table_input, 'transform') else None
        
        return input_nodes, input_time, input_transform
    

    def __repr__(self):
        """Compact representation useful for debugging in notebooks/CLI."""
        return f"MultiTabularDataset(database={self.database.name}, tasks={self.tasks}, cache_dir={self.cache_dir})"

    def select_text_encoder(self):
        """Return the text embedding class configured for this dataset."""
        if self.text_encoder == "glove":
            return GloveTextEmbedding
        elif self.text_encoder == "sb" or self.text_encoder == "llm":
            return SBTextEmbedding
        else:
            raise ValueError(f"Text encoder {self.text_encoder} not found")

    @deprecated("This function is not used by us")
    def initialize_tabular_dataset(self, preprocess_for = 'tabpfn'):
        """Legacy TabPFN-style preprocessing pipeline retained for compatibility."""
        """
            This one is not used by us
        """
        train_table = self.task.get_table('train')
        val_table = self.task.get_table('val')
        test_table = self.task.get_table('test')

        if self.task.task_type == TaskType.LINK_PREDICTION:
            self.target_col_name = "link_pred_baseline_target_column_name"
        else:
            self.target_col_name = self.task.target_col

        dfs: Dict[str, pd.DataFrame] = {}
        target_dfs: Dict[str, pd.DateOffset] = {}
        db = self.database.get_db()
        if self.task.task_type == TaskType.LINK_PREDICTION:
            self.src_entity_table = db.table_dict[self.task.src_entity_table]
            self.dst_entity_table = db.table_dict[self.task.dst_entity_table]
            self.src_entity_df = self.src_entity_table.df
            self.dst_entity_df = self.dst_entity_table.df
        else:
            self.src_entity_table = self.dst_entity_table = db.table_dict[self.task.entity_table]
            self.src_entity_df = self.dst_entity_df = self.src_entity_table.df
        col_to_stype_dict = get_stype_dict(self.database, self.stype_cache_dir, self.database.name)
        self.col_to_stype = col_to_stype_dict
        self.col_to_stype = self.col_to_stype[self.task.entity_table]
        if preprocess_for == 'tabpfn':
            for key, val in self.col_to_stype.items():
                if val == torch_frame.text_embedded:
                    self.col_to_stype[key] = torch_frame.categorical
        remove_pkey_fkey(self.col_to_stype, self.src_entity_table)

        if self.task.task_type != TaskType.LINK_PREDICTION:
            if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
                self.col_to_stype[self.task.target_col] = torch_frame.categorical
            elif self.task.task_type == TaskType.REGRESSION:
                self.col_to_stype[self.task.target_col] = torch_frame.numerical
            elif self.task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                self.col_to_stype[self.task.target_col] = torch_frame.embedding
            else:
                raise ValueError(f"Unsupported task type called {self.task.task_type}")

            # randomly subsample in case training data size is too large.
            if self.sample_size > 0 and self.sample_size < len(train_table):
                sampled_idx = np.random.permutation(len(train_table))[: self.sample_size]
                train_table.df = train_table.df.iloc[sampled_idx]

            for split, table in [
                ("train", train_table),
                ("val", val_table),
                ("test", test_table),
            ]:
                left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
                entity_df = self.src_entity_df.astype({self.src_entity_table.pkey_col: table.df[left_entity].dtype})
                dfs[split] = table.df.merge(
                    entity_df,
                    how="left",
                    left_on=left_entity,
                    right_on=self.src_entity_table.pkey_col,
                )

            train_dataset = torch_frame.data.Dataset(
                df=dfs["train"],
                col_to_stype=self.col_to_stype,
                target_col=self.task.target_col,
                col_to_text_embedder_cfg=TextEmbedderConfig(
                    text_embedder=GloveTextEmbedding(device=self.embed_device),
                    batch_size=256,
                ),
            )
            path = Path(
                f"{self.cache_dir}/{self.database.name}/tasks/{self.task}/materialized/node_train_{self.sample_size}.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            train_dataset = train_dataset.materialize(path=path)

            self.tf_train = train_dataset.tensor_frame
            self.tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
            self.tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

            if self.task.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
            ]:
                self.tune_metric = Metric.ROCAUC
            elif self.task.task_type == TaskType.REGRESSION or self.task.task_type == "regression":
                self.tune_metric = Metric.MAE
            else:
                raise ValueError(f"Task task type is unsupported {self.task.task_type}")
        else:
            raise ValueError(f"Task task type is unsupported {self.task.task_type}")
    
    def get_link_loader(self, data, task, num_neighbors,
                        train_table_input,
                        val_table_input,
                        test_table_input,
                        share_same_time,
                        batch_size,
                        eval_batch_size,
                        temporal_strategy,
                        num_workers,
                        val_timestamp,
                        test_timestamp,
                        num_negative_samples,
                        rank=0,
                        world_size=1,
                        distributed_eval=False):
        """Construct train/val/test link loaders with optional time alignment."""
        train_kwargs = {
            "num_neighbors": num_neighbors,
            "time_attr": "time",
            "src_nodes": train_table_input.src_nodes,
            "dst_nodes": train_table_input.dst_nodes,
            "num_dst_nodes": train_table_input.num_dst_nodes,
            "src_time": train_table_input.src_time,
            "share_same_time": share_same_time,
            "batch_size": batch_size,
            "temporal_strategy": temporal_strategy,
            "shuffle": not share_same_time,
            "num_workers": num_workers,
            "num_negative_samples": num_negative_samples,
        }
        train_loader = LinkNeighborLoader(data=data, **train_kwargs)
        _register_loader_spec(train_loader, LinkNeighborLoader, data, train_kwargs)

        eval_loaders_dict: Dict[str, Tuple[NeighborLoader, NeighborLoader]] = {}
        for split in ["val", "test"]:
            timestamp = val_timestamp if split == "val" else test_timestamp
            seed_time = int(timestamp.timestamp())
            target_table = val_table_input if split == "val" else test_table_input
            src_node_indices = torch.from_numpy(target_table.df[task.src_entity_col].values)

            # Apply distributed sampling for test split when enabled
            if split == "test" and distributed_eval and world_size > 1:
                # Partition the test source node indices across GPUs
                num_samples = len(src_node_indices)
                samples_per_rank = (num_samples + world_size - 1) // world_size
                start_idx = rank * samples_per_rank
                end_idx = min(start_idx + samples_per_rank, num_samples)

                # Slice the source node indices for this rank
                src_node_indices = src_node_indices[start_idx:end_idx]

                logger.info(
                    f"Rank {rank}/{world_size}: Processing test link loader samples "
                    f"[{start_idx}:{end_idx}] out of {num_samples} total"
                )

            src_kwargs = {
                "num_neighbors": num_neighbors,
                "time_attr": "time",
                "input_nodes": (task.src_entity_table, src_node_indices),
                "input_time": torch.full(
                    size=(len(src_node_indices),), fill_value=seed_time, dtype=torch.long
                ),
                "batch_size": eval_batch_size,
                "shuffle": False,
                "num_workers": num_workers,
            }
            dst_kwargs = {
                "num_neighbors": num_neighbors,
                "time_attr": "time",
                "input_nodes": task.dst_entity_table,
                "input_time": torch.full(
                    size=(task.num_dst_nodes,), fill_value=seed_time, dtype=torch.long
                ),
                "batch_size": eval_batch_size,
                "shuffle": False,
                "num_workers": num_workers,
            }
            src_loader = NeighborLoader(data, **src_kwargs)
            dst_loader = NeighborLoader(data, **dst_kwargs)
            _register_loader_spec(src_loader, NeighborLoader, data, src_kwargs)
            _register_loader_spec(dst_loader, NeighborLoader, data, dst_kwargs)
            eval_loaders_dict[split] = (src_loader, dst_loader)
        
        return train_loader, eval_loaders_dict

    




def prepare_transductive_rdb_datasets(data_configs:dict, basic_config: dict,
                     stype_proposal = 'heuristic',
                     graph_construct = 'r2n',
                     sample_with_edge_ft = False,
                     autocomplete_full = True,
                     text_pca_dim: int | None = None):
    """
    Prepare datasets and configurations before spawning training processes.
    Compared to prepare_datasets, this function is used for single RDB dataset.
    It will prepare all data for all tasks in the dataset.
    Args:
        data_config: a dict with db name as key and task list as value
        basic_config: the basic config file
        wandb_project: the name of the wandb project
        stype_proposal: the method to propose the stype of the columns
        graph_construct: the method to construct the graph
        sample_with_edge_ft: whether to sample with edge features
        loader_type: the type of loader to use, node or link
    Returns:
        dbs: a list of MultiTabularDataset objects
    """
    cache_dir = basic_config["cache_dir"]
    seed_everything(basic_config["seed"])
    
    # Create cache directory
    pre_train_name = dict_to_filename(data_configs)
    full_name = f"{pre_train_name}"[:35]
    os.makedirs(f"{cache_dir}/{full_name}", exist_ok=True)

    train_sample_size = basic_config.get("train_sample_size", -1)
    train_sample_seed = basic_config.get("train_sample_seed", basic_config.get("seed", 42))
    dbs = []
    for data_config in data_configs["dbs"]:
    # Prepare datasets
        dataset_name = data_config["db_name"]
        task_lists = data_config["task_name"]
        logger.debug(f"Loading Dataset: {dataset_name}")
        if data_config.get("path", None):
            db = CustomDataset(data_config["path"])
            db.name = dataset_name
            autog = True
        else:
            db = load_relbench_data(dataset_name)
            autog = False
        tasks = [load_relbench_task(dataset_name, task_name) for task_name in task_lists]
        dataset = MultiTabularDataset(
            db, 
            tasks,
            cache_dir, 
            pre_transform=basic_config.get("feature_transform", None),
            text_encoder=basic_config["text_encoder"],
            val_sample_size = basic_config.get("val_sample_size", -1), 
            test_sample_size = basic_config.get("test_sample_size", -1),
            train_sample_size=train_sample_size,
            train_sample_seed=train_sample_seed,
            col_to_stype_mode=stype_proposal, 
            graph_construct=graph_construct,
            sample_with_edge_ft=sample_with_edge_ft,
            embed_device=basic_config.get("embed_device", "cpu"),
            autog=autog,
            autocomplete_full=autocomplete_full,
            text_pca_dim=text_pca_dim
        )        
        logger.debug(f"Dataset size: {get_formatted_size_kb(dataset)}")
        dbs.append(dataset)
    return dbs


def prepare_transductive_rdb_datasets_save_memory(data_configs:dict, basic_config: dict, 
                     stype_proposal = 'heuristic',
                     graph_construct = 'r2n',
                     sample_with_edge_ft = False,
                     autocomplete_full = True,
                     text_pca_dim: int | None = None,
                     db_split_key: str = "dbs", 
                     rfm = False,
                     transductive = True,
                     normalize_target: bool = False):
    """
    Prepare datasets and configurations before spawning training processes.
    Compared to prepare_datasets, this function is used for single RDB dataset.
    It will prepare all data for all tasks in the dataset.
    Args:
        data_config: a dict with db name as key and task list as value
        basic_config: the basic config file
        wandb_project: the name of the wandb project
        stype_proposal: the method to propose the stype of the columns
        graph_construct: the method to construct the graph
        sample_with_edge_ft: whether to sample with edge features
        loader_type: the type of loader to use, node or link
    Returns:
        dbs: a list of MultiTabularDataset objects
    """
    cache_dir = basic_config["cache_dir"]
    seed_everything(basic_config["seed"])
    
    # Create cache directory
    pre_train_name = dict_to_filename(data_configs)
    full_name = f"{pre_train_name}"[:35]
    os.makedirs(f"{cache_dir}/{full_name}", exist_ok=True)

    if db_split_key not in data_configs:
        raise ValueError(f"'{db_split_key}' not found in data_configs")

    train_sample_size = basic_config.get("train_sample_size", -1)
    train_sample_seed = basic_config.get("train_sample_seed", basic_config.get("seed", 42))
    for data_config in data_configs[db_split_key]:
    # Prepare datasets
        dataset_name = data_config["db_name"]
        task_lists = data_config["task_name"]
        logger.debug(f"Loading Dataset: {dataset_name}")
        if data_config.get("path", None):
            db = CustomDataset(data_config["path"])
            db.name = dataset_name
            autog = True
        else:
            db = load_relbench_data(dataset_name)
            autog = False
        tasks = [load_relbench_task(dataset_name, task_name) for task_name in task_lists]
        dataset = MultiTabularDataset(
            db, 
            tasks,
            cache_dir, 
            pre_transform=basic_config.get("feature_transform", None),
            text_encoder=basic_config["text_encoder"],
            val_sample_size = basic_config.get("val_sample_size", -1), 
            test_sample_size = basic_config.get("test_sample_size", -1),
            train_sample_size=train_sample_size,
            train_sample_seed=train_sample_seed,
            col_to_stype_mode=stype_proposal, 
            graph_construct=graph_construct,
            sample_with_edge_ft=sample_with_edge_ft,
            embed_device=basic_config.get("embed_device", "cpu"),
            autog=autog,
            autocomplete_full=autocomplete_full,
            text_pca_dim=text_pca_dim,
            rfm=rfm,
            transductive=transductive,
            normalize_target=normalize_target
        )
        logger.debug(f"Dataset size: {get_formatted_size_kb(dataset)}")
        yield dataset
