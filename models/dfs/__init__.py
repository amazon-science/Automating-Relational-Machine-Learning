from .dfs_preprocess import DFSPreprocess
from .ft_engine import *
from .dfs2sql_engine import *
from .core import DFSConfig
from .dfs_preprocess import DFSPreprocess
from dbinfer.preprocess.transform_preprocess import RDBTransformPreprocess, RDBTransformPreprocessConfig
from dbinfer.device import get_device_info
from data.mtaskdataset import MultiTabularDataset
from data.gboltdataset import get_4dbinfer_format
from dbinfer_bench.pyg_rdb_dataset import DBBRDBDataset
from utils.misc import load_yaml
from typing import Optional, List, Iterable
from pathlib import Path
from loguru import logger
import os
import shutil
import pickle
from relbench.base import TaskType
from .helper import gpu_ipca_two_pass

def get_full_primitives():
    agg_primitives: List[str] = [
        "max",
        "min",
        "mean",
        "count",
        "mode",
        "concat",
        "join",
        "arraymax",
        "arraymin",
        "arraymean",
    ]
    return agg_primitives

def get_simple_primitives():
    agg_primitives: List[str] = [
        "max",
        "min",
        "mean",
        "count",
        "mode",
        "concat",
        "join"
    ]
    return agg_primitives


def _maybe_limit_training_samples(
    dfs_dataset: DBBRDBDataset,
    source_dataset: MultiTabularDataset,
    task_indices: Optional[Iterable[int]] = None,
) -> None:
    """Apply deterministic train subsampling to DFS datasets when requested."""
    sample_size = getattr(source_dataset, "train_sample_size", -1)
    if sample_size is None or sample_size <= 0:
        return
    base_seed = getattr(source_dataset, "train_sample_seed", 42)
    if task_indices is None:
        task_indices = range(len(source_dataset.tasks))
    for idx in task_indices:
        if idx < 0 or idx >= len(source_dataset.tasks):
            continue
        task_name = source_dataset.tasks[idx].name
        dfs_dataset.subsample_train_split(task_name, sample_size, seed=base_seed + idx)


def generate_dfs_data(
    dfs_data_save_path: str,
    dataset: MultiTabularDataset,
    max_depth: int = 2,
    autog: bool = False,
    drop_text: bool = False,
    deep_dfs: bool = False,
    return_path: bool = False,
    load_scalers_from: str = "",
    lte: bool = False,
    agg_primitives: Optional[List[str]] = None,
    primitive_suffix: str = "",
    augment_task_backlinks: bool = False,
) -> DBBRDBDataset:
    """Generate DFS (Deep Feature Synthesis) data for the given dataset.
    Only for tabpfn. This is a faster version which ignores all text embeddings.
    
    Args:
        dataset: MultiTabularDataset to process
        max_depth: Maximum depth for DFS feature generation
        autog: Whether to use autog feature generation
        drop_text: Whether to drop text features
        deep_dfs: Whether to use deep DFS configuration
        return_path: Whether to return the path along with dataset
        load_scalers_from: Path to load saved target scalers from (for evaluation)
        
    Returns:
        DBBRDBDataset: Processed dataset with DFS features
    """
    # Generate postfixes for file naming
    autog_postfix = "_autog" if autog else ""
    text_postfix = "" if drop_text else "_text"
    lte_postfix = "_lte" if lte else ""
    primitive_suffix_postfix = f"_{primitive_suffix}" if primitive_suffix else ""
    backlink_postfix = "_bklink" if augment_task_backlinks else ""
    # Construct base path components
    base_path = f"{dfs_data_save_path}/{dataset.database.name}"
    os.makedirs(base_path, exist_ok=True)
    task_name = dataset.tasks[0].name
    dfs_path = f"{base_path}/dfs_{max_depth}_{task_name}{autog_postfix}{text_postfix}{lte_postfix}{primitive_suffix_postfix}{backlink_postfix}"
    post_dfs_path = f"{base_path}/pfn_post_dfs_{max_depth}_{task_name}{autog_postfix}{text_postfix}{lte_postfix}{primitive_suffix_postfix}{backlink_postfix}"
    load_scalers_from = Path('./scalers') / load_scalers_from

    scalers_ok = False 
    if os.path.exists(os.path.join(load_scalers_from, "target_scalers.pkl")):
        opened_scalers = pickle.load(open(os.path.join(load_scalers_from, "target_scalers.pkl"), "rb"))
        scaler = list(opened_scalers.values())[0]
        if scaler is not None:
            scalers_ok = True
    # Generate DFS data if it doesn't exist
    if not os.path.exists(post_dfs_path) or (not scalers_ok and (dataset.tasks[0].task_type == TaskType.REGRESSION or dataset.tasks[0].task_type == 'regression')):
        # Configure DFS preprocessing
        dfs_config_params = {
            "engine": "dfs2sql",
            "max_depth": max_depth,
            "use_cutoff_time": True,
            "drop_text": drop_text,
            "lte": lte
        }
        # Add custom agg_primitives if provided
        if agg_primitives is not None:
            dfs_config_params["agg_primitives"] = agg_primitives

        dfs_config = DFSConfig(**dfs_config_params)
        
        # Run DFS preprocessing
        dfs_preprocess = DFSPreprocess(dfs_config)
        if augment_task_backlinks:
            four_db_format = get_4dbinfer_format(
                dataset,
                augment_task_backlinks=True,
            )
            dfs_preprocess.run(four_db_format, dfs_path)
        else:
            dfs_preprocess.run(dataset, dfs_path)
        
        # Create RDB dataset from DFS output
        new_rdb = DBBRDBDataset(
            dfs_path,
            f"{dfs_path}/metadata.yaml",
            format="4dbinfer",
            original_name=dataset.database.name
        )
        
        # Load post-processing configuration
        config_file = "configs/deep_dfs.yaml" if deep_dfs else "configs/dfs.yaml"
        dfs_post_config = load_yaml(config_file)
        dfs_post_config['path_to_save_scalers'] = load_scalers_from

        # Run RDB transform preprocessing
        rdb_transform_preprocess = RDBTransformPreprocess(
            RDBTransformPreprocessConfig(**dfs_post_config)
        )
        device = get_device_info()
        
        # Load target scalers if provided (for evaluation)
        if load_scalers_from:
            logger.debug(f"Save target scalers to {load_scalers_from}")
            rdb_transform_preprocess.load_target_scalers(Path(load_scalers_from))
        
        rdb_transform_preprocess.run(new_rdb, post_dfs_path, device)
    
    # Create final transformed dataset
    new_rdb_transformed = DBBRDBDataset(
        post_dfs_path,
        f"{post_dfs_path}/metadata.yaml",
        format="4dbinfer",
        original_name=dataset.database.name
    )

    new_rdb_transformed.scaler_path = load_scalers_from
    _maybe_limit_training_samples(new_rdb_transformed, dataset)
    
    # Clean up intermediate DFS directory if it exists
    if os.path.exists(dfs_path):
        shutil.rmtree(dfs_path)
    
    if return_path:
        return new_rdb_transformed, post_dfs_path
    else:
        return new_rdb_transformed

def generate_dfs_data_pre_post(
    dfs_data_save_path: str,
    dataset: MultiTabularDataset,
    max_depth: int = 2,
    load_scalers_from: str = "",
    mode = "full",
    lte = False,
    text_to_pca = False,
    text_to_pca_dim = 3,
    task_idx = 0,
    agg_primitives: Optional[List[str]] = None,
    primitive_suffix: str = "",
    augment_task_backlinks: bool = False,
) -> DBBRDBDataset:
    """
    Generate DFS data for the given dataset.
    Do the dfs for word embedding too.
    """
    four_db_format = get_4dbinfer_format(
        dataset,
        text_to_pca=text_to_pca,
        text_to_pca_dim=text_to_pca_dim,
        task_idx=task_idx,
        augment_task_backlinks=augment_task_backlinks,
    )
    pre_config = "configs/predfs.yaml"
    pre_config_loaded = load_yaml(pre_config)
    pre_config_pca = "configs/predfs_pca.yaml"
    pre_config_pca_loaded = load_yaml(pre_config_pca)
    post_config = "configs/postdfs.yaml"
    post_config_loaded = load_yaml(post_config)
    # Generate postfixes for file naming
    autog_postfix = ""
    text_postfix = "_text" if not text_to_pca else f"_text_pca_{text_to_pca_dim}"
    lte_postfix = "_lte" if lte else ""
    primitive_suffix_postfix = f"_{primitive_suffix}" if primitive_suffix else ""
    backlink_postfix = "_bklink" if augment_task_backlinks else ""
    post_fix_4db = "4dbinfer"
    load_scalers_from = Path('./scalers') / load_scalers_from
    
    # Construct base path components
    base_path = f"{dfs_data_save_path}/{dataset.database.name}"
    task_name = dataset.tasks[0].name
    pre_dfs_path = f"{base_path}/pre_dfs_{max_depth}_{task_name}{autog_postfix}{text_postfix}{lte_postfix}{primitive_suffix_postfix}{backlink_postfix}_{post_fix_4db}"
    dfs_path = f"{base_path}/dfs_{max_depth}_{task_name}{autog_postfix}{text_postfix}{lte_postfix}{primitive_suffix_postfix}{backlink_postfix}_{post_fix_4db}"
    post_dfs_path = f"{base_path}/post_dfs_{max_depth}_{task_name}{autog_postfix}{text_postfix}{lte_postfix}{primitive_suffix_postfix}{backlink_postfix}_{post_fix_4db}"
    
    # Generate DFS data if it doesn't exist
    if not os.path.exists(post_dfs_path):
        rdb_transform_preprocess = RDBTransformPreprocess(
            RDBTransformPreprocessConfig(**pre_config_loaded if not text_to_pca else pre_config_pca_loaded)
        )
        device = get_device_info()
        rdb_transform_preprocess.run(four_db_format, pre_dfs_path, device)
        # Configure DFS preprocessing
        # Use custom agg_primitives if provided, otherwise fall back to mode
        if agg_primitives is None:
            primitives = get_full_primitives() if mode == "full" else get_simple_primitives()
        else:
            primitives = agg_primitives

        dfs_config = DFSConfig(
            engine="dfs2sql",
            max_depth=max_depth,
            use_cutoff_time=True,
            drop_text=False,
            agg_primitives=primitives,
            lte = lte
        )

        old_rdb = DBBRDBDataset(
            pre_dfs_path,
            f"{pre_dfs_path}/metadata.yaml",
            format="4dbinfer",
            original_name=dataset.database.name
        )

        old_rdb.metadata.test_timestamp_cutoff = four_db_format.metadata.test_timestamp_cutoff
        old_rdb.metadata.val_timestamp_cutoff = four_db_format.metadata.val_timestamp_cutoff

        # Run DFS preprocessing
        dfs_preprocess = DFSPreprocess(dfs_config)
        dfs_preprocess.run(old_rdb, dfs_path)
        
        # Create RDB dataset from DFS output
        new_rdb = DBBRDBDataset(
            dfs_path,
            f"{dfs_path}/metadata.yaml",
            format="4dbinfer",
            original_name=dataset.database.name
        )
        new_rdb.metadata.test_timestamp_cutoff = four_db_format.metadata.test_timestamp_cutoff
        new_rdb.metadata.val_timestamp_cutoff = four_db_format.metadata.val_timestamp_cutoff
    
        rdb_transform_postprocess = RDBTransformPreprocess(
                RDBTransformPreprocessConfig(**post_config_loaded)
            )
        
        if load_scalers_from:
            logger.debug(f"Save target scalers to {load_scalers_from}")
            rdb_transform_postprocess.load_target_scalers(Path(load_scalers_from))
        rdb_transform_postprocess.run(new_rdb, post_dfs_path, device)
        # Create final transformed dataset
    new_rdb_transformed = DBBRDBDataset(
        post_dfs_path,
        f"{post_dfs_path}/metadata.yaml",
        format="4dbinfer",
        original_name=dataset.database.name
    )

    new_rdb_transformed.scaler_path = load_scalers_from
    _maybe_limit_training_samples(new_rdb_transformed, dataset, task_indices=[task_idx])
    
    # Clean up intermediate DFS directory if it exists
    if os.path.exists(pre_dfs_path):
        shutil.rmtree(pre_dfs_path)
    if os.path.exists(dfs_path):
        shutil.rmtree(dfs_path)
    return new_rdb_transformed

def extract_columns_by_name(columns, substring: str, case_sensitive: bool = True):
    """Extract columns whose names contain a specific substring.
    
    Args:
        columns: List of column objects (e.g., DBBColumnSchema)
        substring: String to search for in column names
        case_sensitive: Whether the search should be case sensitive
        
    Returns:
        List of column objects whose names contain the substring
    """
    if not case_sensitive:
        substring = substring.lower()
    
    matching_columns = []
    for column in columns:
        column_name = column.name
        if not case_sensitive:
            column_name = column_name.lower()
        
        if substring in column_name:
            matching_columns.append(column)
    
    return matching_columns

def extract_columns_by_dtype(columns, dtype: str):
    """Extract columns with a specific data type.
    
    Args:
        columns: List of column objects (e.g., DBBColumnSchema)
        dtype: Data type to filter by (e.g., 'float', 'category', 'datetime')
        
    Returns:
        List of column objects with the specified data type
    """
    matching_columns = []
    for column in columns:
        if column.dtype == dtype:
            matching_columns.append(column)
    
    return matching_columns
