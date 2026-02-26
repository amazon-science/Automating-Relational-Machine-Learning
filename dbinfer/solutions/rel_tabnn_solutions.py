"""
Conversion utilities for converting DBBRDBDataset to PyTorch-Frame format.

This module provides functions to convert relational database datasets from the
DBBRDBDataset format to PyTorch-Frame's DatasetWithTransform format, enabling
seamless integration with PyTorch-Frame's tensor processing capabilities.
"""

import pandas as pd
import torch
import torch_frame
from torch_frame.config import TextEmbedderConfig
from typing import Dict, Optional, Any, List, Tuple, Union
import numpy as np
from pathlib import Path

# Import necessary components
from dbinfer_bench.pyg_rdb_dataset import DBBRDBDataset, DBBRDBTask
from dbinfer_bench.dataset_meta import DBBColumnDType, DBBTaskType
from data.relbench import DatasetwithTransform

import torch
from torch_frame.data import DataLoader
from contextgnn.nn.encoder import DEFAULT_STYPE_ENCODER_DICT
from tqdm import tqdm
import numpy as np
from models.embedder import GloveTextEmbedding
from ..evaluator import get_metrics
from torch_frame import Metric
from torch_frame import TaskType as TFTaskType
import joblib
from loguru import logger
from torch_frame.gbdt import LightGBM
from copy import deepcopy


def map_dbb_dtype_to_torch_frame_stype(dbb_dtype: DBBColumnDType) -> torch_frame.stype:
    """
    Map DBBColumnDType to torch_frame.stype.
    
    Args:
        dbb_dtype: DBBColumnDType to convert
        
    Returns:
        torch_frame.stype: Corresponding semantic type
        
    Raises:
        ValueError: If the dtype is not supported
    """
    dtype_mapping = {
        DBBColumnDType.float_t: torch_frame.numerical,
        DBBColumnDType.category_t: torch_frame.categorical,
        DBBColumnDType.datetime_t: torch_frame.timestamp,
        DBBColumnDType.text_t: torch_frame.text_embedded,  # Can be text_tokenized if needed
        DBBColumnDType.timestamp_t: torch_frame.timestamp,
        DBBColumnDType.multi_category_t: torch_frame.multicategorical,
        DBBColumnDType.primary_key: torch_frame.categorical,  # Treat as categorical
        DBBColumnDType.foreign_key: torch_frame.categorical,  # Treat as categorical
        DBBColumnDType.split_t: torch_frame.categorical,  # Split indicators,
        'float': torch_frame.numerical,
        'category': torch_frame.categorical,
        'datetime': torch_frame.timestamp,
        'text': torch_frame.text_embedded,
        'timestamp': torch_frame.timestamp,
        'multicategory': torch_frame.multicategorical,
        'primary_key': torch_frame.categorical,
        'foreign_key': torch_frame.categorical,
    }
    
    if dbb_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported DBBColumnDType: {dbb_dtype}")
    
    return dtype_mapping[dbb_dtype]

def dict_to_dataframe(data_dict: dict) -> pd.DataFrame:
    """
    Converts a dictionary with multi-dimensional NumPy arrays or PyTorch tensors
    into a pandas DataFrame, preserving the structure of the tensor slices.

    This function identifies tensor-like objects (np.ndarray, torch.Tensor) and
    creates a row in the DataFrame for each item in the tensor's first dimension.
    The slice itself (which can be an array or tensor) is placed in the cell.
    Other scalar key-value pairs in the dictionary are treated as metadata and
    are duplicated for each row.

    Args:
        data_dict (dict): The input dictionary. The values can be scalars,
                          strings, NumPy arrays, or PyTorch tensors. It is
                          assumed that all array/tensor values have the same
                          size in their first dimension.

    Returns:
        pd.DataFrame: A pandas DataFrame constructed from the dictionary, where
                      columns corresponding to tensors will contain the tensor slices
                      (e.g., arrays or tensors) as cell values.
    """
    # --- 1. Separate scalar/metadata from tensor-like data ---
    scalar_data = {}
    tensor_data = {}
    for key, value in data_dict.items():
        # Check if the value is a NumPy array or a PyTorch tensor
        if isinstance(value, (np.ndarray, torch.Tensor)):
            tensor_data[key] = value
        else:
            scalar_data[key] = value

    # --- 2. Handle the case with no tensors ---
    # If no tensor-like objects are found, treat the dict as a single record
    if not tensor_data:
        return pd.DataFrame([scalar_data])

    # --- 3. Determine the number of rows ---
    num_rows = -1
    # We can inspect the original tensors directly. We only need the first one
    # to determine the number of rows for the DataFrame.
    first_tensor = next(iter(tensor_data.values()))
    
    # Ensure the tensor is at least 1D to have a shape[0]
    val_shape = getattr(first_tensor, 'shape', None)
    if val_shape is not None and len(val_shape) > 0:
        num_rows = first_tensor.shape[0]
    else: # Handle scalar-like tensors e.g., np.array(5)
        num_rows = 1
        # Reshape all scalar-like tensors to have a first dimension
        for key, value in tensor_data.items():
            if not getattr(value, 'shape', None) or len(value.shape) == 0:
                 tensor_data[key] = value.reshape(1) if hasattr(value, 'reshape') else np.array([value])


    # If tensors were empty, num_rows might not be set properly
    if num_rows <= 0:
        return pd.DataFrame([scalar_data])

    # --- 4. Build the list of records for the DataFrame ---
    records = []
    for i in range(num_rows):
        # Start with the scalar data for each new row
        record = scalar_data.copy()
        
        # Add the i-th slice of each tensor directly into the record.
        # This preserves the slice's dimensions.
        for key, tensor in tensor_data.items():
            record[key] = tensor[i]
            
        records.append(record)

    return pd.DataFrame(records)


def create_column_type_mapping(
    dbb_dataset: DBBRDBDataset,
    task_name: str,
    exclude_keys: bool = True,
    text_as_embedded: bool = True
) -> Dict[str, torch_frame.stype]:
    """
    Create column-to-stype mapping for a specific task based on actual training data.
    
    Args:
        dbb_dataset: Source DBBRDBDataset
        task_name: Name of the task to create mapping for
        exclude_keys: Whether to exclude primary/foreign key columns
        text_as_embedded: Whether to treat text as embedded or tokenized
        
    Returns:
        Dictionary mapping column names to torch_frame semantic types
    """
    task = dbb_dataset.get_task(task_name)
    
    # Get the training data to see all available columns
    train_data = task.train_set
    
    # For relbench format, we have a Table object with .df attribute
    if hasattr(train_data, 'df'):
        df = train_data.df
    else:
        # For 4dbinfer format, we need to construct DataFrame from dict
        if isinstance(train_data, dict):
            df = dict_to_dataframe(train_data)
        else:
            raise ValueError(f"Unsupported train data format: {type(train_data)}")
    
    col_to_stype = {}
    
    # Convert task.metadata.columns into mapping from column names to dtype
    col_dtype_mapping = {
        col.name: col.dtype for col in task.metadata.columns
    }
    
    # Create a mapping from column names to their DBB types
    # We'll need to infer types from the actual data and metadata
    for col_name in df.columns:
        # Skip key columns if requested (we'll infer this from column names or metadata)
        if exclude_keys and any(keyword in col_name.lower() for keyword in ['_pk', '_fk']):
            continue
        if exclude_keys and (col_dtype_mapping[col_name] == 'foreign_key' or col_dtype_mapping[col_name] == 'primary_key'):
            continue
        # Get the column data
        col_data = df[col_name]
        
        # Infer the type based on the data and metadata
        if col_name in col_dtype_mapping:
            # Use metadata if available
            # if col_dtype_mapping[col_name] 
            col_dtype = col_dtype_mapping[col_name]
        else:
            # Infer from data
            if col_data.dtype == 'object':
                # Check if it's text or categorical
                if col_data.str.len().max() > 50:  # Long strings are likely text
                    col_dtype = DBBColumnDType.text_t
                else:
                    col_dtype = DBBColumnDType.category_t
            elif col_data.dtype in ['int64', 'int32']:
                col_dtype = DBBColumnDType.category_t  # Assume categorical for now
            elif col_data.dtype in ['float64', 'float32']:
                col_dtype = DBBColumnDType.float_t
            else:
                col_dtype = DBBColumnDType.category_t  # Default to categorical
        
        # Handle text type preference
        if col_dtype == DBBColumnDType.text_t:
            col_to_stype[col_name] = torch_frame.text_embedded if text_as_embedded else torch_frame.text_tokenized
        elif col_dtype == DBBColumnDType.float_t:
            if col_data.dtype == 'object':
                col_to_stype[col_name] = torch_frame.embedding 
            else:
                col_to_stype[col_name] = map_dbb_dtype_to_torch_frame_stype(col_dtype)
        else:
            col_to_stype[col_name] = map_dbb_dtype_to_torch_frame_stype(col_dtype)
    
    return col_to_stype


def convert_dbb_table_to_dataframe(
    dbb_dataset: DBBRDBDataset,
    task_name: str,
    split: str = "train",
    include_target: bool = True
) -> pd.DataFrame:
    """
    Convert DBBRDBDataset table to pandas DataFrame.
    
    Args:
        dbb_dataset: Source DBBRDBDataset
        task_name: Name of the task
        split: Data split ('train', 'val', 'test')
        include_target: Whether to include target column
        
    Returns:
        pandas DataFrame with the table data
    """
    task = dbb_dataset.get_task(task_name)
    
    # Get the appropriate split
    if split == "train":
        table_data = task.train_set
    elif split == "val" or split == "validation":
        table_data = task.validation_set
    elif split == "test":
        table_data = task.test_set
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    # For relbench format, we have a Table object with .df attribute
    if hasattr(table_data, 'df'):
        df = table_data.df.copy()
    else:
        # For 4dbinfer format, we need to construct DataFrame from dict
        if isinstance(table_data, dict):
            df = dict_to_dataframe(table_data)
        else:
            raise ValueError(f"Unsupported table data format: {type(table_data)}")
    
    return df


def setup_text_embedder_config(
    text_columns: List[str],
    embedder_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> Dict[str, TextEmbedderConfig]:
    """
    Setup text embedder configuration for text columns.
    
    Args:
        text_columns: List of text column names
        embedder_model: Name of the embedding model
        batch_size: Batch size for embedding generation
        
    Returns:
        Dictionary mapping column names to TextEmbedderConfig
    """
    embedder = GloveTextEmbedding(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return embedder


def _format_dataset_name(base: str, suffix: Optional[str]) -> str:
    if not suffix:
        return base
    safe_suffix = str(suffix).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not safe_suffix:
        return base
    return f"{base}_{safe_suffix}"


def convert_dbb_to_torch_frame_dataset(
    dbb_dataset: DBBRDBDataset,
    task_name: str,
    task_type: str,
    split: str = "train",
    exclude_keys: bool = True,
    text_as_embedded: bool = True,
    embedder_batch_size: int = 4096,
    dataset_name: Optional[str] = None,
    transform: Optional[str] = None,
    device: Optional[torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
    cache_path: Optional[str] = None,
    remove_add_feature: bool = True,
    remove_cat_temporal_information: bool = True,
    sample_size: int = 50000,
    scaler_path: Optional[str] = None,
    val_sample_batch: int = 2000,
    *,
    target_column_override: Optional[str] = None,
    dataset_name_suffix: Optional[str] = None,
) -> DatasetwithTransform:
    """
    Convert DBBRDBDataset to PyTorch-Frame DatasetWithTransform.
    
    Args:
        dbb_dataset: Source DBBRDBDataset
        task_name: Name of the task to convert
        task_type: Type of task ('classification', 'regression', 'multiclass_classification')
        split: Data split to use ('train', 'val', 'test')
        exclude_keys: Whether to exclude primary/foreign key columns
        text_as_embedded: Whether to treat text as embedded or tokenized
        embedder_model: Text embedding model name
        embedder_batch_size: Batch size for text embedding
        dataset_name: Name for caching purposes
        transform: Transform to apply ('distribution', 'none', etc.)
        device: Device to load tensors on
        cache_path: Path for caching materialized dataset
        target_column_override: Optional alternate target column to supervise on
        dataset_name_suffix: Optional suffix to append to dataset_name/cache files for disambiguation
        
    Returns:
        DatasetWithTransform: PyTorch-Frame dataset
        
    Raises:
        ValueError: If task or table not found
        RuntimeError: If conversion fails
    """

    # Get task and validate
    task = dbb_dataset.get_task(task_name)
    if dataset_name is None:
        dataset_name = f"{dbb_dataset.dataset_name}_{task_name}_{split}"
    dataset_name = _format_dataset_name(dataset_name, dataset_name_suffix)
    
    # Convert table to DataFrame
    df = convert_dbb_table_to_dataframe(dbb_dataset, task_name, split, include_target=True)
    
    # Create column type mapping
    col_to_stype = create_column_type_mapping(
        dbb_dataset, task_name, exclude_keys, text_as_embedded
    )
    
    # Filter columns to only include those in the mapping
    available_columns = set(df.columns)
    target_column = target_column_override or task.metadata.target_column
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataframe for task '{task_name}' (split={split})."
        )

    # if task_type == 'multiclass_classification':
    #     scaler_path = Path(scaler_path) / 'target_scalers.pkl'
    #     scaler = list(joblib.load(scaler_path).values())[0]
    #     df[target_column] = scaler.inverse_transform(df[target_column].values.reshape(-1, 1)).reshape(-1).astype(int)

    
    # Ensure target column is included
    if target_column not in col_to_stype and target_column in available_columns:
        # Infer target column type based on task type
        if task.metadata.task_type in [DBBTaskType.binary_classification, DBBTaskType.classification]:
            col_to_stype[target_column] = torch_frame.categorical
        elif task.metadata.task_type == DBBTaskType.regression:
            col_to_stype[target_column] = torch_frame.numerical
        else:
            col_to_stype[target_column] = torch_frame.categorical  # Default
    
    # Filter DataFrame to only include mapped columns
    valid_columns = [col for col in df.columns if col in col_to_stype]
    logger.debug(f"Number of columns before filtering: {len(df.columns)}")
    df = df[valid_columns]
    if f'{dbb_dataset.tasks[0].metadata.target_table}.{dbb_dataset.tasks[0].metadata.target_column}' in df.columns:
        df = df.drop(columns=[f'{dbb_dataset.tasks[0].metadata.target_table}.{dbb_dataset.tasks[0].metadata.target_column}'])
        col_to_stype = {col: stype for col, stype in col_to_stype.items() if col != f'{dbb_dataset.tasks[0].metadata.target_table}.{dbb_dataset.tasks[0].metadata.target_column}'}

    if remove_add_feature:
        df = df.drop(columns=[col for col in df.columns if 'add_feature' in col])
        col_to_stype = {col: stype for col, stype in col_to_stype.items() if 'add_feature' not in col}
    
    if remove_cat_temporal_information:
        df = df.drop(columns=[col for col in df.columns if 'YEAR' in col or 'MONTH' in col or 'DAY' in col or 'DAYOFWEEK' in col])
        col_to_stype = {col: stype for col, stype in col_to_stype.items() if 'YEAR' not in col and 'MONTH' not in col and 'DAY' not in col and 'DAYOFWEEK' not in col}
    logger.debug(f"Number of columns after filtering: {len(df.columns)}")
    # Setup text embedder configuration
    text_columns = [col for col, stype in col_to_stype.items() 
                    if stype == torch_frame.text_embedded and col in df.columns]
    
    col_to_text_embedder_cfg = None
    if text_columns and text_as_embedded:
        col_to_text_embedder_cfg = TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device),
            batch_size=embedder_batch_size
        )
    
    # Create split column if needed
    split_col = None
    if 'split' in df.columns:
        split_col = 'split'
    
    if sample_size is not None and split == 'train':
        ## sample when training is too large
        sampled_idx = np.random.permutation(len(df))[: sample_size]
        df = df.iloc[sampled_idx]
    
    if val_sample_batch is not None and split == 'val':
        sampled_idx = np.random.permutation(len(df))[: val_sample_batch]
        df = df.iloc[sampled_idx]
    
    # Create DatasetWithTransform
    torch_frame_dataset = DatasetwithTransform(
        df=df,
        col_to_stype=col_to_stype,
        target_col=target_column,
        split_col=split_col,
        col_to_text_embedder_cfg=col_to_text_embedder_cfg,
        dataset_name=dataset_name
    )
    
    # Materialize the dataset
    torch_frame_dataset = torch_frame_dataset.materialize(
        device=device,
        transform=transform,
        path=cache_path
    )
    
    return torch_frame_dataset
        
    # except Exception as e:
    #     raise RuntimeError(f"Failed to convert DBBRDBDataset to PyTorch-Frame: {str(e)}") from e


def convert_all_splits(
    dbb_dataset: DBBRDBDataset,
    task_name: str,
    task_type: str,
    exclude_keys: bool = True,
    text_as_embedded: bool = True,
    embedder_batch_size: int = 4096,
    dataset_name: Optional[str] = None,
    transform: Optional[str] = None,
    device: Optional[torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
    cache_dir: Optional[str] = None, 
    remove_add_feature: bool = True,
    remove_cat_temporal_information: bool = True,
    sample_size: int = 200000,
    scaler_path: Optional[str] = None,
    val_sample_batch: int = 200,
    target_column_override: Optional[str] = None,
    dataset_name_suffix: Optional[str] = None,
) -> Tuple[DatasetwithTransform, DatasetwithTransform, DatasetwithTransform]:
    """
    Convert all splits (train, val, test) of a DBBRDBDataset task to PyTorch-Frame format.
    
    Args:
        dbb_dataset: Source DBBRDBDataset
        task_name: Name of the task to convert
        task_type: Type of task ('classification', 'regression', 'multiclass_classification')
        exclude_keys: Whether to exclude primary/foreign key columns
        text_as_embedded: Whether to treat text as embedded or tokenized
        embedder_model: Text embedding model name
        embedder_batch_size: Batch size for text embedding
        dataset_name: Name for caching purposes
        transform: Transform to apply ('distribution', 'none', etc.)
        device: Device to load tensors on
        cache_dir: Directory for caching materialized datasets
        target_column_override: Optional alternate target column to supervise on
        dataset_name_suffix: Optional suffix added to dataset/cache names (useful when swapping targets)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if dataset_name is None:
        dataset_name = f"{dbb_dataset.dataset_name}_{task_name}"
    dataset_name = _format_dataset_name(dataset_name, dataset_name_suffix)
    
    datasets = {}
    
    for split in ["train", "val", "test"]:
        cache_path = None
        split_dataset_name = f"{dataset_name}_{split}"
        if cache_dir is not None:
            cache_path = str(Path(cache_dir) / f"{split_dataset_name}.pt")

        datasets[split] = convert_dbb_to_torch_frame_dataset(
            dbb_dataset=dbb_dataset,
            task_name=task_name,
            task_type=task_type,
            split=split,
            exclude_keys=exclude_keys,
            text_as_embedded=text_as_embedded,
            embedder_batch_size=embedder_batch_size,
            dataset_name=split_dataset_name,
            transform=transform,
            device=device,
            cache_path=cache_path,
            remove_add_feature=remove_add_feature,
            remove_cat_temporal_information=remove_cat_temporal_information,
            sample_size=sample_size if split in ['train', 'val'] else None,
            scaler_path=scaler_path,
            val_sample_batch=val_sample_batch,
            target_column_override=target_column_override,
        )
    
    
    return datasets["train"], datasets["val"], datasets["test"]


def get_dataset_info(torch_frame_dataset: DatasetwithTransform) -> Dict[str, Any]:
    """
    Get information about the converted PyTorch-Frame dataset.
    
    Args:
        torch_frame_dataset: Converted PyTorch-Frame dataset
        
    Returns:
        Dictionary containing dataset information
    """
    info = {
        "num_rows": len(torch_frame_dataset),
        "num_cols": len(torch_frame_dataset.feat_cols),
        "target_col": torch_frame_dataset.target_col,
        "task_type": torch_frame_dataset.task_type,
        "column_types": torch_frame_dataset.col_to_stype,
        "is_materialized": torch_frame_dataset.is_materialized,
    }
    
    if torch_frame_dataset.is_materialized:
        tensor_frame = torch_frame_dataset.tensor_frame
        info.update({
            "tensor_frame_shape": (tensor_frame.num_rows, tensor_frame.num_cols),
            "semantic_types": tensor_frame.stypes,
            "device": tensor_frame.device,
            "has_target": tensor_frame.y is not None,
        })
    
    return info




def check_task_type(task: DBBRDBTask) -> Tuple[str, int]:
    """
    Check the task type and return the task type and number of classes.
    """
    if task.metadata.task_type == 'regression':
        return 'regression', 1
    elif task.metadata.task_type == 'classification':
        num_classes = len(np.unique(task.train_set[task.metadata.target_column]))
        return 'classification', num_classes
    elif task.metadata.task_type == 'retrieval':
        return 'retrieval', 1
    else:
        raise ValueError(f"Unsupported task type: {task.metadata.task_type}")

class ValidationMetricsTracker:
    """Tracks validation metrics across epochs for checkpoint selection and early stopping.

    For regression tasks, combines MAE and R2 with 50-50 weight after normalization.
    For classification tasks, uses a single primary metric.
    """

    def __init__(
        self,
        task_type: str,
        primary_metric: str,
        patience: int = 10,
        min_delta: float = 0.0,
        epsilon: float = 1e-10
    ):
        """
        Args:
            task_type: Type of task ('regression', 'classification', 'binary_classification', 'multiclass_classification')
            primary_metric: Primary metric for non-regression tasks
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change in monitored quantity to qualify as an improvement
            epsilon: Small constant to avoid division by zero in normalization
        """
        self.task_type = task_type
        self.primary_metric = primary_metric
        self.patience = patience
        self.min_delta = min_delta
        self.epsilon = epsilon

        # Store all validation metrics across epochs
        self.all_metrics = []  # List of metric dicts, one per epoch
        self.best_combined_score = None
        self.best_epoch = None
        self.counter = 0
        self.should_stop = False

    def add_epoch_metrics(self, metrics: Dict[str, float]) -> None:
        """Add validation metrics for the current epoch.

        Args:
            metrics: Dictionary of metric names to values
        """
        self.all_metrics.append(metrics.copy())

    def _normalize_metrics(self) -> List[Dict[str, float]]:
        """Normalize metrics across all epochs to [0, 1] range.

        Returns:
            List of normalized metric dictionaries
        """
        if not self.all_metrics:
            return []

        # Extract all metric names
        metric_names = set()
        for metrics in self.all_metrics:
            metric_names.update(metrics.keys())

        # Compute min/max for each metric
        metric_ranges = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in self.all_metrics if metric_name in m]
            if values:
                metric_ranges[metric_name] = {
                    'min': min(values),
                    'max': max(values),
                }

        # Normalize each epoch's metrics
        normalized = []
        for metrics in self.all_metrics:
            norm_metrics = {}
            for metric_name, value in metrics.items():
                min_val = metric_ranges[metric_name]['min']
                max_val = metric_ranges[metric_name]['max']
                range_val = max_val - min_val + self.epsilon

                # Determine if higher is better for this metric
                # MAE, MSE, RMSE: lower is better (invert)
                # R2, accuracy, AUROC, F1: higher is better
                lower_is_better = metric_name.lower() in ['mae', 'mse', 'rmse', 'loss']

                if lower_is_better:
                    # Invert: lower values get higher normalized scores
                    norm_metrics[metric_name] = (max_val - value) / range_val
                else:
                    # Higher values get higher normalized scores
                    norm_metrics[metric_name] = (value - min_val) / range_val

            normalized.append(norm_metrics)

        return normalized

    def get_combined_score(self, epoch_idx: int) -> float:
        """Compute combined score for a specific epoch.

        For regression: 50% MAE + 50% R2 (after normalization)
        For classification: primary metric only

        Args:
            epoch_idx: Index of the epoch

        Returns:
            Combined score (higher is better)
        """
        if epoch_idx >= len(self.all_metrics):
            return float('-inf')

        normalized = self._normalize_metrics()
        norm_metrics = normalized[epoch_idx]

        if self.task_type == 'regression':
            # Combine MAE and R2 with 50-50 weight
            mae_score = norm_metrics.get('mae', 0.0)
            r2_score = norm_metrics.get('r2', 0.0)
            combined_score = 0.5 * mae_score + 0.5 * r2_score
        else:
            # Use primary metric only
            combined_score = norm_metrics.get(self.primary_metric, 0.0)

        return combined_score

    def check_early_stopping(self) -> bool:
        """Check if training should stop based on validation metrics.

        Returns:
            True if training should stop, False otherwise
        """
        if not self.all_metrics:
            return False

        current_epoch_idx = len(self.all_metrics) - 1
        current_score = self.get_combined_score(current_epoch_idx)

        if self.best_combined_score is None:
            self.best_combined_score = current_score
            self.best_epoch = current_epoch_idx
            return False

        # Check if current score is better (higher is always better after normalization)
        improved = current_score > self.best_combined_score + self.min_delta

        if improved:
            self.best_combined_score = current_score
            self.best_epoch = current_epoch_idx
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False

    def get_best_epoch(self) -> int:
        """Get the epoch index with the best combined score.

        Returns:
            Best epoch index
        """
        if not self.all_metrics:
            return 0

        best_idx = 0
        best_score = float('-inf')

        for idx in range(len(self.all_metrics)):
            score = self.get_combined_score(idx)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def get_best_metrics(self) -> Dict[str, float]:
        """Get the validation metrics from the best epoch.

        Returns:
            Dictionary of metric names to values
        """
        if not self.all_metrics:
            return {}

        best_idx = self.get_best_epoch()
        return self.all_metrics[best_idx].copy()


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change in monitored quantity to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.should_stop = False

    def __call__(self, score: float, model: torch.nn.Module, is_higher_better: bool = True) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric score
            model: PyTorch model
            is_higher_better: Whether higher scores are better

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {name: param.clone().detach() for name, param in model.named_parameters()}
            return False

        if is_higher_better:
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {name: param.clone().detach() for name, param in model.named_parameters()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                # Restore best weights
                for name, param in model.named_parameters():
                    if name in self.best_weights:
                        param.data.copy_(self.best_weights[name])
            return True

        return False


def rel_tabnn_solution(
    dfs_dataset: DBBRDBDataset,
    task_name: str,
    model_type: str,
    model_config: dict,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    save_model: bool = False,
    model_weight_name: str = 'model_weight.pt',
    return_predictions: bool = False,
    target_column_override: Optional[str] = None,
    dataset_name_suffix: Optional[str] = None,
):
    """Train a ResNet/FT-Transformer model on DFS-generated dataset.
    
    Args:
        dfs_dataset: DFS-generated dataset
        task_name: Name of the task
        model_type: Type of model ('resnet' or 'ft_transformer')
        model_config: Configuration dictionary for the model
        target_column_override: Optional alternate target column for supervision
        dataset_name_suffix: Optional suffix to differentiate cached datasets when relabeling
    """    
    scaler_path = dfs_dataset.scaler_path
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_sample_batch = model_config.get('val_sample_size', 2000)
    model_config['max_steps_per_epoch'] = 500
    
    # Get the task from dataset
    task = dfs_dataset.tasks[0]  # Assuming single task for now
    task_type, num_classes = check_task_type(task)
    is_classification = task_type == 'classification'

    if (task_type == 'classification' or task_type == 'binary_classification') and num_classes > 2:
        task_type = 'multiclass_classification'
    elif task_type == 'binary_classification' and num_classes == 2:
        task_type = 'binary_classification'
    elif task_type == 'classification' and num_classes == 2:
        task_type = 'binary_classification'

    remove_add_feature = model_config['remove_add_feature']
    remove_cat_temporal_information = model_config['remove_cat_temporal_information']

    # Convert datasets to torch_frame format
    train_ds, val_ds, test_ds = convert_all_splits(
        dbb_dataset=dfs_dataset,
        task_type=task_type,
        task_name=task_name,
        transform="none",
        cache_dir="./datasets",
        remove_add_feature=remove_add_feature,
        remove_cat_temporal_information=remove_cat_temporal_information, 
        sample_size = None,
        scaler_path=scaler_path,
        val_sample_batch=val_sample_batch,
        target_column_override=target_column_override,
        dataset_name_suffix=dataset_name_suffix,
    )

    best_model = None
    
    # Create model based on model_type
    if model_type.lower() == 'resnet':
        model = create_resnet_model(train_ds, model_config, task_type, num_classes)
    elif model_type.lower() == 'ft_transformer':
        model = create_ft_transformer_model(train_ds, model_config, task_type, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_ds.tensor_frame, batch_size=model_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds.tensor_frame, batch_size=model_config['batch_size'])
    test_loader = DataLoader(test_ds.tensor_frame, batch_size=model_config['eval_batch_size'])
    # more_train_loader = DataLoader(more_train_ds.tensor_frame, batch_size=model_config['batch_size'], shuffle=True)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=model_config['gamma'])
    
    loss_fn = {
        'bce': torch.nn.BCEWithLogitsLoss(),
        'bpr': torch.nn.BCEWithLogitsLoss(),
        'mse': torch.nn.MSELoss(),
        'mae': torch.nn.L1Loss(),
        'ce': torch.nn.CrossEntropyLoss(),
    }[model_config['loss_fn']]

    # Training function
    def train_epoch(epoch: int) -> float:
        model.train()
        loss_accum = total_count = 0
        
        for i, tf in tqdm(enumerate(train_loader), desc=f'Epoch: {epoch}'):
            if i >= model_config['max_steps_per_epoch']:
                break
            tf = tf.to(device)
            pred = model(tf)
            
            if is_classification:
                if pred.dim() == 2 and pred.shape[1] == 1:
                    pred = pred.squeeze(1)
                    loss = loss_fn(pred, tf.y.view(-1).float())
                else:
                    loss = loss_fn(pred, tf.y.view(-1).long())
            else:
                loss = loss_fn(pred.view(-1), tf.y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            loss_accum += float(loss) * len(tf.y)
            total_count += len(tf.y)
            optimizer.step()
        scheduler.step()
        return loss_accum / total_count
    
    # Evaluation function
    @torch.no_grad()
    def evaluate(
        loader: DataLoader,
        return_predictions: bool = False,
        model_override: Optional[torch.nn.Module] = None,
    ) -> Union[
        Dict[str, float],
        Tuple[Dict[str, float], torch.Tensor, torch.Tensor],
    ]:
        eval_model = model_override if model_override is not None else model
        eval_model.eval()
        all_preds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        
        for tf in tqdm(loader, desc="Evaluating"):
            tf = tf.to(device)
            pred = eval_model(tf)
            all_preds.append(pred.cpu())
            all_labels.append(tf.y.cpu())
        
        if not all_preds:
            empty_metrics = {metric_name: 0.0 for metric_name in get_metrics(task_type, num_classes)}
            if return_predictions:
                return empty_metrics, torch.empty(0), torch.empty(0)
            return empty_metrics

        raw_pred = torch.cat(all_preds, dim=0)
        raw_label = torch.cat(all_labels, dim=0)

        metric_fns = get_metrics(task_type, num_classes)

        processed_pred: torch.Tensor
        processed_label: torch.Tensor

        if task_type in ('classification', 'binary_classification'):
            if num_classes == 2:
                if raw_pred.dim() == 2 and raw_pred.shape[1] == 1:
                    probs_pos = torch.sigmoid(raw_pred.squeeze(1))
                    processed_pred = torch.stack([1 - probs_pos, probs_pos], dim=1)
                elif raw_pred.dim() == 1:
                    probs_pos = torch.sigmoid(raw_pred)
                    processed_pred = torch.stack([1 - probs_pos, probs_pos], dim=1)
                elif raw_pred.dim() == 2 and raw_pred.shape[1] == 2:
                    processed_pred = raw_pred
                else:
                    processed_pred = raw_pred.reshape(-1, num_classes)
            else:
                processed_pred = raw_pred.reshape(-1, num_classes)
            processed_label = raw_label.squeeze() if raw_label.dim() > 1 else raw_label
        elif task_type == 'regression':
            pred_flat = raw_pred.reshape(-1)
            label_flat = raw_label.reshape(-1)
            loaded_scaler = list(joblib.load(scaler_path / 'target_scalers.pkl').values())[0]
            pred_flat = torch.from_numpy(loaded_scaler.inverse_transform(pred_flat.numpy().reshape(-1, 1))).reshape(-1)
            label_flat = torch.from_numpy(loaded_scaler.inverse_transform(label_flat.numpy().reshape(-1, 1))).reshape(-1)
            if clamp_min is not None and clamp_max is not None:
                pred_flat = torch.clamp(pred_flat, clamp_min, clamp_max)
            processed_pred = pred_flat
            processed_label = label_flat
        elif task_type == 'multiclass_classification':
            processed_pred = raw_pred.reshape(-1, num_classes)
            processed_label = raw_label.squeeze() if raw_label.dim() > 1 else raw_label
        else:
            processed_pred = raw_pred
            processed_label = raw_label

        processed_pred = processed_pred.detach()
        processed_label = processed_label.detach()
        metric_pred = processed_pred.cpu()
        metric_label = processed_label.cpu()

        metrics: Dict[str, float] = {}
        for metric_name, metric_fn in metric_fns.items():
            try:
                metrics[metric_name] = metric_fn(None, metric_pred, metric_label).item()
            except Exception as e:
                print(f"Warning: Could not calculate {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        if return_predictions:
            return metrics, metric_pred, metric_label
        return metrics
    
    # Training loop with early stopping
    num_epochs = model_config.get('num_epochs', 20)
    patience = model_config.get('patience', 5)
    min_delta = model_config.get('min_delta', 0.001)

    test_predictions_tensor: Optional[torch.Tensor] = None
    test_ground_truth_tensor: Optional[torch.Tensor] = None

    # Determine primary metric for early stopping
    if task_type == 'classification':
        primary_metric = 'accuracy' if num_classes > 2 else 'auroc'
    elif task_type == 'binary_classification':
        primary_metric = 'auroc'
    elif task_type == 'multiclass_classification':
        primary_metric = 'accuracy'
    else:
        primary_metric = 'mae'

    # Initialize validation metrics tracker
    metrics_tracker = ValidationMetricsTracker(
        task_type=task_type,
        primary_metric=primary_metric,
        patience=patience,
        min_delta=min_delta
    )

    # Store all epoch data for checkpoint selection
    all_test_metrics = []
    all_models = []

    print(f"Training {type(model).__name__} on {task_name}")
    print(f"Task type: {task_type} ({num_classes} classes)" if task_type == 'classification' else f"Task type: {task_type}")
    print(f"Device: {device}")
    print(f"Primary metric: {primary_metric}")
    if task_type == 'regression':
        print(f"Checkpoint selection: 50% MAE + 50% R2 (normalized)")
    else:
        print(f"Checkpoint selection: {primary_metric}")
    print(f"Early stopping patience: {patience}, min_delta: {min_delta}")
    print("-" * 80)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(epoch)
        val_metrics = evaluate(val_loader)
        test_metrics = evaluate(test_loader)

        # Add validation metrics to tracker
        metrics_tracker.add_epoch_metrics(val_metrics)

        # Store test metrics and model for this epoch
        all_test_metrics.append(test_metrics)
        all_models.append(deepcopy(model).cpu())

        # Print metrics
        print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}')
        print(f'  Val:   {", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])}')
        print(f'  Test:  {", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])}')

        # Check early stopping
        if metrics_tracker.check_early_stopping():
            print(f'\nEarly stopping triggered at epoch {epoch}!')
            break

        print()

    # Select best checkpoint based on validation metrics
    best_epoch_idx = metrics_tracker.get_best_epoch()
    best_val_metrics = metrics_tracker.get_best_metrics()
    best_test_metrics = all_test_metrics[best_epoch_idx]
    best_model = all_models[best_epoch_idx]

    print("-" * 80)
    print(f'Best epoch: {best_epoch_idx + 1}')
    print(f'Best Val Metrics: {", ".join([f"{k}: {v:.4f}" for k, v in best_val_metrics.items()])}')
    print(f'Best Test Metrics: {", ".join([f"{k}: {v:.4f}" for k, v in best_test_metrics.items()])}')
    print(f'Training stopped at epoch: {epoch}')

    if return_predictions:
        inference_model = deepcopy(best_model).cpu() if best_model is not None else deepcopy(model).cpu()
        inference_model = inference_model.to(device)
        metrics_with_preds = evaluate(
            test_loader,
            return_predictions=True,
            model_override=inference_model,
        )
        if isinstance(metrics_with_preds, tuple):
            best_test_metrics, test_predictions_tensor, test_ground_truth_tensor = metrics_with_preds
        inference_model = inference_model.cpu()
        if best_model is None:
            best_model = inference_model

    if save_model and best_model is not None:
        torch.save(best_model.state_dict(), model_weight_name)

    return {
        'best_val_metrics': best_val_metrics,
        'best_test_metrics': best_test_metrics,
        'primary_metric': primary_metric,
        'task_type': task_type,
        'num_classes': num_classes,
        'early_stopping_triggered': metrics_tracker.should_stop,
        'best_val_score': metrics_tracker.best_combined_score,
        'best_epoch': best_epoch_idx + 1,
        'epochs_trained': epoch,
        'patience': patience,
        'min_delta': min_delta,
        'model_weight_name': model_weight_name,
        'test_predictions': test_predictions_tensor.cpu() if test_predictions_tensor is not None else None,
        'test_ground_truth': test_ground_truth_tensor.cpu() if test_ground_truth_tensor is not None else None,
    }

def lightgbm_solution(
    dfs_dataset: DBBRDBDataset,
    db_name: str,
    task_name: str,
    model_config: dict,
    sample_size: int = 50000,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    return_predictions: bool = False,
):
    """Train a LightGBM model on DFS-generated dataset.
    
    Args:
        dfs_dataset: DFS-generated dataset
        task_name: Name of the task
        model_config: Configuration dictionary for the model
    
    Returns:
        dict: Training results with metrics
    """
    try:
        scaler_path = dfs_dataset.scaler_path
    except:
        scaler_path = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the task from dataset
    task = dfs_dataset.tasks[0]  # Assuming single task for now
    task_type, num_classes = check_task_type(task)
    # is_classification = task_type == 'classification'

    if task.metadata.task_type == 'binary_classification' or (task.metadata.task_type == 'classification' and num_classes > 2):
        task_type = 'multiclass_classification'

    # Convert datasets to torch_frame format
    train_ds, val_ds, test_ds = convert_all_splits(
        dbb_dataset=dfs_dataset,
        task_name=task_name,
        task_type=task_type,
        transform="none",
        cache_dir="./datasets",
        sample_size=sample_size,
        scaler_path=scaler_path
    )




    if task_type in [
        'binary_classification', 'classification'
    ]:
        tune_metric = Metric.ROCAUC
    elif task_type == 'regression':
        tune_metric = Metric.MAE
    elif task_type == 'multiclass_classification':
        tune_metric = Metric.ACCURACY
    else:
        raise ValueError(f"Task task type is unsupported {task.task_type}")
    
    task_type = {
        'binary_classification': TFTaskType.BINARY_CLASSIFICATION,
        'classification': TFTaskType.BINARY_CLASSIFICATION,
        'regression': TFTaskType.REGRESSION,
        'multiclass_classification': TFTaskType.MULTICLASS_CLASSIFICATION,
        'multilabel_classification': TFTaskType.MULTILABEL_CLASSIFICATION,
    }[task_type]

    _, num_classes = check_task_type(task)

    model = LightGBM(
        task_type=task_type,
        metric=tune_metric,
        probability=True,
        num_classes=(
            num_classes
            if task_type == TFTaskType.MULTICLASS_CLASSIFICATION
            else None
        ),
    )
    model.tune(tf_train=train_ds.tensor_frame, tf_val=val_ds.tensor_frame, num_trials=model_config.get('num_trials', 10))

    train_pred = model.predict(tf_test=train_ds.tensor_frame.cpu())
    val_pred = model.predict(tf_test=val_ds.tensor_frame.cpu())
    test_pred = model.predict(tf_test=test_ds.tensor_frame.cpu())
    test_predictions_tensor: Optional[torch.Tensor] = None
    if return_predictions:
        if isinstance(test_pred, torch.Tensor):
            test_predictions_tensor = test_pred.detach().cpu()
        else:
            test_predictions_tensor = torch.as_tensor(test_pred)
        label_tensor = test_ds.tensor_frame.y
        if isinstance(label_tensor, torch.Tensor):
            label_tensor = label_tensor.detach().cpu()
        else:
            label_tensor = torch.as_tensor(label_tensor)
        if task_type == TFTaskType.REGRESSION and scaler_path is not None:
            try:
                loaded_scaler = list(joblib.load(scaler_path / 'target_scalers.pkl').values())[0]
                label_np = label_tensor.numpy().reshape(-1, 1)
                label_tensor = torch.from_numpy(
                    loaded_scaler.inverse_transform(label_np)
                ).reshape(-1)
            except Exception as exc:
                logger.warning(f"Failed to inverse transform regression labels for LightGBM: {exc}")
        else:
            label_tensor = label_tensor.reshape(-1)
        test_ground_truth_tensor = label_tensor
    else:
        test_ground_truth_tensor = None
        
    # Evaluation function
    @torch.no_grad()
    def evaluate(pred, label, task_type) -> Dict[str, float]:
        # Get appropriate metrics for the task
        metric_fns = get_metrics(task_type, num_classes)
        
        # Calculate metrics
        metrics = {}
        for metric_name, metric_fn in metric_fns.items():
            try:
                if task_type == 'classification' or task_type == 'binary_classification':
                    # For classification, ensure proper shapes
                    if num_classes == 2:
                        # Binary classification: for torchmetrics, we need probabilities for both classes
                        if pred.dim() == 2 and pred.shape[1] == 1:
                            # Convert single logit to 2-class probabilities
                            probs_neg = torch.sigmoid(-pred.squeeze(1))  # P(class=0)
                            probs_pos = torch.sigmoid(pred.squeeze(1))   # P(class=1)
                            pred = torch.stack([probs_neg, probs_pos], dim=1)
                        elif pred.dim() == 1:
                            # Convert single logit to 2-class probabilities
                            probs_neg = torch.sigmoid(-pred)  # P(class=0)
                            probs_pos = torch.sigmoid(pred)   # P(class=1)
                            pred = torch.stack([probs_neg, probs_pos], dim=1)
                        elif pred.dim() == 2 and pred.shape[1] == 2:
                            # Already in correct format
                            pass
                        else:
                            pred = pred.reshape(-1, num_classes)
                        
                        # Ensure label is 1D
                        if label.dim() > 1:
                            label = label.squeeze()
                    # else:
                    #     # Multi-class classification: pred should be (N, num_classes), label should be (N,)
                    #     pred = pred.reshape(-1, num_classes)
                    #     if label.dim() > 1:
                    #         label = label.squeeze()
                    
                    metrics[metric_name] = metric_fn(None, pred.cpu(), label.cpu()).item()
                elif task_type == 'regression':
                    # Regression: both pred and label should be 1D
                    pred_flat = pred.reshape(-1).cpu()
                    label_flat = label.reshape(-1).cpu()
                    ## when do regression, use sklearn's standard scaler's reverse transform to get the original scale
                    loaded_scaler = list(joblib.load(scaler_path / 'target_scalers.pkl').values())[0]
                    pred_flat = torch.from_numpy(loaded_scaler.inverse_transform(pred_flat.reshape(-1, 1))).reshape(-1)
                    label_flat = torch.from_numpy(loaded_scaler.inverse_transform(label_flat.reshape(-1, 1))).reshape(-1)
                    if clamp_min is not None and clamp_max is not None:
                        pred_flat = torch.clamp(pred_flat, clamp_min, clamp_max)
                    metrics[metric_name] = metric_fn(None, pred_flat.cpu(), label_flat.cpu()).item()
                elif task_type == 'multiclass_classification':
                    pred = pred.reshape(-1, num_classes)
                    if label.dim() > 1:
                        label = label.squeeze()
                    metrics[metric_name] = metric_fn(None, pred.cpu(), label.cpu()).item()
                else:
                    # Other task types
                    metrics[metric_name] = metric_fn(None, pred.cpu(), label.cpu()).item()
            except Exception as e:
                print(f"Warning: Could not calculate {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        return metrics

    train_metrics = evaluate(train_pred, train_ds.tensor_frame.y, task_type.value)
    val_metrics = evaluate(val_pred, val_ds.tensor_frame.y, task_type.value)
    test_metrics = evaluate(test_pred, test_ds.tensor_frame.y, task_type.value)
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_predictions': test_predictions_tensor,
        'test_ground_truth': test_ground_truth_tensor,
    }


def create_lightgbm_config(
    num_trials: int = 10,
    sample_size: int = 200000,
    seed: int = 42,
    **kwargs
) -> dict:
    """Create a LightGBM configuration dictionary.
    
    Args:
        num_trials: Number of hyperparameter tuning trials
        sample_size: Number of training samples to use (0 for all)
        seed: Random seed
        **kwargs: Additional configuration parameters
        
    Returns:
        dict: Configuration dictionary
    """
    config = {
        'num_trials': num_trials,
        'sample_size': sample_size,
        'seed': seed,
    }
    config.update(kwargs)
    return config


def create_resnet_model(train_dataset: DatasetwithTransform, model_config: dict, 
                        task_type: str, num_classes: int):
    """Create a ResNet model for the converted dataset.
    
    Args:
        train_dataset: Converted training dataset
        model_config: Model configuration dictionary
        task_type: Type of task ('classification', 'regression')
        num_classes: Number of classes (1 for regression, >=2 for classification)
        
    Returns:
        Configured ResNet model
    """
    from torch_frame.nn import ResNet
    
    normalization = {
        'layernorm': 'layer_norm',
        'batchnorm': 'batch_norm',
        'none': None, 
        None: None,
    }[model_config['normalization']]
    
    # Determine output channels
    if task_type == 'regression':
        out_channels = 1
    else:
        # For classification, always use num_classes (2 for binary, >2 for multiclass)
        out_channels = num_classes

    stype_encoder_dict = {
        key: value[0](**value[1]) for key, value in DEFAULT_STYPE_ENCODER_DICT.items()
    }
    model = ResNet(
        channels=model_config['channels'],
        out_channels=out_channels,
        num_layers=model_config['num_layers'],
        col_stats=train_dataset.col_stats,
        col_names_dict=train_dataset.tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict, 
        normalization=normalization,
        dropout_prob=model_config['dropout_prob'],
    )
    
    return model


def create_ft_transformer_model(train_dataset: DatasetwithTransform, model_config: dict, 
                                task_type: str, num_classes: int):
    """Create an FT-Transformer model for the converted dataset.
    
    Args:
        train_dataset: Converted training dataset
        model_config: Model configuration dictionary
        task_type: Type of task ('classification', 'regression')
        num_classes: Number of classes (1 for regression, >=2 for classification)
        
    Returns:
        Configured FT-Transformer model
    """
    from torch_frame.nn import FTTransformer
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # Determine output channels
    if task_type == 'regression':
        out_channels = 1
    else:
        # For classification, always use num_classes (2 for binary, >2 for multiclass)
        out_channels = num_classes if num_classes > 2 else 1

    stype_encoder_dict = {
        key: value[0](**value[1]) for key, value in DEFAULT_STYPE_ENCODER_DICT.items()
    }
    
    model = FTTransformer(
        channels=model_config['channels'],
        out_channels=out_channels,
        num_layers=model_config['num_layers'],
        col_stats=train_dataset.col_stats,
        col_names_dict=train_dataset.tensor_frame.col_names_dict,
        stype_encoder_dict=stype_encoder_dict
    )
    
    return model


if __name__ == "__main__":
    # Run example (commented out to avoid execution during import)
    # example_conversion()
    pass
