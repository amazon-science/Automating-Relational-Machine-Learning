from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
try:
    import cupy as cp
    from cuml.decomposition import PCA
except ImportError:
    cp = None
    PCA = None
import logging
import pickle
import hashlib
from pathlib import Path
from dbinfer_bench import DBBColumnDType
import pandas as pd

from ...device import DeviceInfo
from .base import (
    rdb_transform,
    ColumnData,
    RDBData,
    RDBTransform,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class TextPCATransformConfig(pydantic.BaseModel):
    input_dim: int = 300          # Expected input embedding dimension
    output_dim: int = 3           # Target PCA output dimension  
    l2_normalize: bool = True     # L2 normalize embeddings before PCA
    whiten: bool = False          # Whether to whiten the components
    cache_dir: str = "pca_cache"  # Directory for caching fitted models
    batch_size: int = 1000        # Batch size for PCA processing

def _identify_text_embedding_columns(rdb_data: RDBData) -> List[Tuple[str, str]]:
    """Identify columns that are text embeddings (float columns with text origin)."""
    text_embedding_columns = []
    for table_name, table in rdb_data.tables.items():
        for col_name, col_data in table.items():
            # Look for float columns that likely came from text embeddings
            if (col_data.metadata.get('dtype') == DBBColumnDType.float_t and
                col_data.data.ndim > 1 and col_data.data.shape[1] == 300):
                text_embedding_columns.append((table_name, col_name))
    return text_embedding_columns

def _get_column_embeddings(rdb_data: RDBData, table_name: str, col_name: str) -> np.ndarray:
    """Get embeddings for a specific column."""
    embeddings = rdb_data.tables[table_name][col_name].data
    if embeddings.ndim == 1:
        # Handle case where embeddings are flattened
        logger.warning(f"Column {table_name}.{col_name} has 1D data, assuming single embedding per row")
        embeddings = embeddings.reshape(-1, embeddings.shape[0] // len(embeddings) if len(embeddings) > 0 else 1)
    return embeddings


def _get_train_test_indices(
    rdb_data: RDBData,
    table_name: str,
    num_rows: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Get indices for train+val (pre-test) and test (post-test) data based on timestamp cutoff.
    Note: In this setting, validation data is visible, so we use test_timestamp_cutoff.

    Returns:
        train_indices: Indices of rows before test_timestamp_cutoff (train+val)
        test_indices: Indices of rows after test_timestamp_cutoff (None if no cutoff or time column)
    """
    # Use test_timestamp_cutoff to avoid data leakage (validation is visible in this setting)
    # This ensures PCA is fit on train+val data, then applied to test data
    cutoff = rdb_data.test_timestamp_cutoff
    if cutoff is None:
        logger.warning(f"No test_timestamp_cutoff found, using all data for training. This will cause data leakage!")
        return np.arange(num_rows), None

    # Check if this table has a time column
    if rdb_data.time_column_name is None or table_name not in rdb_data.time_column_name:
        logger.info(f"No time column found for table {table_name}, using all data for training")
        return np.arange(num_rows), None

    time_col_name = rdb_data.time_column_name[table_name]

    # Get the time column data
    if time_col_name not in rdb_data.tables[table_name]:
        logger.warning(f"Time column {time_col_name} not found in table {table_name}")
        return np.arange(num_rows), None

    time_data = rdb_data.tables[table_name][time_col_name].data

    # Normalize both time_data and cutoff to the same type for comparison
    # Check if time_data is pandas datetime or numpy datetime64
    if hasattr(time_data, 'dtype') and np.issubdtype(time_data.dtype, np.datetime64):
        # time_data is datetime64, convert cutoff to datetime64
        if isinstance(cutoff, str):
            try:
                cutoff = pd.Timestamp(cutoff)
            except Exception as e:
                logger.warning(f"Could not parse test_timestamp_cutoff: {e}, using all data")
                return np.arange(num_rows), None
        elif isinstance(cutoff, (int, np.integer)):
            # Convert int64 timestamp to datetime
            cutoff = pd.Timestamp(cutoff)
        # If already a Timestamp, use as-is

    else:
        # time_data is numeric (int64), convert cutoff to int64
        if isinstance(cutoff, str):
            try:
                cutoff = pd.Timestamp(cutoff).value
            except Exception as e:
                logger.warning(f"Could not parse test_timestamp_cutoff: {e}, using all data")
                return np.arange(num_rows), None
        elif hasattr(cutoff, 'value'):
            # Already a Timestamp object, get the int64 value
            cutoff = cutoff.value
        # If already int64, use as-is

    # Split indices based on timestamp
    train_mask = time_data < cutoff
    test_mask = time_data >= cutoff

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    logger.info(f"Split data for {table_name}: {len(train_indices)} train+val, {len(test_indices)} test rows (cutoff: {rdb_data.test_timestamp_cutoff})")

    if len(test_indices) == 0:
        return train_indices, None

    return train_indices, test_indices

def _get_pca_cache_path(
    cache_dir: str,
    table_name: str,
    col_name: str,
    num_train_rows: int,
    num_total_rows: int,
    config: TextPCATransformConfig
) -> Path:
    """
    Generate cache file path for PCA model per column.
    Includes both train and total rows to ensure cache invalidation when data split changes.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # Create filename with table, column, rows, and key config parameters
    config_suffix = f"{config.input_dim}to{config.output_dim}"
    if config.l2_normalize:
        config_suffix += "_l2norm"
    if config.whiten:
        config_suffix += "_whiten"

    # Include both train and total rows to ensure proper cache invalidation
    filename = f"pca_{table_name}_{col_name}_train{num_train_rows}_total{num_total_rows}_{config_suffix}.pkl"
    return cache_dir / filename

@rdb_transform
class TextPCATransform(RDBTransform):
    config_class = TextPCATransformConfig
    name = "text_pca"

    def __init__(self, config: TextPCATransformConfig):
        super().__init__(config)
        self.processed_columns = []
    

    def fit(
        self,
        rdb_data: RDBData,
        device: DeviceInfo
    ) -> None:
        logger.info("Fitting TextPCA transform")
        
        # Only identify text embedding columns, don't fit PCA models yet
        self.processed_columns = _identify_text_embedding_columns(rdb_data)
        
        if not self.processed_columns:
            logger.warning("No text embedding columns found, PCA transform will be a no-op")
            return
        
        logger.info(f"Found {len(self.processed_columns)} text embedding columns to process: {self.processed_columns}")

    def transform(
        self,
        rdb_data: RDBData,
        device: DeviceInfo,
        metadata : Dict[str, str] = {}
    ) -> RDBData:
        logger.info("Applying TextPCA transform with time-aware fitting")

        if not self.processed_columns:
            logger.info("No text embedding columns found, returning original data")
            return rdb_data

        # Create new RDB data with transformed embeddings
        new_tables = {}

        for table_name, table in rdb_data.tables.items():
            new_table = {}

            if len(table) == 0:
                logger.warning(f"Table {table_name} has no columns, skipping")
                new_tables[table_name] = table
                continue

            first_col_name = next(iter(table.keys()))

            # Get train+val/test split indices based on timestamp
            train_indices, test_indices = _get_train_test_indices(
                rdb_data, table_name, table[first_col_name].data.shape[0]
            )

            for col_name, col_data in table.items():
                if (table_name, col_name) in self.processed_columns:
                    logger.info(f"Transforming column {table_name}.{col_name}")

                    # Get original embeddings for this specific column
                    embeddings = _get_column_embeddings(rdb_data, table_name, col_name)

                    # Validate input dimension
                    if embeddings.shape[1] != self.config.input_dim:
                        logger.warning(
                            f"Expected input dimension {self.config.input_dim}, got {embeddings.shape[1]} for {table_name}.{col_name}"
                        )

                    

                    # Check cache for this specific column's PCA result
                    cache_path = _get_pca_cache_path(
                        self.config.cache_dir, table_name, col_name,
                        len(train_indices), embeddings.shape[0], self.config
                    )

                    # Apply PCA transformation with time-aware splitting
                    transformed = None
                    if cache_path.exists():
                        logger.info(f"Loading cached PCA result from: {cache_path}")
                        try:
                            with open(cache_path, 'rb') as f:
                                transformed = pickle.load(f)
                        except Exception as e:
                            logger.warning(f"Failed to load cached result: {e}")

                    if transformed is None:
                        logger.info(f"Computing time-aware PCA for {table_name}.{col_name}: {embeddings.shape[1]}D -> {self.config.output_dim}D")
                        logger.info(f"Train+Val samples: {len(train_indices)}, Test samples: {len(test_indices) if test_indices is not None else 0}")

                        try:
                            # Get training+validation data (pre-test timestamp)
                            train_embeddings = embeddings[train_indices]

                            # Convert to GPU
                            train_embeddings_gpu = cp.asarray(train_embeddings, dtype=cp.float32)

                            # L2 normalize if requested
                            if self.config.l2_normalize:
                                norms = cp.linalg.norm(train_embeddings_gpu, axis=1, keepdims=True) + 1e-12
                                train_embeddings_gpu = train_embeddings_gpu / norms

                            # Fit PCA on train+val data (CRITICAL: validation is visible, test is not)
                            pca = PCA(
                                n_components=self.config.output_dim,
                                whiten=self.config.whiten
                            )
                            pca.fit(train_embeddings_gpu)

                            # Transform training+validation data
                            train_transformed = pca.transform(train_embeddings_gpu)
                            train_transformed = cp.asnumpy(train_transformed).astype(np.float32)

                            # Transform test data if it exists
                            if test_indices is not None and len(test_indices) > 0:
                                test_embeddings = embeddings[test_indices]
                                test_embeddings_gpu = cp.asarray(test_embeddings, dtype=cp.float32)

                                # L2 normalize if requested
                                if self.config.l2_normalize:
                                    norms = cp.linalg.norm(test_embeddings_gpu, axis=1, keepdims=True) + 1e-12
                                    test_embeddings_gpu = test_embeddings_gpu / norms

                                # Transform using fitted PCA
                                test_transformed = pca.transform(test_embeddings_gpu)
                                test_transformed = cp.asnumpy(test_transformed).astype(np.float32)

                                # Concatenate in original order
                                transformed = np.empty((embeddings.shape[0], self.config.output_dim), dtype=np.float32)
                                transformed[train_indices] = train_transformed
                                transformed[test_indices] = test_transformed
                            else:
                                # No test data, just use train+val transformed
                                transformed = train_transformed

                            # Cache the transformed result
                            logger.info(f"Caching PCA result to: {cache_path}")
                            with open(cache_path, 'wb') as f:
                                pickle.dump(transformed, f)

                        except Exception as e:
                            logger.error(f"Failed to compute PCA for {table_name}.{col_name}: {e}")
                            logger.info("Falling back to original data")
                            new_table[col_name] = col_data
                            continue

                    # Split PCA result into multiple single-dimension columns
                    for pca_dim in range(transformed.shape[1]):
                        # Create new column name with PCA dimension suffix
                        pca_col_name = f"{col_name}_pca_{pca_dim}"

                        # Extract single dimension
                        single_dim_data = transformed[:, pca_dim].reshape(-1)

                        # Update metadata for each PCA dimension
                        pca_metadata = col_data.metadata.copy()
                        pca_metadata['in_size'] = 1
                        pca_metadata['pca_reduced'] = True
                        pca_metadata['original_dim'] = embeddings.shape[1]
                        pca_metadata['pca_component'] = pca_dim
                        pca_metadata['original_column'] = col_name

                        new_table[pca_col_name] = ColumnData(pca_metadata, single_dim_data)

                    # Remove the original multi-dimensional embedding column
                    # (don't add it to new_table)
                else:
                    # Keep original column unchanged
                    new_table[col_name] = col_data

            new_tables[table_name] = new_table

        return RDBData(
            tables=new_tables,
            column_groups=rdb_data.column_groups,
            relationships=rdb_data.relationships,
            val_timestamp_cutoff=rdb_data.val_timestamp_cutoff,
            test_timestamp_cutoff=rdb_data.test_timestamp_cutoff,
            time_column_name=rdb_data.time_column_name
        )