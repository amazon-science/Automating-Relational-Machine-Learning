from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import logging
import pickle
import hashlib
from pathlib import Path
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from .base import (
    rdb_transform,
    ColumnData,
    RDBData,
    RDBTransform,
)
from models.dfs.helper import gpu_ipca_two_pass

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class FloatCompressConfig(pydantic.BaseModel):
    compress_threshold: float = 80
    compress_target: float = 8
    cache_dir: str = "float_compress_cache"
    l2_normalize: bool = True
    batch_size: int = 4000

def _identify_float_columns(rdb_data: RDBData, threshold: float) -> List[Tuple[str, str]]:
    """Identify columns that are text embeddings (float columns with text origin)."""
    float_tables = {}
    total_float_dim = {}
    non_float_columns = {}
    for table_name, table in rdb_data.tables.items():
        table_total_float_dim = 0
        matched_columns = []
        non_floats = []
        for col_name, col_data in table.items():
            # Look for float columns that likely came from text embeddings
            if (col_data.metadata.get('dtype') == DBBColumnDType.float_t) and '_pca' not in col_name:
                ## we don't want to compress pca-ed columns
                if col_data.data.ndim == 1:
                    table_total_float_dim += 1
                else:
                    table_total_float_dim += col_data.data.shape[1]
                matched_columns.append(col_name)
            else:
                non_floats.append(col_name)
        non_float_columns[table_name] = non_floats
        total_float_dim[table_name] = table_total_float_dim
        if table_total_float_dim > threshold:
            float_tables[table_name] = matched_columns
    return float_tables, total_float_dim, non_float_columns

def _get_column_embeddings(rdb_data: RDBData, table_name: str, col_names: List[str]) -> np.ndarray:
    """Get embeddings for a specific column."""
    total_embeddings = []
    for col_name in col_names:
        embeddings = rdb_data.tables[table_name][col_name].data
        if embeddings.ndim == 1:
            # Handle case where embeddings are flattened
            logger.warning(f"Column {table_name}.{col_name} has 1D data, assuming single embedding per row")
            if not isinstance(embeddings, np.ndarray):
                embeddings = embeddings.values
            ## fill nan with mean otherwise pca can't work
            nanmean = np.nanmean(embeddings)
            embeddings[np.isnan(embeddings)] = nanmean
            embeddings = embeddings.reshape(-1, embeddings.shape[0] // len(embeddings) if len(embeddings) > 0 else 1)
        total_embeddings.append(embeddings)
    return np.concatenate(total_embeddings, axis=1)



def _make_batch_iterator(data: np.ndarray, batch_size: int):
    """Create a batch iterator function for the data."""
    def batch_iter():
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    return batch_iter

def _get_pca_cache_path(cache_dir: str, table_name: str, config: FloatCompressConfig) -> Path:
    """Generate cache file path for PCA model per column."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # Create filename with table, column, rows, and key config parameters
    config_suffix = f"{config.compress_threshold}to{config.compress_target}"
    if config.l2_normalize:
        config_suffix += "_l2norm"
    
    filename = f"float_compress_{table_name}_{config_suffix}.pkl"
    return cache_dir / filename

@rdb_transform
class FloatCompressTransform(RDBTransform):
    config_class = FloatCompressConfig
    name = "float_compress"

    def __init__(self, config: FloatCompressConfig):
        super().__init__(config)
        self.processed_columns = []
    

    def fit(
        self,
        rdb_data: RDBData,
        device: DeviceInfo
    ) -> None:
        logger.info("Fitting FloatCompress transform")
        
        # Only identify text embedding columns, don't fit PCA models yet
        self.processed_tables, self.total_float_dim, self.non_float_columns = _identify_float_columns(rdb_data, self.config.compress_threshold)
        
        if not self.processed_tables:
            logger.warning("No float columns found, FloatCompress transform will be a no-op")
            return
        
        logger.info(f"Found {len(self.processed_tables)} tables to process: {self.processed_tables}")

    def transform(
        self,
        rdb_data: RDBData,
        device: DeviceInfo,
        metadata : Dict[str, str] = {}
    ) -> RDBData:
        logger.info("Applying FloatCompress transform")
        
        if not self.processed_tables:
            logger.info("No float tables found, returning original data")
            return rdb_data
        
        # Create new RDB data with transformed embeddings
        new_tables = {}
        
        for table_name, table in rdb_data.tables.items():
            new_table = {}
            
            if table_name in self.processed_tables:
                col_names = self.processed_tables[table_name]
                # Get original embeddings for this specific column
                embeddings = _get_column_embeddings(rdb_data, table_name, col_names)
                    
                    
                # Check cache for this specific column's PCA result
                cache_path = _get_pca_cache_path(
                    self.config.cache_dir, table_name, 
                    self.config
                )
                    
                # Apply PCA transformation (gpu_ipca_two_pass returns transformed data directly)
                transformed = None
                if cache_path.exists():
                    logger.info(f"Loading cached PCA result from: {cache_path}")
                    try:
                        with open(cache_path, 'rb') as f:
                            transformed = pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load cached result: {e}")
                
                if transformed is None:
                    logger.info(f"Computing PCA for table {table_name}: {embeddings.shape[1]}D -> {self.config.compress_target}D")
                    
                    make_batches = _make_batch_iterator(embeddings, self.config.batch_size)
                    
                    try:
                        transformed = gpu_ipca_two_pass(
                            make_batches,
                            n_components=self.config.compress_target,
                            l2_normalize_rows=self.config.l2_normalize,
                            n_samples=len(embeddings)
                        )
                        
                        # Cache the transformed result
                        logger.info(f"Caching PCA result to: {cache_path}")
                        with open(cache_path, 'wb') as f:
                            pickle.dump(transformed, f)
                            
                    except Exception as e:
                        logger.error(f"Failed to compute PCA for {table_name}: {e}")
                        logger.info("Falling back to original data")
                        continue
                
                # Split PCA result into multiple single-dimension columns
                for pca_dim in range(transformed.shape[1]):
                    # Create new column name with PCA dimension suffix
                    pca_col_name = f"{table_name}_pca_{pca_dim}"
                    
                    # Extract single dimension
                    single_dim_data = transformed[:, pca_dim].reshape(-1)
                    
                    # Update metadata for each PCA dimension
                    pca_metadata = {}
                    pca_metadata['in_size'] = 1
                    pca_metadata['pca_reduced'] = True
                    pca_metadata['original_dim'] = embeddings.shape[1]
                    pca_metadata['pca_component'] = pca_dim
                    pca_metadata['dtype'] = DBBColumnDType.float_t
                    new_table[pca_col_name] = ColumnData(pca_metadata, single_dim_data)
                
                # Remove the original multi-dimensional embedding column
                # (don't add it to new_table)
                for col_name in self.non_float_columns[table_name]:
                    new_table[col_name] = table[col_name]
            else:
                # Keep original column unchanged
                new_table = table
        
            new_tables[table_name] = new_table
        
        result = RDBData(
            tables=new_tables,
            column_groups=rdb_data.column_groups,
            relationships=rdb_data.relationships
        )
        if hasattr(rdb_data, 'val_timestamp_cutoff'):
            result.val_timestamp_cutoff = rdb_data.val_timestamp_cutoff
        if hasattr(rdb_data, 'test_timestamp_cutoff'):
            result.test_timestamp_cutoff = rdb_data.test_timestamp_cutoff
        if hasattr(rdb_data, 'time_column_name'):
            result.time_column_name = rdb_data.time_column_name
        return result