from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import logging
import hashlib
import pickle
import os
from pathlib import Path
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class GloveTextEmbeddingTransformConfig(pydantic.BaseModel):
    model_name : str = "sentence-transformers/average_word_embeddings_glove.6B.300d"  # SentenceTransformers model with 384 dim, we'll project to 100
    dim : int = 300
    max_num_procs : int = 1
    cache_dir : str = "embeddingcache"


def _get_cache_path(cache_dir, embedding_name, num_rows):
    """Generate cache file path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    filename = f"{embedding_name}_{num_rows}.pkl"
    return cache_dir / filename

def _run_one_proc(proc_id, model, dim, data, projection_matrix=None):
    """Process text data in one process."""
    embeddings = []
    if proc_id == 0:
        iterator = tqdm(data, desc="Generating embeddings")
    else:
        iterator = data
    
    for text in iterator:
        if not isinstance(text, str) or len(str(text).strip()) == 0:
            embed = np.zeros(dim)
        else:
            # Use SentenceTransformers to encode
            embed = model.encode([str(text)])[0]  # Returns 384-dim vector
            # Project to target dimension (100) if projection matrix provided
            if projection_matrix is not None:
                embed = embed @ projection_matrix
        embeddings.append(embed)
    
    return np.stack(embeddings).astype('float32')

@column_transform
class GloveTextEmbeddingTransform(ColumnTransform):
    config_class = GloveTextEmbeddingTransformConfig
    name = "glove_text_embedding"
    input_dtype = DBBColumnDType.text_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : GloveTextEmbeddingTransformConfig):
        super().__init__(config)
        logger.info(f"Loading SentenceTransformer model: {config.model_name}")
        self.model = SentenceTransformer(config.model_name)
        
        # Create projection matrix to reduce from 384 to target dimension (100)
        self.relbench_embedding_cache_path = os.environ.get("RELBENCH_EMBEDDING_CACHE_PATH", "cache_data/embedding")
        model_dim = self.model.get_sentence_embedding_dimension()
        if model_dim != config.dim:
            logger.info(f"Creating projection matrix from {model_dim} to {config.dim}")
            # Use random projection matrix (can be improved with PCA later)
            np.random.seed(42)  # For reproducibility
            self.projection_matrix = np.random.normal(0, 1/np.sqrt(model_dim), 
                                                    (model_dim, config.dim)).astype('float32')
        else:
            self.projection_matrix = None

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        self.new_meta = {
            'dtype' : self.output_dtypes[0],
            'in_size' : self.config.dim,
        }

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo,
        metadata : Dict[str, str] = {}
    ) -> List[ColumnData]:
        data = column.data
        
        # Get table and column names for caching
        embedding_name = column.metadata.get('name', 'unknown')
        num_rows = len(data)
        
        cache_path = _get_cache_path(self.config.cache_dir, embedding_name, num_rows)
        
        if cache_path.exists():
            logger.info(f"Loading cached embeddings from: {cache_path}")
            with open(cache_path, 'rb') as f:
                new_data = pickle.load(f)
        else:
            logger.info(f"Generating new embeddings for {embedding_name} ({num_rows} rows)")

            potential_relbench_cached_one = os.path.join(self.relbench_embedding_cache_path, f"{metadata['rdb_name']}_{embedding_name}_{num_rows}.pt")


            ## create a soft link to the relbench cached one
            if os.path.exists(potential_relbench_cached_one):
                logger.info(f"Using relbench cached embeddings from: {potential_relbench_cached_one}")
                new_data = torch.load(potential_relbench_cached_one, map_location='cpu').numpy()
                return [ColumnData(self.new_meta, new_data)]
            
            # Generate embeddings
            num_procs = min(device.cpu_count // 2, self.config.max_num_procs)
            if num_procs > 1:
                logger.info("Spawn workers to generate embeddings using multi-CPU kernels.")
                ctx = mp.get_context('spawn')
                worklist = np.array_split(data, num_procs)
                with ctx.Pool(processes=num_procs) as pool:
                    results = []
                    for proc_id in range(num_procs):
                        rst = pool.apply_async(
                            _run_one_proc,
                            (proc_id, self.model, self.config.dim, worklist[proc_id], self.projection_matrix)
                        )
                        results.append(rst)
                    results = [rst.get() for rst in results]
                new_data = np.concatenate(results, axis=0)
            else:
                new_data = _run_one_proc(0, self.model, self.config.dim, data, self.projection_matrix)
            
            # Cache the results
            logger.info(f"Caching embeddings to: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(new_data, f)
        
        return [ColumnData(self.new_meta, new_data)]