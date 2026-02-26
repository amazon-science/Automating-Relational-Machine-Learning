# PyTorch-Frame: DatasetWithTransform and TensorFrame Notes

## Overview
This document provides comprehensive notes on using `DatasetWithTransform` and `TensorFrame` in PyTorch-Frame, with a focus on the caching mechanisms for embedding generation.

## 1. TensorFrame Structure

### What is TensorFrame?
- **Purpose**: A tensor frame holds PyTorch tensors for each table column, organized by semantic types (stype)
- **Core Components**:
  - `feat_dict`: Dictionary mapping semantic types to tensors
  - `col_names_dict`: Maps semantic types to column names
  - `y`: Optional target values
  - `num_rows`/`num_cols`: Dimensions

### Key Features:
- **Semantic Type Organization**: Columns are grouped by types like:
  - `torch_frame.numerical`: Numerical features
  - `torch_frame.categorical`: Categorical features  
  - `torch_frame.text_embedded`: Text embeddings
  - `torch_frame.multicategorical`: Multi-category features
  - `torch_frame.timestamp`: Time-based features

- **Missing Value Handling**: 
  - `float('NaN')` for floating-point tensors
  - `-1` for integer tensors

- **Device Management**: Built-in support for GPU/CPU transfer via `.to()`, `.cuda()`, `.cpu()`

### Usage Example:
```python
tf = torch_frame.TensorFrame(
    feat_dict = {
        torch_frame.numerical: torch.randn(10, 2),
        torch_frame.categorical: torch.randint(0, 5, (10, 3)),
    },
    col_names_dict = {
        torch_frame.numerical: ['num_1', 'num_2'],
        torch_frame.categorical: ['cat_1', 'cat_2', 'cat_3'],
    },
)
```

## 2. Dataset Base Class

### Core Functionality:
- **Initialization**: Takes DataFrame, column-to-stype mapping, target column
- **Materialization**: Converts DataFrame to TensorFrame representation
- **Caching**: Supports saving/loading materialized datasets
- **Splitting**: Built-in train/val/test split support

### Key Methods:
- `materialize(device, path, col_stats)`: Converts to tensor representation
- `get_split(split)`: Returns train/val/test subset
- `index_select(index)`: Row-wise filtering
- `col_select(cols)`: Column-wise filtering

## 3. DatasetWithTransform (Custom Implementation)

### Location: `data/relbench.py:574-742`

### Key Features:
1. **Caching for Text Embeddings**: Uses `CachedEmbeddingTensorMapper`
2. **Distribution Transforms**: Built-in statistical transformations
3. **Dataset Name Tracking**: For cache path generation

### Constructor Parameters:
```python
def __init__(
    self,
    df: DataFrame,
    col_to_stype: dict[str, torch_frame.stype],
    target_col: str | None = None,
    split_col: str | None = None,
    col_to_sep: str | None | dict[str, str | None] = None,
    col_to_text_embedder_cfg: dict[str, TextEmbedderConfig] | TextEmbedderConfig | None = None,
    col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig] | TextTokenizerConfig | None = None,
    col_to_image_embedder_cfg: dict[str, ImageEmbedderConfig] | ImageEmbedderConfig | None = None,
    col_to_time_format: str | None | dict[str, str | None] = None,
    dataset_name: str = "",  # Custom parameter for caching
)
```

### Distribution Transforms:
```python
distribution_transform = {
    torch_frame.numerical: distribution_encoder,
    torch_frame.categorical: distribution_categorical_encoder
}
```

### Materialize Method Enhancements:
- **Transform Parameter**: `transform` can be 'distribution', 'distribution_old', or 'none'
- **Caching Integration**: Passes `dataset_name` to converter for cache path generation
- **NaN Handling**: Automatic NaN filling with configurable strategies

## 4. Caching Mechanisms

### CachedEmbeddingTensorMapper

**Location**: `data/relbench.py:446-484`

**Key Features**:
- **Cache Path Generation**: `{cache_path}/{dataset_name}_{col_name}_{num_rows}.pt`
- **Automatic Loading**: Checks cache existence before embedding generation
- **Fallback**: Uses standard embedding generation if cache miss
- **Batch Processing**: Supports mini-batch embedding with progress tracking

**Implementation**:
```python
class CachedEmbeddingTensorMapper(EmbeddingTensorMapper):
    def forward(
        self,
        ser: Series,
        *,
        device: torch.device | None = None,
        dataset_name: str = "",
        col_name: str = "",
        os_cache_path: str = "/localscratch/chenzh85/relbench/embedding"
    ) -> MultiEmbeddingTensor:
        cache_path = os.path.join(os_cache_path, f"{dataset_name}_{col_name}_{ser.shape[0]}.pt")
        
        if os.path.exists(cache_path):
            print(f"Loading embedding from cache: {cache_path}")
            return torch.load(cache_path, map_location='cpu')
        else:
            # Generate embeddings and save to cache
            # ... embedding generation code ...
            torch.save(values, cache_path)
            print(f"Saving embedding to cache: {cache_path}")
        
        return MultiEmbeddingTensor(...)
```

### DataFrameToTensorFrameConverterWithCache

**Location**: `data/relbench.py:486-562`

**Key Features**:
- **Selective Caching**: Only caches `text_embedded` type columns
- **NaN Handling**: Automatic NaN filling for numerical and categorical features
- **Cache Path Integration**: Passes dataset name and column name to mappers

**NaN Filling Strategies**:
- **Numerical**: Uses `fill_nan_columnwise` with "mean" strategy
- **Categorical**: Uses `fill_nan_columnwise` with "most_frequent" strategy

## 5. Tensor Mappers

### EmbeddingTensorMapper (Base)
**Location**: `torch_frame/data/mapper.py:385-446`

**Functionality**:
- Converts raw data (text, images) to embeddings
- Supports batch processing to avoid GPU OOM
- Returns `MultiEmbeddingTensor` with proper offset tracking

### Key Methods:
- `forward(ser, device)`: Converts series to embeddings
- `backward(tensor)`: Reverse conversion (limited support)

### Other Important Mappers:
- **NumericalTensorMapper**: Handles numerical data with NaN support
- **CategoricalTensorMapper**: Maps categories to indices with -1 for missing
- **MultiCategoricalTensorMapper**: Handles multiple categories per cell
- **TimestampTensorMapper**: Converts timestamps to multiple time components

## 6. Utility Functions

### NaN Handling: `fill_nan_columnwise`
**Location**: `data/relbench.py:79-143`

**Strategies**:
- "zero": Fill with zeros
- "mean": Fill with column mean
- "median": Fill with column median
- "min"/"max": Fill with column min/max
- "value": Fill with specified value
- "most_frequent": Fill with most frequent value

### Distribution Encoders
**Location**: `data/relbench.py:145-167`

**Types**:
- `distribution_encoder`: Cumulative distribution for numerical features
- `distribution_categorical_encoder`: Probability distribution for categorical features

## 7. Best Practices

### For Embedding Caching:
1. Set meaningful `dataset_name` for cache organization
2. Use consistent cache paths across experiments
3. Monitor cache directory size for cleanup
4. Consider cache invalidation when data changes

### For Dataset Creation:
1. Define semantic types carefully before materialization
2. Use `materialize()` with path parameter for persistent caching
3. Handle missing values appropriately before tensor conversion
4. Consider memory usage with large datasets

### For Transform Usage:
1. Choose appropriate transform strategy based on data distribution
2. Apply transforms consistently across train/val/test splits
3. Cache transformed datasets for repeated experiments

## 8. Example Usage

```python
# Create dataset with caching
dataset = DatasetwithTransform(
    df=dataframe,
    col_to_stype=column_types,
    target_col="target",
    col_to_text_embedder_cfg=embedder_config,
    dataset_name="my_dataset_v1"
)

# Materialize with caching and transform
dataset = dataset.materialize(
    device=device,
    path="cached_dataset.pt",
    transform="distribution"
)

# Access tensor frame
tensor_frame = dataset.tensor_frame
print(f"Shape: {tensor_frame.num_rows} x {tensor_frame.num_cols}")
print(f"Semantic types: {tensor_frame.stypes}")

# Get specific column
col_feat = tensor_frame.get_col_feat("text_column")
```

This caching system significantly reduces embedding generation time for repeated experiments while maintaining flexibility in data transformations and semantic type handling.