from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch_frame
from loguru import logger
from relbench.base import TaskType
from relbench.datasets import get_dataset, on_disks
from relbench.tasks import get_task
from torch import Tensor
from torch_frame.data import Dataset
from torch_frame.data.dataset import (
    DataFrame,
    DataFrameToTensorFrameConverter,
    MultiEmbeddingTensor,
    MultiNestedTensor,
    TensorData,
    TensorFrame,
    TextEmbedderConfig,
    TextTokenizerConfig,
    ImageEmbedderConfig,
)
from torch_frame.data.mapper import (
    EmbeddingTensorMapper,
    TensorMapper,
    Series,
    _get_default_numpy_dtype,
    tqdm,
)
from torch_frame.data.stats import StatType, compute_col_stats
from torch_frame.utils.split import SPLIT_TO_NUM
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss

try:
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover - CPU fallback environments
    cp = None

def load_ondisk_relbench_data(dataset_name: str):
    dataset = get_dataset(dataset_name, download=False)
    dataset.name = dataset_name
    return dataset

def load_relbench_data(dataset_name: str):
    if dataset_name in on_disks:
        logger.info(f"Loading on-disk dataset {dataset_name}")
        return load_ondisk_relbench_data(dataset_name)
    dataset = get_dataset(dataset_name)
    dataset.name = dataset_name
    return dataset

def load_relbench_task(dataset_name: str, task_name: str):
    task = get_task(dataset_name, task_name)
    task.name = task_name
    task.out_channels = 1
    return task
 
def fill_nan_columnwise(tensor, strategy="mean", fill_value=None):
    """
    Fills NaN values in a tensor column-wise without using a for loop.

    Args:
        tensor (torch.Tensor): The input tensor with shape [batch_size, num_columns].
        strategy (str): The strategy to use for filling NaNs. Options are:
            "zero": Fill with zeros.
            "mean": Fill with the mean of the non-NaN values in each column.
            "median": Fill with the median of the non-NaN values in each column.
            "min": Fill with the minimum of the non-NaN values in each column.
            "max": Fill with the maximum of the non-NaN values in each column.
            "value": Fill with a specified fill_value.
            "most_frequent": Fill with the most frequent value in each column.
        fill_value (float, optional): The value to use when strategy is "value".

    Returns:
        torch.Tensor: The tensor with NaNs filled column-wise.
    """

    if not torch.is_floating_point(tensor):
        raise ValueError("Input tensor must be a float tensor.")
    
    if tensor.ndim == 1:
        tensor = tensor.view(-1, 1)

    nan_mask = torch.isnan(tensor)

    if not torch.any(nan_mask):
        return tensor

    filled_tensor = tensor.clone()

    if strategy == "zero":
        filled_tensor[nan_mask] = 0.0
    elif strategy == "mean":
        col_means = torch.nanmean(tensor, dim=0, keepdim=True)
        filled_tensor = torch.where(nan_mask, col_means, tensor)
    elif strategy == "median":
        col_medians = torch.nanmedian(tensor, dim=0, keepdim=True).values
        filled_tensor = torch.where(nan_mask, col_medians, tensor)
    elif strategy == "min":
        col_mins = torch.nanmin(tensor, dim=0, keepdim=True).values
        filled_tensor = torch.where(nan_mask, col_mins, tensor)
    elif strategy == "max":
        col_maxs = torch.nanmax(tensor, dim=0, keepdim=True).values
        filled_tensor = torch.where(nan_mask, col_maxs, tensor)
    elif strategy == "value":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'value'.")
        filled_tensor[nan_mask] = fill_value
    elif strategy == "most_frequent":
        for col_idx in range(tensor.shape[1]): #using for loop only on columns, not on the batch dimension.
            col = tensor[:, col_idx]
            non_nan_col = col[~torch.isnan(col)]
            if non_nan_col.numel() > 0:
                values, counts = torch.unique(non_nan_col, return_counts=True)
                most_frequent_value = values[torch.argmax(counts)]
                filled_tensor[nan_mask[:, col_idx], col_idx] = most_frequent_value
            else:
                filled_tensor[nan_mask[:, col_idx], col_idx] = 0.0 #if all values are nan, fill with 0.
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    return filled_tensor

def distribution_encoder(feat: torch.Tensor) -> torch.Tensor:
    num_cols = feat.shape[1]
    for col in range(num_cols):
        unique, counts = torch.unique(feat[:, col], return_counts=True)
        probs = counts.cumsum(dim = 0) / feat.shape[0]
        indices = torch.searchsorted(unique, feat[:, col], side='right') - 1
        new_array = probs[indices]
        feat[:, col] = new_array
    return feat

def distribution_categorical_encoder(feat: torch.Tensor) -> torch.Tensor:
    """
    Encode categorical features as a distribution of their values
    Shouldn't do cum sum here since there's no inherent order for categorical features
    """
    num_cols = feat.shape[1]
    for col in range(num_cols):
        unique, counts = torch.unique(feat[:, col], return_counts=True)
        probs = counts / feat.shape[0]
        indices = torch.searchsorted(unique, feat[:, col], side='right') - 1
        new_array = probs[indices]
        feat[:, col] = new_array
    return feat

def get_task_meta(task, link_args=None, train_table=None, transductive=True):
    """Get the metadata of the task.

    Args:
        task: the task object
        link_args: the args for link prediction
        train_table: optional pre-loaded train table (avoids re-loading for regression)
        transductive: whether the task is transductive
    """
    # Normalize task_type: if it was converted to a string, convert it back to enum
    tt = task.task_type
    if isinstance(tt, str):
        try:
            tt = TaskType(tt)
        except ValueError:
            pass  # leave as string for downstream checks

    if tt == TaskType.LINK_PREDICTION or str(tt).lower() == 'recommendation':
        task_type = 'link'
    else:
        task_type = 'node'

    clamp_min, clamp_max = None, None
    if task_type == 'node':
        if tt == TaskType.BINARY_CLASSIFICATION and transductive:
            out_channels = 1
            loss_fn = BCEWithLogitsLoss()
            tune_metric = "roc_auc"
            higher_is_better = True
        elif tt == TaskType.BINARY_CLASSIFICATION and not transductive:
            # For cross-data training, use cross entropy loss for unified interface
            out_channels = 2
            loss_fn = CrossEntropyLoss()
            tune_metric = "roc_auc"
            higher_is_better = True
        elif tt == TaskType.REGRESSION:
            out_channels = 1
            loss_fn = L1Loss()
            tune_metric = "mae"
            higher_is_better = False
            if train_table is None:
                train_table = task.get_table("train")
            clamp_min, clamp_max = np.percentile(
                train_table.df[task.target_col].to_numpy(), [2, 98]
            )
        elif tt == TaskType.MULTICLASS_CLASSIFICATION:
            out_channels = task.num_classes
            loss_fn = CrossEntropyLoss()
            tune_metric = "accuracy"
            higher_is_better = True
        elif tt == TaskType.MULTILABEL_CLASSIFICATION:
            out_channels = task.num_labels
            loss_fn = BCEWithLogitsLoss()
            tune_metric = "multilabel_auprc_macro"
            higher_is_better = True
        else:
            raise ValueError(f"Task type {tt} is unsupported")
    elif task_type == 'link':
        out_channels = link_args.out_channels
        loss_fn = link_args.loss_fn
        tune_metric = link_args.tune_metric
        higher_is_better = True
    else:
        raise ValueError(f"Task type {task_type} is unsupported")
    return {
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "out_channels": out_channels,
        "loss_fn": loss_fn,
        "tune_metric": tune_metric,
        "higher_is_better": higher_is_better,
        "task_type": task_type
    }


class CachedEmbeddingTensorMapper(EmbeddingTensorMapper):
    def __init__(self, embedder, batch_size, encoder_name):
        super().__init__(embedder, batch_size)
        self.encoder_name = encoder_name
    def forward(
        self,
        ser: Series,
        *,
        device: torch.device | None = None,
        dataset_name: str = "",
        col_name: str = "",
        os_cache_path: str = ""
    ) -> MultiEmbeddingTensor:
        if not os_cache_path:
            os_cache_path = f"{os.environ.get('XDG_CACHE_HOME', 'cache_data')}/embedding"
        if self.encoder_name == "llm":
            cache_path = os.path.join(os_cache_path, f"{dataset_name}_{col_name}_{ser.shape[0]}_llm.pt")
        else:
            cache_path = os.path.join(os_cache_path, f"{dataset_name}_{col_name}_{ser.shape[0]}.pt")
        if os.path.exists(cache_path):
            logger.info(f"Loading embedding from cache: {cache_path}")
            cached = torch.load(cache_path, map_location='cpu')
            if isinstance(cached, Tensor):
                return MultiEmbeddingTensor(
                    num_rows=len(ser),
                    num_cols=1,
                    values=cached,
                    offset=torch.tensor([0, cached.shape[1] if cached.dim() > 1 else cached.shape[0]]),
                ).to(device)
            return cached
        else:
            if self.embedder is not None:
                ser = ser.astype(str)
                ser_list = ser.tolist()
                if self.batch_size is None:
                    values = self.embedder(ser_list)
                else:
                    emb_list = []
                    for i in tqdm(range(0, len(ser_list), self.batch_size),
                                desc="Embedding raw data in mini-batch"):
                        emb = self.embedder(ser_list[i:i + self.batch_size])
                        emb_list.append(emb.to(device))
                    values = torch.cat(emb_list, dim=0)
            else:
                dtype = _get_default_numpy_dtype()
                values = torch.from_numpy(np.stack(ser.values).astype(dtype))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(values, cache_path)
            logger.info(f"Saving embedding to cache: {cache_path}")
        return MultiEmbeddingTensor(
            num_rows=len(ser),
            num_cols=1,
            values=values,
            offset=torch.tensor([0, len(values[0])]),
        ).to(device)


class DataFrameToTensorFrameConverterWithCache(DataFrameToTensorFrameConverter):
    def __init__(self, *args, rfm_preprocess: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.rfm_preprocess = rfm_preprocess

    def _get_mapper(self, col: str) -> TensorMapper:
        stype = self.col_to_stype[col]
        if stype == torch_frame.text_embedded:
            text_embedder_cfg = self.col_to_text_embedder_cfg[col]
            return CachedEmbeddingTensorMapper(
                text_embedder_cfg.text_embedder,
                text_embedder_cfg.batch_size,
                text_embedder_cfg.encoder_name,
            )
        else: 
            return super()._get_mapper(col)
    
    def __call__(
        self,
        df: DataFrame,
        device: torch.device | None = None,
        dataset_name: str = "",
    ) -> TensorFrame:
        r"""Convert a given :class:`DataFrame` object into :class:`TensorFrame`
        object.
        """
        xs_dict: dict[torch_frame.stype, list[TensorData]] = defaultdict(list)

        for stype, col_names in self.col_names_dict.items():
            for col in col_names:
                ## cache the text_embedded type
                if self.col_to_stype[col] == torch_frame.text_embedded:
                    out = self._get_mapper(col).forward(
                        df[col],
                        device=device,
                        dataset_name=dataset_name,
                        col_name=col,
                    )
                    if not isinstance(out, MultiEmbeddingTensor):
                        out = MultiEmbeddingTensor.from_tensor_list([out])

                else:
                    out = self._get_mapper(col).forward(df[col], device=device)
                    ## fill nan 
                    if stype == torch_frame.numerical:
                        if isinstance(out, MultiEmbeddingTensor):
                            out = MultiEmbeddingTensor.from_tensor_list([fill_nan_columnwise(out.values, strategy="mean")])
                        else:
                            old_shape = out.shape
                            out = fill_nan_columnwise(out, strategy="mean")
                            out = out.reshape(old_shape)
                    elif stype == torch_frame.categorical:
                        if isinstance(out, MultiEmbeddingTensor):
                            out = MultiEmbeddingTensor.from_tensor_list([fill_nan_columnwise(out.values.float(), strategy="most_frequent")])
                        else:
                            old_shape = out.shape
                            out = out.float()
                            out = fill_nan_columnwise(out, strategy="most_frequent")
                            out = out.reshape(old_shape)
                xs_dict[stype].append(out)

        feat_dict = {}
        for stype, xs in xs_dict.items():
            if stype.use_multi_nested_tensor:
                feat_dict[stype] = MultiNestedTensor.cat(xs, dim=1)
            elif stype.use_dict_multi_nested_tensor:
                feat_dict[stype] = {}
                for key in xs[0].keys():
                    feat_dict[stype][key] = MultiNestedTensor.cat(
                        [x[key] for x in xs], dim=1)
            elif stype.use_multi_embedding_tensor:
                if self.rfm_preprocess:
                    feat_dict[stype] = {
                        col: tensor if isinstance(tensor, MultiEmbeddingTensor) else MultiEmbeddingTensor.from_tensor_list([tensor])
                        for col, tensor in zip(self.col_names_dict[stype], xs)
                    }
                else:
                    if not isinstance(xs[0], MultiEmbeddingTensor):
                        xs = MultiEmbeddingTensor.from_tensor_list(xs)
                        feat_dict[stype] = xs
                    else:
                        feat_dict[stype] = MultiEmbeddingTensor.cat(xs, dim=1)
            else:
                feat_dict[stype] = torch.stack(xs, dim=1)

        y: Tensor | None = None
        if self.target_col is not None and self.target_col in df:
            y = self._get_mapper(self.target_col).forward(
                df[self.target_col], device=device)

        tf = TensorFrame(feat_dict, self.col_names_dict, y, skip_validation=self.rfm_preprocess)
        return self._merge_feat(tf)

class DatasetwithTransform(Dataset):
    distribution_transform = {
        torch_frame.numerical: distribution_encoder,
        torch_frame.categorical: distribution_categorical_encoder
    }
    def __init__(
        self,
        df: Union[DataFrame, Dict[str, Any]],
        col_to_stype: dict[str, torch_frame.stype],
        target_col: str | None = None,
        split_col: str | None = None,
        col_to_sep: str | None | dict[str, str | None] = None,
        col_to_text_embedder_cfg: dict[str, TextEmbedderConfig]
        | TextEmbedderConfig | None = None,
        col_to_text_tokenizer_cfg: dict[str, TextTokenizerConfig]
        | TextTokenizerConfig | None = None,
        col_to_image_embedder_cfg: dict[str, ImageEmbedderConfig]
        | ImageEmbedderConfig | None = None,
        col_to_time_format: str | None | dict[str, str | None] = None,
        dataset_name: str = "",
        text_pca_dim: int | None = None,
    ):
        super().__init__(df, col_to_stype, target_col, split_col, col_to_sep, col_to_text_embedder_cfg, col_to_text_tokenizer_cfg, col_to_image_embedder_cfg, col_to_time_format)
        self.dataset_name = dataset_name
        self.text_pca_dim = text_pca_dim
        self.rfm_cache_paths: Dict[str, Any] = {}
        self._rfm_preprocess = False

    def _get_tensorframe_converter(self) -> DataFrameToTensorFrameConverterWithCache:
        return DataFrameToTensorFrameConverterWithCache(
                col_to_stype=self.col_to_stype,
                col_stats=self._col_stats,
                target_col=self.target_col,
                col_to_sep=self.col_to_sep,
                col_to_text_embedder_cfg=self.col_to_text_embedder_cfg,
                col_to_text_tokenizer_cfg=self.col_to_text_tokenizer_cfg,
                col_to_image_embedder_cfg=self.col_to_image_embedder_cfg,
                col_to_time_format=self.col_to_time_format,
                rfm_preprocess=self._rfm_preprocess,
            )
    
    # ------------------------------------------------------------------
    # RFM preprocessing helpers
    # ------------------------------------------------------------------
    def _resolve_rfm_cache_dir(self, cache_path: str | None) -> str:
        if cache_path is not None:
            if cache_path.endswith(('.pt', '.pth')):
                base_dir = osp.splitext(cache_path)[0] + "_rfm"
            else:
                base_dir = cache_path
        else:
            default_root = os.environ.get('REL_BENCH_CACHE')
            if default_root is None:
                default_root = osp.join(os.getcwd(), 'rfm_cache')
            name = self.dataset_name or 'relbench_dataset'
            base_dir = osp.join(default_root, 'rfm_preproc', name)

        os.makedirs(base_dir, exist_ok=True)
        self.rfm_cache_paths['base_dir'] = base_dir
        return base_dir

    def _get_train_indices(self) -> np.ndarray:
        if self.split_col is None:
            return np.arange(len(self.df))
        split_values = self.df[self.split_col].to_numpy()
        mask = split_values == SPLIT_TO_NUM['train']
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            return np.arange(len(self.df))
        return indices

    def _get_text_pca_dim(self, original_dim: int) -> int:
        if self.text_pca_dim is not None and self.text_pca_dim > 0:
            return min(self.text_pca_dim, original_dim)
        env_dim = os.environ.get('RFM_TEXT_PCA_DIM')
        if env_dim is not None:
            try:
                env_value = int(env_dim)
                if env_value > 0:
                    return min(env_value, original_dim)
            except ValueError:
                logger.warning("Invalid RFM_TEXT_PCA_DIM value: {}", env_dim)
        return min(3, original_dim)

    @staticmethod
    def _replace_nan_with_mean(array: np.ndarray, fallback_mean: np.ndarray | None = None) -> np.ndarray:
        if array.size == 0:
            return array
        if fallback_mean is None:
            mean = np.nanmean(array, axis=0)
        else:
            mean = fallback_mean
        mean = np.where(np.isnan(mean), 0.0, mean)
        filled = np.where(np.isnan(array), mean.reshape(1, -1), array)
        return filled

    @staticmethod
    def _refresh_tensor_frame_mapping(tf: TensorFrame) -> None:
        mapping: dict[str, tuple[torch_frame.stype, int]] = {}
        for stype_name, cols in tf.col_names_dict.items():
            if not isinstance(cols, list):
                continue
            for idx, col in enumerate(cols):
                mapping[col] = (stype_name, idx)
        tf._col_to_stype_idx = mapping

    def _apply_simple_categorical_encoding(
        self,
        tf: TensorFrame,
        *,
        train_df: pd.DataFrame,
    ) -> TensorFrame:
        if torch_frame.categorical not in tf.feat_dict:
            return tf

        categorical_cols = tf.col_names_dict.get(torch_frame.categorical, [])
        if not categorical_cols:
            tf.feat_dict.pop(torch_frame.categorical, None)
            tf.col_names_dict.pop(torch_frame.categorical, None)
            return tf

        encoder = OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )

        try:
            encoder.fit(train_df[categorical_cols])
        except ValueError:
            encoder.fit(train_df[categorical_cols].astype(str))

        full_encoded = encoder.transform(self.df[categorical_cols])
        encoded_tensor = torch.from_numpy(full_encoded.astype(np.float32))

        device = tf.feat_dict[torch_frame.numerical].device if torch_frame.numerical in tf.feat_dict else encoded_tensor.device
        encoded_tensor = encoded_tensor.to(device)

        if torch_frame.numerical in tf.feat_dict:
            tf.feat_dict[torch_frame.numerical] = torch.cat(
                [tf.feat_dict[torch_frame.numerical].to(device=device, dtype=torch.float32), encoded_tensor],
                dim=1,
            )
            tf.col_names_dict[torch_frame.numerical] = (
                tf.col_names_dict.get(torch_frame.numerical, []) + list(categorical_cols)
            )
        else:
            tf.feat_dict[torch_frame.numerical] = encoded_tensor
            tf.col_names_dict[torch_frame.numerical] = list(categorical_cols)

        tf.feat_dict.pop(torch_frame.categorical, None)
        tf.col_names_dict.pop(torch_frame.categorical, None)

        for idx, col in enumerate(categorical_cols):
            series = pd.Series(full_encoded[:, idx])
            self._col_stats[col] = compute_col_stats(series, torch_frame.numerical)

        self._refresh_tensor_frame_mapping(tf)
        return tf

    def _apply_text_pca_transform(
        self,
        tf: TensorFrame,
        cache_dir: str
    ) -> TensorFrame:
        text_feat_dict = tf.feat_dict.get(torch_frame.embedding)
        numeric_tensor = tf.feat_dict.get(torch_frame.numerical)

        if text_feat_dict is None:
            return tf

        pca_cache_dir = osp.join(cache_dir, 'text_pca')
        os.makedirs(pca_cache_dir, exist_ok=True)
        self.rfm_cache_paths.setdefault('text_pca', {})

        device = numeric_tensor.device if numeric_tensor is not None else torch.device('cpu')
        dtype = numeric_tensor.dtype if numeric_tensor is not None else torch.float32

        base_columns: list[Tensor] = []
        base_names: list[str] = []

        if numeric_tensor is not None:
            base_columns = [numeric_tensor[:, idx] for idx in range(numeric_tensor.size(1))]
            base_names = list(tf.col_names_dict.get(torch_frame.numerical, []))

        new_columns: list[Tensor] = []
        new_names: list[str] = []

        if isinstance(text_feat_dict, dict):

            def _to_numpy(array: Any) -> np.ndarray | None:
                if array is None:
                    return None
                if cp is not None and isinstance(array, cp.ndarray):  # type: ignore[attr-defined]
                    return cp.asnumpy(array)
                if hasattr(array, "get"):
                    return array.get()
                return np.asarray(array)

            from models.dfs import gpu_ipca_two_pass
            use_gpu_pca = gpu_ipca_two_pass is not None and cp is not None

            for col_name, met in text_feat_dict.items():
                column_tensor = met.values.to(torch.float32)
                if column_tensor.dim() == 1:
                    column_tensor = column_tensor.unsqueeze(1)
                elif column_tensor.dim() > 2:
                    column_tensor = column_tensor.view(column_tensor.size(0), -1)

                if column_tensor.numel() == 0:
                    continue

                original_dim = column_tensor.size(1)
                target_dim = self._get_text_pca_dim(original_dim)
                if target_dim <= 0:
                    target_dim = original_dim

                column_np = column_tensor.detach().cpu().numpy().astype(np.float32)

                model_id = f"{self.dataset_name or 'dataset'}_{col_name}_{target_dim}.joblib"
                pca_model_path = osp.join(pca_cache_dir, model_id)

                reduced_np: np.ndarray

                if osp.exists(pca_model_path):
                    pca_model = joblib.load(pca_model_path)
                    fallback_mean = _to_numpy(getattr(pca_model, "mean_", None))
                    column_np_filled = self._replace_nan_with_mean(
                        column_np,
                        fallback_mean=fallback_mean,
                    )

                    model_module = getattr(type(pca_model), "__module__", "")
                    is_gpu_model = "cuml" in model_module.lower()

                    if is_gpu_model:
                        if cp is None:
                            raise RuntimeError(
                                "Loaded GPU IncrementalPCA model but CuPy is not available. "
                                "Install CuPy to reuse cached GPU PCA models or delete the cache."
                            )
                        column_gpu = cp.asarray(column_np_filled, dtype=cp.float32)  # type: ignore[attr-defined]
                        reduced_gpu = pca_model.transform(column_gpu)
                        reduced_np = cp.asnumpy(reduced_gpu).astype(np.float32)
                    elif hasattr(pca_model, "transform"):
                        reduced_np = np.asarray(
                            pca_model.transform(column_np_filled),
                            dtype=np.float32,
                        )
                    else:
                        reduced_np = column_np_filled
                else:
                    column_np_filled = self._replace_nan_with_mean(column_np)

                    if use_gpu_pca:
                        total_rows = column_np_filled.shape[0]
                        batch_size = max(256, target_dim * 4)

                        def make_batches() -> Iterable[np.ndarray]:
                            for start_idx in range(0, total_rows, batch_size):
                                batch = column_np_filled[start_idx:start_idx + batch_size]
                                if batch.size == 0:
                                    continue
                                yield batch

                        pca_model, reduced_np = gpu_ipca_two_pass(
                            make_batches,
                            n_components=target_dim,
                            batch_size=batch_size,
                            dtype=cp.float32,  # type: ignore[attr-defined]
                            out_dtype=np.float32,
                            n_samples=column_np_filled.shape[0],
                            return_model=True,
                        )
                        joblib.dump(pca_model, pca_model_path)
                    else:
                        raise RuntimeError(
                            "IncrementalPCA requires either cuML (GPU) to be installed."
                        )

                reduced_tensor = torch.from_numpy(reduced_np).to(device=device, dtype=dtype)
                generated_names = [
                    f"{col_name}_feat_{dim_idx + 1}"
                    for dim_idx in range(reduced_tensor.shape[1])
                ]

                for dim_idx, new_name in enumerate(generated_names):
                    new_columns.append(reduced_tensor[:, dim_idx])
                    new_names.append(new_name)

                features_filename = model_id.replace(".joblib", "_features.npy")
                features_path = osp.join(pca_cache_dir, features_filename)
                np.save(features_path, reduced_np, allow_pickle=False)

                self.rfm_cache_paths['text_pca'][col_name] = {
                    "model": pca_model_path,
                    "features": features_path,
                }
                self._col_stats.pop(col_name, None)

            tf.feat_dict.pop(torch_frame.embedding, None)
            tf.col_names_dict.pop(torch_frame.embedding, None)
        else:
            raise ValueError(f"text_feat_dict is not a dict: {text_feat_dict}")

    

        combined_tensors = base_columns + new_columns
        combined_names = base_names + new_names

        if combined_tensors:
            stacked = torch.stack(combined_tensors, dim=1).to(device=device, dtype=dtype)
            tf.feat_dict[torch_frame.numerical] = stacked
            tf.col_names_dict[torch_frame.numerical] = combined_names
        else:
            tf.feat_dict.pop(torch_frame.numerical, None)
            tf.col_names_dict.pop(torch_frame.numerical, None)

        tf.feat_dict.pop(torch_frame.text_embedded, None)
        tf.col_names_dict.pop(torch_frame.text_embedded, None)
        tf.feat_dict.pop(torch_frame.embedding, None)
        tf.col_names_dict.pop(torch_frame.embedding, None)

        self._refresh_tensor_frame_mapping(tf)
        return tf


    def _apply_numeric_normalization(
        self,
        tf: TensorFrame,
        train_indices: np.ndarray,
        cache_dir: str,
    ) -> None:
        if torch_frame.numerical not in tf.feat_dict:
            return

        scaler_dir = osp.join(cache_dir, 'scalers')
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_path = osp.join(scaler_dir, 'standard.joblib')

        numeric_tensor = tf.feat_dict[torch_frame.numerical]
        numeric_np = numeric_tensor.detach().cpu().numpy().astype(np.float64)

        if train_indices.size == 0 or train_indices.size == numeric_np.shape[0]:
            train_np = numeric_np
        else:
            train_np = numeric_np[train_indices]

        if osp.exists(scaler_path):
            scaler: StandardScaler = joblib.load(scaler_path)
            if scaler.mean_.shape[0] != numeric_np.shape[1]:
                scaler = StandardScaler()
                train_np_tmp = self._replace_nan_with_mean(train_np)
                scaler.fit(train_np_tmp)
                joblib.dump(scaler, scaler_path)
        else:
            train_np = self._replace_nan_with_mean(train_np)
            scaler = StandardScaler()
            scaler.fit(train_np)
            joblib.dump(scaler, scaler_path)

        numeric_np = self._replace_nan_with_mean(numeric_np, fallback_mean=scaler.mean_)
        transformed = scaler.transform(numeric_np)
        transformed_tensor = torch.from_numpy(transformed).to(numeric_tensor.device).type_as(numeric_tensor)
        tf.feat_dict[torch_frame.numerical] = transformed_tensor
        self.rfm_cache_paths['numeric_scaler'] = scaler_path

    def _recompute_metadata_after_transform(
        self,
        tf: TensorFrame,
        *,
        original_col_to_stype: Dict[str, torch_frame.stype],
        original_col_stats: Dict[str, Dict[StatType, Any]],
    ) -> None:
        final_col_to_stype: Dict[str, torch_frame.stype] = {
            col: stype
            for stype, cols in tf.col_names_dict.items()
            for col in cols
        }

        if self.target_col is not None:
            final_col_to_stype[self.target_col] = original_col_to_stype.get(
                self.target_col,
                self.col_to_stype.get(self.target_col, torch_frame.categorical),
            )

        numeric_stats: Dict[str, Dict[StatType, Any]] = {}
        if torch_frame.numerical in tf.feat_dict:
            numeric_tensor = tf.feat_dict[torch_frame.numerical]
            numeric_np = numeric_tensor.detach().cpu().numpy()
            numeric_names = tf.col_names_dict.get(torch_frame.numerical, [])
            for idx, col in enumerate(numeric_names):
                series = pd.Series(numeric_np[:, idx])
                numeric_stats[col] = compute_col_stats(series, torch_frame.numerical)

        new_col_stats: Dict[str, Dict[StatType, Any]] = {}
        for col, stype in final_col_to_stype.items():
            if col in numeric_stats:
                stats = numeric_stats[col]
            else:
                stats = original_col_stats.get(col)
                if stats is None and col in self.df:
                    stats = compute_col_stats(
                        self.df[col],
                        stype,
                        sep=self.col_to_sep.get(col, None),
                        time_format=self.col_to_time_format.get(col, None),
                    )
            if stats is not None:
                new_col_stats[col] = stats

        removed_cols = set(original_col_to_stype.keys()) - set(final_col_to_stype.keys())
        for mapping in (
            self.col_to_sep,
            self.col_to_text_embedder_cfg,
            self.col_to_text_tokenizer_cfg,
            self.col_to_image_embedder_cfg,
            self.col_to_time_format,
        ):
            if isinstance(mapping, dict):
                for col in removed_cols:
                    mapping.pop(col, None)

        self.col_to_stype = final_col_to_stype
        self._col_stats = new_col_stats
        self._refresh_tensor_frame_mapping(tf)

    def _apply_simple_numerical_normalization(
        self,
        tf: TensorFrame,
        *,
        skew_threshold: float = 0.99,
    ) -> TensorFrame:
        if torch_frame.numerical not in tf.feat_dict:
            return tf

        numeric_tensor = tf.feat_dict[torch_frame.numerical]
        device = numeric_tensor.device
        dtype = numeric_tensor.dtype

        numeric_np = numeric_tensor.detach().cpu().numpy().astype(np.float64)
        imputer = SimpleImputer(strategy='median')
        numeric_np = imputer.fit_transform(numeric_np)

        transformed_cols: list[np.ndarray] = []
        col_names = tf.col_names_dict.get(torch_frame.numerical, [])
        for idx, col_name in enumerate(col_names):
            column = numeric_np[:, idx: idx + 1]
            series = pd.Series(column[:, 0])
            skewness = series.skew(skipna=True)
            if np.isnan(skewness):
                skewness = 0.0

            if abs(skewness) > skew_threshold:
                transformer = QuantileTransformer(
                    n_quantiles=min(1000, max(10, column.shape[0])),
                    output_distribution='normal',
                    random_state=0,
                )
            else:
                transformer = StandardScaler()

            column_transformed = transformer.fit_transform(column)
            transformed_cols.append(column_transformed)

        transformed_np = np.concatenate(transformed_cols, axis=1)
        transformed_tensor = torch.from_numpy(transformed_np).to(device=device, dtype=dtype)
        tf.feat_dict[torch_frame.numerical] = transformed_tensor

        # Update numerical column stats to reflect normalized values
        for idx, col_name in enumerate(col_names):
            series = pd.Series(transformed_np[:, idx])
            self._col_stats[col_name] = compute_col_stats(series, torch_frame.numerical)

        self._to_tensor_frame_converter = self._get_tensorframe_converter()

        return tf

    def _apply_rfm_preprocessing(
        self,
        *,
        device: torch.device | None,
        cache_path: str | None,
    ) -> TensorFrame:
        cache_dir = self._resolve_rfm_cache_dir(cache_path)
        train_indices = self._get_train_indices()
        original_col_to_stype = dict(self.col_to_stype)
        original_col_stats = dict(self._col_stats)

        tf = self._tensor_frame
        train_df = self.df if train_indices.size == len(self.df) else self.df.iloc[train_indices]

        tf = self._apply_simple_categorical_encoding(tf, train_df=train_df)
        tf = self._apply_text_pca_transform(tf, cache_dir)
        self._apply_numeric_normalization(tf, train_indices, cache_dir)

        self._recompute_metadata_after_transform(
            tf,
            original_col_to_stype=original_col_to_stype,
            original_col_stats=original_col_stats,
        )
        self._to_tensor_frame_converter = self._get_tensorframe_converter()
        self._tensor_frame = tf
        return tf

    def materialize(self, device: torch.device | None = None, path: str | None = None, col_stats: Dict[str, Dict[StatType, Any]] | None = None, transform: str | None = None) -> Dataset:
        r"""Materializes the dataset into a tensor representation. From this
        point onwards, the dataset should be treated as read-only.

        Args:
            device (torch.device, optional): Device to load the
                :class:`TensorFrame` object. (default: :obj:`None`)
            path (str, optional): If path is specified and a cached file
                exists, this will try to load the saved the
                :class:`TensorFrame` object and :obj:`col_stats`.
                If :obj:`path` is specified but a cached file does not exist,
                this will perform materialization and then save the
                :class:`TensorFrame` object and :obj:`col_stats` to
                :obj:`path`. If :obj:`path` is :obj:`None`, this will
                materialize the dataset without caching.
                (default: :obj:`None`)
            col_stats (Dict[str, Dict[StatType, Any]], optional): optional
            col_stats provided by the user. If not provided, the statistics
            is calculated from the dataframe itself. (default: :obj:`None`)

        """
        if self.is_materialized:
            # Materialized without specifying path at first and materialize
            # again by specifying the path
            if path is not None and not osp.isfile(path):
                torch_frame.save(self._tensor_frame, self._col_stats, path)
            return self

        if path is not None and osp.isfile(path):
            # Load tensor_frame and col_stats
            self._tensor_frame, self._col_stats = torch_frame.load(
                path, device)
            # Instantiate the converter
            self._to_tensor_frame_converter = self._get_tensorframe_converter()
            # Mark the dataset has been materialized
            self._is_materialized = True
            return self

        # 1. Fill column statistics:
        self._rfm_preprocess = transform in {'rfm_preprocess', 'rfm', 'pca', 'pca_numerical_normalization'}

        if col_stats is None:
            # calculate from data if col_stats is not provided
            for col, stype in self.col_to_stype.items():
                ser = self.df[col]
                self._col_stats[col] = compute_col_stats(
                    ser,
                    stype,
                    sep=self.col_to_sep.get(col, None),
                    time_format=self.col_to_time_format.get(col, None),
                )
                # For a target column, sort categories lexicographically
                # such that we do not accidentally swap labels in binary
                # classification tasks.
                if col == self.target_col and stype == torch_frame.categorical:
                    index, value = self._col_stats[col][StatType.COUNT]
                    if len(index) == 2:
                        ser = pd.Series(index=index, data=value).sort_index()
                        index, value = ser.index.tolist(), ser.values.tolist()
                        self._col_stats[col][StatType.COUNT] = (index, value)
        else:
            # basic validation for the col_stats provided by the user
            for col_, stype_ in self.col_to_stype.items():
                assert col_ in col_stats, \
                    f"{col_} is not specified in the provided col_stats"
                stats_ = col_stats[col_]
                assert all([key_ in stats_
                            for key_ in StatType.stats_for_stype(stype_)]), \
                    "not all required stats are calculated" \
                    f" in the provided col_stats for {col}"

            self._col_stats = col_stats

        # 2. Create the `TensorFrame`:
        self._to_tensor_frame_converter = self._get_tensorframe_converter()
        self._tensor_frame = self._to_tensor_frame_converter(self.df, device, self.dataset_name)

        if transform in ('distribution_old', 'distribution'):
            categorical_transform = self.distribution_transform[torch_frame.categorical]
            numerical_transform = self.distribution_transform[torch_frame.numerical]

            for stype, value in self._tensor_frame.feat_dict.items():
                if stype == torch_frame.categorical:
                    self._tensor_frame.feat_dict[stype] = categorical_transform(value)
                elif stype == torch_frame.numerical:
                    self._tensor_frame.feat_dict[stype] = numerical_transform(value)

        elif transform in ('rfm_preprocess', 'rfm'):
            self._tensor_frame = self._apply_rfm_preprocessing(
                device=device,
                cache_path=path,
            )

        elif transform == 'pca':
            cache_dir = self._resolve_rfm_cache_dir(path)
            original_col_to_stype = dict(self.col_to_stype)
            original_col_stats = dict(self._col_stats)
            self._tensor_frame = self._apply_text_pca_transform(self._tensor_frame, cache_dir)
            self._recompute_metadata_after_transform(
                self._tensor_frame,
                original_col_to_stype=original_col_to_stype,
                original_col_stats=original_col_stats,
            )

        elif transform == 'pca_numerical_normalization':
            cache_dir = self._resolve_rfm_cache_dir(path)
            train_indices = self._get_train_indices()
            original_col_to_stype = dict(self.col_to_stype)
            original_col_stats = dict(self._col_stats)
            self._tensor_frame = self._apply_text_pca_transform(self._tensor_frame, cache_dir)
            self._apply_numeric_normalization(self._tensor_frame, train_indices, cache_dir)
            self._recompute_metadata_after_transform(
                self._tensor_frame,
                original_col_to_stype=original_col_to_stype,
                original_col_stats=original_col_stats,
            )

        elif transform == 'numerical_normalization':
            self._tensor_frame = self._apply_simple_numerical_normalization(
                self._tensor_frame
            )

        elif transform == 'none':
            logger.info("Load data without transform")

        else:
            raise ValueError(f"Transform {transform} not supported")

        # 3. Update col stats based on `TensorFrame`
        self._update_col_stats()

        # 4. Mark the dataset as materialized:
        self._is_materialized = True

        if path is not None:
            os.makedirs(osp.dirname(path), exist_ok=True)
            # Cache the dataset if user specifies the path
            torch_frame.save(self._tensor_frame, self._col_stats, path)

        return self
