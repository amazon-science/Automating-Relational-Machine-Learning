import contextlib
import importlib
import json
import os
import sys
import time
from functools import cache
from pathlib import Path

import ml_dtypes
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings


ANGLE_HASH_DIM = 4


def _pre_root() -> Path:
    env_override = os.environ.get("RT_PRE_DIR")
    if env_override:
        return Path(env_override)
    home = Path(os.environ.get("HOME", "."))
    data_pre = home / "data" / "pre"
    scratch_pre = home / "scratch" / "pre"
    return data_pre if data_pre.exists() else scratch_pre


def _load_json(path: Path, description: str, hint: str | None = None):
    try:
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError as err:
        message = f"Missing {description} at {path}"
        if hint:
            message += f"\nHint: {hint}"
        raise FileNotFoundError(message) from err
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Failed to parse {description} at {path}: {err}") from err


def _verify_preprocessed_artifacts(
    db_name: str, embedding_model: str, use_rfm_preprocessing: bool
) -> None:
    suffix = "_rfm" if use_rfm_preprocessing else ""
    pre_path = _pre_root() / db_name
    required = [
        ("nodes", pre_path / f"nodes{suffix}.rkyv"),
        ("text embeddings", pre_path / f"text_emb_{embedding_model}.bin"),
        ("offsets", pre_path / f"offsets{suffix}.rkyv"),
        ("p2f adjacency", pre_path / f"p2f_adj{suffix}.rkyv"),
    ]

    missing = [(label, path) for label, path in required if not path.exists()]
    if missing:
        missing_descriptions = "\n".join(
            f"- {label}: {path}" for label, path in missing
        )
        hint = (
            f"Did you run relational_transformer/scripts/preprocess_all_databases.sh {db_name}?"
        )
        raise FileNotFoundError(
            f"Missing required preprocessed files for {db_name}:\n{missing_descriptions}\nHint: {hint}"
        )

def _import_rustler_sampler():
    from rustler import Sampler as _RustSampler  # type: ignore[no-redef]

    return _RustSampler


def _rustler_source_token(project_root: Path) -> str:
    mtimes = []
    for path in project_root.rglob("*.rs"):
        with contextlib.suppress(OSError):
            mtimes.append(path.stat().st_mtime)
    with contextlib.suppress(OSError):
        mtimes.append((project_root / "Cargo.toml").stat().st_mtime)
    return str(max(mtimes) if mtimes else 0.0)


def _build_rustler(project_root: Path, token: str):
    build_flag = project_root / ".rustler_build_complete"
    lock_file = project_root / ".rustler_build.lock"

    while True:
        if build_flag.exists():
            with contextlib.suppress(OSError):
                if build_flag.read_text().strip() == token:
                    return _import_rustler_sampler()
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            time.sleep(1)
            continue
        else:
            try:
                import maturin_import_hook
                from maturin_import_hook.settings import MaturinSettings

                maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))
                importlib.invalidate_caches()
                if "rustler" in sys.modules:
                    rustler_mod = importlib.reload(sys.modules["rustler"])
                else:
                    rustler_mod = importlib.import_module("rustler")
                sampler_cls = rustler_mod.Sampler  # type: ignore[attr-defined]
                build_flag.write_text(token)
                return sampler_cls
            finally:
                os.close(fd)
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(lock_file)


def _ensure_rustler_sampler() -> type:
    project_root = Path(__file__).resolve().parents[2] / "rustler"
    token = _rustler_source_token(project_root)

    try:
        sampler_cls = _import_rustler_sampler()
    except ImportError:  # pragma: no cover
        sampler_cls = _build_rustler(project_root, token)
    else:
        build_flag = project_root / ".rustler_build_complete"
        needs_rebuild = True
        with contextlib.suppress(OSError):
            if build_flag.exists() and build_flag.read_text().strip() == token:
                needs_rebuild = False
        if needs_rebuild:
            sampler_cls = _build_rustler(project_root, token)

    return sampler_cls


Sampler = _ensure_rustler_sampler()


@cache
def _load_column_index(db_name: str, use_rfm_preprocessing: bool) -> dict:
    """
    Load the column index mapping for a dataset (cached).
    """
    suffix = "_rfm" if use_rfm_preprocessing else ""
    column_index_path = _pre_root() / db_name / f"column_index{suffix}.json"
    hint = (
        f"Did you run relational_transformer/scripts/preprocess_all_databases.sh {db_name}?"
    )
    return _load_json(column_index_path, "column index", hint=hint)


def get_column_index(
    column_name: str, table_name: str, db_name: str, use_rfm_preprocessing: bool = False
) -> int:
    """
    Get the index of a column in the text embeddings for a given dataset.
    """
    column_index = _load_column_index(db_name, use_rfm_preprocessing)
    target = f"{column_name} of {table_name}"

    if target not in column_index:
        raise ValueError(
            f'Column "{target}" not found in column_index.json for dataset {db_name}.'
        )

    return column_index[target]


def _load_shallow_meta(db_name: str, use_rfm_preprocessing: bool) -> dict | None:
    suffix = "_rfm" if use_rfm_preprocessing else ""
    meta_path = _pre_root() / db_name / f"shallow_meta{suffix}.json"
    if not meta_path.exists():
        return None
    try:
        with meta_path.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError as err:  # pragma: no cover - configuration error
        raise RuntimeError(f"Failed to parse shallow metadata at {meta_path}: {err}") from err


class RelationalDataset(Dataset):
    def __init__(
        self,
        tasks,  # List of task tuples: (db_name, table_name, target_column, split, columns_to_drop)
        batch_size,  # Number of samples per batch
        seq_len,  # Maximum sequence length for each sample
        rank,  # Current process rank in distributed training
        world_size,  # Total number of processes in distributed training
        max_bfs_width,  # Maximum breadth for BFS traversal during sampling
        embedding_model,  # Name of the text embedding model used
        d_text,  # Dimensionality of text embeddings
        seed,  # Random seed for reproducibility
        use_rfm_preprocessing: bool = False,  # Whether to use RFM (Relational Foundation Model) preprocessing
        shallow_dim: int = 3,  # Default dimensionality for shallow features
        shallow_c_dim: int = 4,  # Default dimensionality for shallow categorical features
    ):
        # List of tuples containing (db_name, node_offset, num_nodes) for each task
        dataset_tuples = []
        # Column indices for target columns in each task
        target_column_indices = []
        # Column indices to exclude (e.g., to prevent data leakage)
        drop_column_indices = []
        # Flag indicating whether RFM preprocessing is used
        self.use_rfm_preprocessing = use_rfm_preprocessing
        # Cache for shallow metadata to avoid repeated file reads
        shallow_meta_cache: dict[str, dict | None] = {}
        # Maximum shallow feature dimension found across all datasets
        max_shallow_dim = 0
        # Maximum shallow categorical feature dimension found across all datasets
        max_shallow_c_dim = 0
        # Metadata for each task including db, table, target column, split info
        self.task_metadata: list[dict[str, object]] = []

        for db_name, table_name, target_column, split, columns_to_drop in tasks:
            raw_split = split
            _verify_preprocessed_artifacts(db_name, embedding_model, self.use_rfm_preprocessing)
            if split == "train":
                split = "Train"
            elif split == "val":
                split = "Val"
            elif split == "test":
                split = "Test"

            suffix = "_rfm" if self.use_rfm_preprocessing else ""
            table_info_path = _pre_root() / db_name / f"table_info{suffix}.json"
            hint = (
                f"Did you run relational_transformer/scripts/preprocess_all_databases.sh {db_name}?"
            )
            table_info = _load_json(table_info_path, "table info", hint=hint)

            table_info_key = (
                f"{table_name}:Db"
                if f"{table_name}:Db" in table_info
                else f"{table_name}:{split}"
            )
            info = table_info[table_info_key]
            node_idx_offset = info["node_idx_offset"]
            num_nodes = info["num_nodes"]

            if self.use_rfm_preprocessing:
                meta = shallow_meta_cache.get(db_name)
                if meta is None:
                    meta = _load_shallow_meta(db_name, self.use_rfm_preprocessing)
                    shallow_meta_cache[db_name] = meta
                if meta:
                    max_shallow_dim = max(
                        max_shallow_dim,
                        int(meta.get("shallow_dim", 0) or 0),
                    )
                    max_shallow_c_dim = max(
                        max_shallow_c_dim,
                        int(meta.get("shallow_c_dim", 0) or 0),
                    )

            target_idx = get_column_index(
                target_column, table_name, db_name, self.use_rfm_preprocessing
            )
            target_column_indices.append(target_idx)

            drop_indices = [
                get_column_index(col, table_name, db_name, self.use_rfm_preprocessing)
                for col in columns_to_drop
            ]
            drop_column_indices.append(drop_indices)

            dataset_tuples.append((
                (
                    f"{db_name}::rfm"
                    if self.use_rfm_preprocessing
                    else db_name
                ),
                node_idx_offset,
                num_nodes,
            ))
            self.task_metadata.append(
                {
                    "db": db_name,
                    "table": table_name,
                    "target": target_column,
                    "split": raw_split,
                    "node_idx_offset": int(node_idx_offset),
                    "num_nodes": int(num_nodes),
                    "drop_columns": list(columns_to_drop),
                }
            )

        if self.use_rfm_preprocessing:
            resolved_shallow = max_shallow_dim if max_shallow_dim > 0 else shallow_dim
            resolved_shallow_c = max_shallow_c_dim if max_shallow_c_dim > 0 else shallow_c_dim
            # Dimensionality of shallow features (at least 1)
            self.d_shallow = max(1, resolved_shallow)
            # Dimensionality of shallow categorical features (at least 1)
            self.d_shallow_c = max(1, resolved_shallow_c)
        else:
            # When not using RFM preprocessing, use text embedding dimension for shallow features
            self.d_shallow = d_text
            self.d_shallow_c = d_text

        # Rust-based sampler for efficient batch generation and BFS traversal
        self.sampler = Sampler(
            dataset_tuples,
            batch_size,
            seq_len,
            rank,
            world_size,
            max_bfs_width,
            embedding_model,
            d_text,
            seed,
            target_column_indices,
            drop_column_indices,
        )

        set_dims = getattr(self.sampler, "set_feature_dims", None)
        if callable(set_dims):
            set_dims(self.d_shallow, self.d_shallow_c)
        else:
            warnings.warn(
                "rustler sampler built without set_feature_dims; consider rebuilding for full compatibility",
                RuntimeWarning,
            )

        get_dims = getattr(self.sampler, "feature_dims_py", None)
        if callable(get_dims):
            dims = get_dims()
            if dims != (d_text, self.d_shallow, self.d_shallow_c):
                raise RuntimeError(
                    "Sampler reported feature dimensions {} but dataset computed {}".format(
                        dims,
                        (d_text, self.d_shallow, self.d_shallow_c),
                    )
                )
        else:
            warnings.warn(
                "rustler sampler built without feature_dims_py; skipping dimension verification",
                RuntimeWarning,
            )
        if callable(get_dims):
            sampler_dims = get_dims()
            if sampler_dims != (d_text, self.d_shallow, self.d_shallow_c):
                raise RuntimeError(
                    "Sampler reported feature dimensions {} but dataset computed {}".format(
                        sampler_dims,
                        (d_text, self.d_shallow, self.d_shallow_c),
                    )
                )

        # Maximum sequence length for each sample
        self.seq_len = seq_len
        # Dimensionality of text embeddings
        self.d_text = d_text

    def __len__(self):
        return self.sampler.len_py()

    def __getitem__(self, batch_idx):
        """
        Returns a batch dictionary with the following fields:

        Structural fields (shape: [batch_size, seq_len]):
        - node_idxs: Global node indices in the graph
        - sem_types: Semantic type of each node (e.g., numerical, categorical, text)
        - masks: Attention mask indicating valid positions
        - is_targets: Binary mask indicating target nodes for prediction
        - is_task_nodes: Binary mask indicating nodes from the task table
        - is_padding: Binary mask indicating padding positions
        - table_name_idxs: Index of the table each node belongs to
        - col_name_idxs: Index of the column each node represents
        - class_value_idxs: Index for categorical values (class labels)

        Neighbor fields (shape: [batch_size, seq_len, 5]):
        - f2p_nbr_idxs: Foreign-to-primary key neighbor indices (up to 5 neighbors)

        Feature value fields:
        - number_values: Numerical feature values [batch_size, seq_len, 1]
        - datetime_values: Datetime feature values [batch_size, seq_len, 1]
        - boolean_values: Boolean feature values [batch_size, seq_len, 1]
        - text_values: Text embedding values [batch_size, seq_len, d_text]
        - shallow_values: Shallow feature embeddings [batch_size, seq_len, d_shallow]
        - shallow_c_values: Shallow categorical embeddings [batch_size, seq_len, d_shallow_c]
        - col_name_values: Column name embeddings [batch_size, seq_len, d_text]

        Metadata:
        - true_batch_size: Actual batch size (may be smaller than configured batch_size)
        """
        tup = self.sampler.batch_py(batch_idx)
        out = dict(tup)
        # Convert numpy arrays to PyTorch tensors, handling bfloat16 conversion for feature values
        for k, v in out.items():
            if k in [
                "number_values",
                "datetime_values",
                "text_values",
                "shallow_values",
                "shallow_c_values",
                "col_name_values",
                "boolean_values",
            ]:
                if k in ['shallow_values', 'shallow_c_values']:
                    ## skip them
                    continue
                out[k] = torch.from_numpy(v.view(np.float16)).view(torch.bfloat16)
            elif k == "true_batch_size":
                pass
            else:
                out[k] = torch.from_numpy(v)

        # Reshape structural fields to [batch_size, seq_len]
        out["node_idxs"] = out["node_idxs"].view(-1, self.seq_len)
        out["sem_types"] = out["sem_types"].view(-1, self.seq_len)
        out["masks"] = out["masks"].view(-1, self.seq_len)
        out["is_targets"] = out["is_targets"].view(-1, self.seq_len)
        out["is_task_nodes"] = out["is_task_nodes"].view(-1, self.seq_len)
        out["is_padding"] = out["is_padding"].view(-1, self.seq_len)
        out["table_name_idxs"] = out["table_name_idxs"].view(-1, self.seq_len)
        out["col_name_idxs"] = out["col_name_idxs"].view(-1, self.seq_len)
        out["class_value_idxs"] = out["class_value_idxs"].view(-1, self.seq_len)

        # Reshape neighbor indices to [batch_size, seq_len, 5]
        out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, self.seq_len, 5)
        # Reshape feature values to [batch_size, seq_len, feature_dim]
        out["number_values"] = out["number_values"].view(-1, self.seq_len, 1)
        out["datetime_values"] = out["datetime_values"].view(-1, self.seq_len, 1)
        out["boolean_values"] = (
            out["boolean_values"].view(-1, self.seq_len, 1).bfloat16()
        )
        out["text_values"] = out["text_values"].view(-1, self.seq_len, self.d_text)
        # out["shallow_values"] = out["shallow_values"].view(
        #     -1, self.seq_len, self.d_shallow
        # )
        # out["shallow_c_values"] = out["shallow_c_values"].view(
        #     -1, self.seq_len, self.d_shallow_c
        # )
        out["col_name_values"] = out["col_name_values"].view(
            -1, self.seq_len, self.d_text
        )

        return out
