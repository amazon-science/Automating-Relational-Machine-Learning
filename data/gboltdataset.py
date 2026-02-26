"""Utilities for converting RelBench datasets to GBolt / DBInfer-Bench format.

Provides functions to generate YAML metadata from RelBench database objects,
convert between PyG RDB and 4DBInfer dataset representations, and produce
cached DBInfer-format datasets for downstream training.
"""

import os.path as osp
from functools import lru_cache
from typing import Any

import torch_frame as tfr
from loguru import logger
from relbench.base import TaskType

from data.mtaskdataset import MultiTabularDataset
from dbinfer_bench.pyg_rdb_dataset import (
    DBBRDBDataset,
    DBBRDBDatasetCreator,
    DBBRDBTaskCreator,
)
from dbinfer_bench.rdb_dataset import DBBRDBDataset as DBBRDBDataset_4dbinfer
from dbinfer_bench.rdb_dataset import DBBTaskType
from utils.input import write_dict_to_yaml


def generate_yaml_metadata(
    db_obj, tasks, cache_dir: str, col_to_stype_dict: dict, dfs: bool = True
) -> dict:
    """Generate a YAML-serialisable metadata dict for a RelBench database.

    Args:
        db_obj: RelBench database object with ``get_db()``, ``val_timestamp``
            and ``test_timestamp`` attributes.
        tasks: Iterable of RelBench task objects.
        cache_dir: Root cache directory for the dataset.
        col_to_stype_dict: Mapping ``{table_name: {col_name: stype}}``.
        dfs: Whether the dataset uses DFS-based features.

    Returns:
        A dict representing the full dataset metadata including table schemas
        and task definitions.
    """
    data_dict = convert_relbench_to_gbolt_format(db_obj, cache_dir, col_to_stype_dict)
    data_dict['val_timestamp_cutoff'] = db_obj.val_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    data_dict['test_timestamp_cutoff'] = db_obj.test_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    data_dict['tasks'] = []
    for task in tasks:
        task_dict, task_data = convert_task(task)
        data_dict['tasks'].append(task_dict)
    return data_dict


def convert_relbench_to_gbolt_format(
    db_obj, cache_dir: str, col_to_stype_dict: dict
) -> dict:
    """Convert a RelBench database schema into GBolt metadata format.

    Iterates over every table and column described in *col_to_stype_dict*,
    mapping each column to a GBolt-compatible dtype (``primary_key``,
    ``foreign_key``, ``float``, ``category``, ``text``, ``datetime``, or
    ``multi_category``).

    Args:
        db_obj: RelBench database object.
        cache_dir: Root cache directory used as the ``source`` field.
        col_to_stype_dict: Mapping ``{table_name: {col_name: stype}}``.

    Returns:
        A dict with keys ``dataset_name``, ``tables``, and ``source``.
    """
    db = db_obj.get_db()
    data_dict = {
        'dataset_name': db_obj.name,
        'tables': [],
        'source': cache_dir,
    }
    for table_name, col_info in col_to_stype_dict.items():
        table_dict = {
            'name': table_name,
            'format': 'parquet',
            'source': f'{table_name}.parquet',
            'columns': [],
        }
        table_obj = db.table_dict[table_name]
        table_dict['time_column'] = table_obj.time_col
        for col_name, stype in col_info.items():
            col_dict: dict[str, Any] = {'name': col_name}
            if col_name == table_obj.pkey_col:
                col_dict['dtype'] = 'primary_key'
            elif col_name in table_obj.fkey_col_to_pkey_table:
                col_dict['dtype'] = 'foreign_key'
                pkey_table_name = table_obj.fkey_col_to_pkey_table[col_name]
                pkey_table_obj = db.table_dict[pkey_table_name]
                col_dict['link_to'] = f'{pkey_table_name}.{pkey_table_obj.pkey_col}'
            else:
                if stype == tfr.numerical or stype == "numerical":
                    col_dict['dtype'] = 'float'
                elif stype == tfr.categorical or stype == "categorical":
                    col_dict['dtype'] = 'category'
                elif stype == tfr.text_embedded or stype == "text_embedded":
                    col_dict['dtype'] = 'text'
                elif stype == "datetime":
                    col_dict['dtype'] = 'datetime'
                elif stype == tfr.multicategorical or stype == 'multicategorical':
                    col_dict['dtype'] = 'multi_category'
                elif stype == tfr.timestamp:
                    col_dict['dtype'] = 'datetime'
                elif stype == "timestamp":
                    col_dict['dtype'] = 'category'
            table_dict['columns'].append(col_dict)
        data_dict['tables'].append(table_dict)
    return data_dict


def convert_task(task) -> tuple[dict, dict]:
    """Convert a RelBench task object into a GBolt task metadata dict.

    Handles binary classification, multiclass classification, regression,
    and link prediction task types. Also extracts per-split DataFrames and
    infers column schemas from the training split.

    Args:
        task: A RelBench task object.

    Returns:
        A tuple ``(task_dict, task_data)`` where *task_dict* is the metadata
        and *task_data* maps split names to their DataFrames.
    """
    task_dict: dict[str, Any] = {'format': 'numpy'}

    if task.task_type in (TaskType.BINARY_CLASSIFICATION, 'binary_classification'):
        task_dict['evaluation_metric'] = 'auroc'
        task_dict['task_type'] = 'classification'
        task_dict['entity_column'] = task.entity_col
        task_dict['target_column'] = task.target_col
        task_dict['target_table'] = task.entity_table
    elif task.task_type in (TaskType.MULTICLASS_CLASSIFICATION, 'multiclass_classification'):
        task_dict['evaluation_metric'] = 'accuracy'
        task_dict['task_type'] = 'classification'
        task_dict['entity_column'] = task.entity_col
        task_dict['target_column'] = task.target_col
        task_dict['target_table'] = task.entity_table
        task_dict['num_classes'] = task.num_classes
    elif task.task_type in (TaskType.REGRESSION, 'regression'):
        task_dict['evaluation_metric'] = 'mae'
        task_dict['task_type'] = 'regression'
        task_dict['entity_column'] = task.entity_col
        task_dict['target_column'] = task.target_col
        task_dict['target_table'] = task.entity_table
    elif task.task_type in (TaskType.LINK_PREDICTION, 'link_prediction'):
        task_dict['evaluation_metric'] = 'map'
        task_dict['task_type'] = 'retrieval'
        task_dict['entity_column'] = task.src_entity_col
        task_dict['entity_table'] = task.src_entity_table
        task_dict['target_column'] = task.dst_entity_col
        task_dict['target_table'] = task.dst_entity_table

    task_dict['time_column'] = task.time_col
    task_dict['name'] = task.name
    task_dict['source'] = f'{task.name}/{{split}}.parquet'

    if task.task_type in (TaskType.BINARY_CLASSIFICATION, 'binary_classification'):
        task_dict['num_classes'] = 2
    elif task.task_type in (TaskType.LINK_PREDICTION, 'link_prediction'):
        task_dict['num_dst_nodes'] = task.num_dst_nodes

    task_data = {}
    for split in ['train', 'val', 'test']:
        task_info = task.get_table(split, mask_input_cols=False)
        task_data[split] = task_info.df
        if split == 'train':
            pk_fk_dict = task_info.fkey_col_to_pkey_table
            df = task_info.df
            columns = []
            for column_name in df.columns:
                col: dict[str, Any] = {}
                if column_name == task_info.time_col:
                    col['name'] = column_name
                    col['dtype'] = 'datetime'
                    columns.append(col)
                elif column_name in pk_fk_dict:
                    if task.auto:
                        if task_info.fkey_col_to_pkey_table[column_name] != task.entity_table:
                            continue
                        else:
                            col['name'] = column_name
                            col['dtype'] = 'foreign_key'
                            col['link_to'] = f'{task.entity_table}.{task.entity_col}'
                    else:
                        col['name'] = column_name
                        col['dtype'] = 'foreign_key'
                        col['link_to'] = f'{task.entity_table}.{task.entity_col}'
                    if col:
                        columns.append(col)
                else:
                    if task.auto:
                        if column_name != task.target_col:
                            continue
                    col['name'] = column_name
                    if task.task_type in (TaskType.BINARY_CLASSIFICATION, 'binary_classification',
                                          TaskType.MULTICLASS_CLASSIFICATION, 'multiclass_classification'):
                        col['dtype'] = 'category'
                    else:
                        col['dtype'] = 'float'
                    if col:
                        columns.append(col)
            task_dict['columns'] = columns
    return task_dict, task_data


def generate(dataset: MultiTabularDataset, dfs: bool = True) -> DBBRDBDataset:
    """Generate a GBolt DBBRDBDataset from a MultiTabularDataset.

    Writes YAML metadata to disk and constructs a ``DBBRDBDataset`` pointing
    at the cached parquet files.

    Args:
        dataset: The source multi-table dataset.
        dfs: Whether to use DFS-based features.

    Returns:
        A ``DBBRDBDataset`` instance ready for training.
    """
    _, col_to_stype_dict = dataset.get_embed_feature_dict()
    data_dict = generate_yaml_metadata(
        dataset.database, dataset.tasks, dataset.cache_dir, col_to_stype_dict, dfs
    )
    file_config_path = osp.join(dataset.cache_dir, f'{dataset.database.name}.yaml')
    write_dict_to_yaml(data_dict, file_config_path)
    logger.info(f"Generated YAML metadata for {dataset.database.name} at {file_config_path}")

    data_cache_path = osp.join(dataset.cache_dir, f'{dataset.database.name}/db')
    return DBBRDBDataset(data_cache_path, file_config_path)


def pyg_rdb_to_4dbinfer(dataset: DBBRDBDataset, output_path: str) -> str:
    """Convert a PyG RDB dataset to the 4DBInfer on-disk format.

    Copies table data and task splits (including retrieval-specific columns)
    into the directory at *output_path*.

    Args:
        dataset: Source PyG ``DBBRDBDataset``.
        output_path: Destination directory for the converted dataset.

    Returns:
        The *output_path* string.
    """
    ctor = DBBRDBDatasetCreator(dataset.metadata.dataset_name)
    ctor.replace_tables_from(dataset)
    for orig_task in dataset.tasks:
        task_name = orig_task.metadata.name
        task_ctor = DBBRDBTaskCreator(task_name)
        task_ctor.copy_fields_from(orig_task.metadata)
        for col_meta in orig_task.metadata.columns:
            col_meta = dict(col_meta)
            col_name = col_meta.pop('name')
            task_ctor.add_task_data(
                col_name,
                orig_task.train_set.df[col_name],
                orig_task.validation_set.df[col_name],
                orig_task.test_set.df[col_name],
                **col_meta,
            )
        if orig_task.metadata.task_type == DBBTaskType.retrieval:
            for extra_col in [
                orig_task.metadata.key_prediction_label_column,
                orig_task.metadata.key_prediction_query_idx_column,
            ]:
                task_ctor.add_task_data(
                    extra_col,
                    None,
                    orig_task.validation_set[extra_col],
                    orig_task.test_set[extra_col],
                    dtype=None,
                )
        ctor.add_task(task_ctor)

    ctor.done(output_path)
    return output_path


@lru_cache(maxsize=1)
def get_4dbinfer_format(
    dataset: MultiTabularDataset,
    text_to_pca: bool = False,
    text_to_pca_dim: int = 3,
    task_idx: int = 0,
    augment_task_backlinks: bool = False,
) -> DBBRDBDataset:
    """Generate a cached DBInfer-format dataset for a single task.

    Similar to :func:`generate` but targets a single task (by index) and
    supports task-backlink augmentation for inductive settings.

    Args:
        dataset: The source multi-table dataset.
        text_to_pca: Whether to apply PCA to text embeddings (reserved).
        text_to_pca_dim: Target PCA dimensionality (reserved).
        task_idx: Index of the task to convert.
        augment_task_backlinks: Whether to augment with task backlinks.

    Returns:
        A ``DBBRDBDataset`` instance for the selected task.
    """
    col_to_stype_dict = dataset.get_col_to_stype_dict()
    data_dict = generate_yaml_metadata(
        dataset.database,
        dataset.tasks[task_idx:task_idx + 1],
        dataset.cache_dir,
        col_to_stype_dict,
        dfs=False,
    )
    file_config_path = osp.join(dataset.cache_dir, f'{dataset.database.name}.yaml')
    write_dict_to_yaml(data_dict, file_config_path)
    logger.info(f"Generated YAML metadata for {dataset.database.name} at {file_config_path}")
    data_cache_path = osp.join(dataset.cache_dir, f'{dataset.database.name}/db')
    return DBBRDBDataset(
        data_cache_path,
        file_config_path,
        augment_task_backlinks=augment_task_backlinks,
        backlink_task_indices=[task_idx],
    )
