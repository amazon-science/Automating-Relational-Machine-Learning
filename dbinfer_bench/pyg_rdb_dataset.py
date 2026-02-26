# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any, Set
from copy import deepcopy
import os
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    ForeignKey,
    Uuid,
    Float,
    ARRAY,
    VARCHAR,
    DateTime
)

from . import yaml_utils
from .dataset_meta import (
    DBBTaskType,
    DBBTaskEvalMetric,
    DBBTaskMeta,
    DBBColumnDType,
    DBBColumnSchema,
    DBBTableSchema,
    DBBTableDataFormat,
    DBBColumnID,
    DBBRelationship,
    DBBRDBDatasetMeta,
)
from .table_loader import get_table_data_loader
from .table_writer import get_table_data_writer
from relbench.base import Table
from functools import lru_cache
from typing import Self
from relbench.base import BaseTask, Table
from data.relbench import load_relbench_task
from pyarrow import parquet as pq
import pyarrow as pa
import json
from typing import Union
from functools import lru_cache
from .rdb_dataset import DBBRDBTask as DBBRDBTask_4db, DBBRDBDataset as DBBRDBDataset_4db

__all__ = ['DBBRDBTask', 'DBBRDBDataset', 'DBBRDBTaskCreator',
           'DBBRDBDatasetCreator', 'load_rdb_data']



class NamedTable(Table):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
    
    def __repr__(self) -> str:
        return (
            f"Table(name={self.name},\n"
            f"  df=\n{self.df},\n"
            f"  fkey_col_to_pkey_table={self.fkey_col_to_pkey_table},\n"
            f"  pkey_col={self.pkey_col},\n"
            f"  time_col={self.time_col}"
            f")"
        )
    
    def is_primary_key(self, col: str) -> bool:
        return col == self.pkey_col
    
    def is_foreign_key(self, col: str) -> bool:
        return col in self.fkey_col_to_pkey_table
    
    def save(self, path: Union[str, os.PathLike]) -> None:
        r"""Save the table to a parquet file.

        Stores other attributes as parquet metadata.
        """
        assert str(path).endswith(".parquet")
        metadata = {
            "fkey_col_to_pkey_table": self.fkey_col_to_pkey_table,
            "pkey_col": self.pkey_col,
            "time_col": self.time_col,
            "name": self.name,
        }

        # Convert DataFrame to a PyArrow Table
        table = pa.Table.from_pandas(self.df, preserve_index=False)

        # Add metadata to the PyArrow Table
        metadata_bytes = {
            key: json.dumps(value).encode("utf-8") for key, value in metadata.items()
        }

        table = table.replace_schema_metadata(
            {**table.schema.metadata, **metadata_bytes}
        )

        # Write the PyArrow Table to a Parquet file using pyarrow.parquet
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> Self:
        r"""Load a table from a parquet file."""
        assert str(path).endswith(".parquet")

        # Read the Parquet file using pyarrow
        table = pa.parquet.read_table(path)
        df = table.to_pandas()

        # Extract metadata
        metadata_bytes = table.schema.metadata
        metadata = {
            key.decode("utf-8"): json.loads(value.decode("utf-8"))
            for key, value in metadata_bytes.items()
            if key in [b"fkey_col_to_pkey_table", b"pkey_col", b"time_col", b"name"]
        }
        table_name = str(path).split('/')[-1].split('.')[0]
        return cls(
            df=df,
            fkey_col_to_pkey_table=metadata["fkey_col_to_pkey_table"],
            pkey_col=metadata["pkey_col"],
            time_col=metadata["time_col"],
            name=table_name,
        )

@dataclass
class DBBRDBTask:
    metadata: DBBTaskMeta
    task: BaseTask
    train_set: NamedTable
    validation_set: NamedTable
    test_set: NamedTable

class DBBRDBDataset:

    def __init__(
        self,
        dataset_path: str, 
        config_metadata_path: str,
        format: str = 'relbench',
        original_name: str = None,
        augment_task_backlinks: bool = False,
        backlink_task_indices: Optional[List[int]] = None,
    ):
        self.path = Path(dataset_path)
        assert config_metadata_path.endswith(".yaml")
        self.config_metadata_path = Path(config_metadata_path)
        self._metadata = self._load_metadata()
        self.format = format
        self.original_name = original_name
        self._augment_backlinks = augment_task_backlinks
        self._augment_task_indices: Optional[Set[int]] = (
            set(backlink_task_indices) if backlink_task_indices is not None else None
        )
        self._train_sample_indices: dict[str, np.ndarray] = {}
        if format == 'relbench':
            ## for pre-processing
            self._load_data()
            if self._augment_backlinks:
                self._augment_task_neighbor_tables()
        elif format == '4dbinfer':
            ## after generating features
            self._load_data_4dbinfer()
        else:
            raise ValueError(f"Unknown format {format}")

    def _load_metadata(self):
        return yaml_utils.load_pyd(DBBRDBDatasetMeta, self.config_metadata_path)

    @lru_cache(maxsize=1)
    def _load_data(self):
        # Load tables.
        self._tables = {}
        for table_schema in self.metadata.tables:
            table_path = self.path / table_schema.source
            table = NamedTable.load(table_path)
            table.name = table_schema.name
            self._tables[table_schema.name] = table
            # loader = get_table_data_loader(table_schema.format)
            # self._tables[table_schema.name] = loader(table_path)

        self._tasks = []
        for task_meta in self.metadata.tasks:
            task = load_relbench_task(self.metadata.dataset_name, task_meta.name)
            task.name = task_meta.name
            train_set = task.get_table('train', mask_input_cols = False)
            validation_set = task.get_table('val', mask_input_cols = False)
            test_set = task.get_table('test', mask_input_cols = False)
            self._tasks.append(DBBRDBTask(
                task_meta,
                task,
                train_set,
                validation_set,
                test_set
            ))
    
    @lru_cache(maxsize=1)
    def _load_data_4dbinfer(self):
        # Load tables.
        self._tables = {}
        for table_schema in self.metadata.tables:
            table_path = self.path / table_schema.source
            loader = get_table_data_loader(table_schema.format)
            self._tables[table_schema.name] = loader(table_path)

        # Load tasks.
        self._tasks = []
        try:
            for task_meta in self.metadata.tasks:
                loader = get_table_data_loader(task_meta.format)
                def _load_split(split):
                    table_path = self.path / task_meta.source.format(split=split)
                    return loader(table_path)
                train_set = _load_split('train')
                validation_set = _load_split('validation')
                test_set = _load_split('test')
                task = load_relbench_task(self.original_name, task_meta.name)
                self._tasks.append(DBBRDBTask(
                    task_meta, task, train_set, validation_set, test_set))
        except FileNotFoundError:
            print(f"No tasks found in {self.path}")

    @property
    def dataset_name(self) -> str:
        return self.metadata.dataset_name

    @property
    def metadata(self) -> DBBRDBDatasetMeta:
        return self._metadata

    @property
    def tasks(self) -> List[DBBRDBTask]:
        return self._tasks

    @property
    def tables(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self._tables

    def _get_table_schema_by_name(self, name: str) -> Optional[DBBTableSchema]:
        for schema in self._metadata.tables:
            if schema.name == name:
                return schema
        return None

    def _augment_task_neighbor_tables(self) -> None:
        if not self._tasks or not self._tables:
            return
        allowed = self._augment_task_indices
        augmented_pairs: Set[Tuple[str, str]] = set()
        schema_cache = {schema.name: schema for schema in self._metadata.tables}
        for idx, task in enumerate(self._tasks):
            if allowed is not None and idx not in allowed:
                continue
            target_table_name = getattr(task.metadata, "target_table", None)
            if not target_table_name or target_table_name not in self._tables:
                continue
            target_table = self._tables[target_table_name]
            target_schema = schema_cache.get(target_table_name)
            if target_schema is None or target_table.pkey_col is None:
                continue
            fkey_mapping = target_table.fkey_col_to_pkey_table or {}
            for fk_col, neighbor_name in fkey_mapping.items():
                pair_key = (target_table_name, neighbor_name)
                if neighbor_name not in self._tables or pair_key in augmented_pairs:
                    continue
                neighbor_table = self._tables[neighbor_name]
                neighbor_schema = schema_cache.get(neighbor_name)
                if neighbor_schema is None or neighbor_table.pkey_col is None:
                    continue
                self._create_augmented_task_table(
                    target_table_name,
                    target_table,
                    target_schema,
                    neighbor_name,
                    neighbor_table,
                    neighbor_schema,
                    fk_col,
                    schema_cache,
                )
                augmented_pairs.add(pair_key)

    def _create_augmented_task_table(
        self,
        target_table_name: str,
        target_table: NamedTable,
        target_schema: DBBTableSchema,
        neighbor_name: str,
        neighbor_table: NamedTable,
        neighbor_schema: DBBTableSchema,
        fk_column: str,
        schema_cache: Dict[str, DBBTableSchema],
    ) -> None:
        if fk_column not in target_table.df.columns:
            return
        neighbor_pk = neighbor_table.pkey_col
        if neighbor_pk is None or neighbor_pk not in neighbor_table.df.columns:
            return
        target_time_col = target_table.time_col
        neighbor_time_col = neighbor_table.time_col

        temp_fk_col = "__neighbor_fk_tmp__"
        target_subset = target_table.df.copy()
        target_subset[temp_fk_col] = target_subset[fk_column]

        merge_cols = [neighbor_pk]
        if neighbor_time_col and neighbor_time_col in neighbor_table.df.columns:
            merge_cols.append(neighbor_time_col)
        neighbor_keys = neighbor_table.df[merge_cols].copy()

        merged = neighbor_keys.merge(
            target_subset,
            left_on=neighbor_pk,
            right_on=temp_fk_col,
            how="left",
            suffixes=("", "_target"),
        )
        if temp_fk_col in merged.columns:
            merged.drop(columns=[temp_fk_col], inplace=True)

        if (
            neighbor_time_col
            and target_time_col
            and neighbor_time_col in merged.columns
            and target_time_col in merged.columns
        ):
            merged = merged[
                merged[target_time_col].isna()
                | (merged[target_time_col] <= merged[neighbor_time_col])
            ]

        sort_cols = [neighbor_pk]
        ascending = [True]
        if target_time_col and target_time_col in merged.columns:
            sort_cols.append(target_time_col)
            ascending.append(False)
        if len(merged) > 0:
            merged = merged.sort_values(sort_cols, ascending=ascending)
            dedup = merged.drop_duplicates(subset=[neighbor_pk], keep="first")
        else:
            dedup = merged

        if neighbor_pk in dedup.columns:
            dedup = dedup.set_index(neighbor_pk)
        else:
            dedup.index = pd.Index([], name=neighbor_pk)

        neighbor_index = pd.Index(neighbor_table.df[neighbor_pk], name=neighbor_pk)
        if len(dedup) > 0:
            aligned = dedup.reindex(neighbor_index)
        else:
            aligned = pd.DataFrame(index=neighbor_index)

        target_columns = list(target_table.df.columns)
        for column in target_columns:
            if column not in aligned.columns:
                aligned[column] = pd.NA
        aligned = aligned[target_columns].reset_index(drop=True)

        aug_pk_col = f"{neighbor_name}_{neighbor_pk}_as_{target_table_name}_aug_id"
        new_table_name = f"{target_table_name}__aug__from__{neighbor_name}"
        if new_table_name in self._tables:
            return

        aug_df = pd.DataFrame({aug_pk_col: neighbor_index.values})
        for column in target_columns:
            aug_df[column] = aligned[column].values

        new_table = NamedTable(
            name=new_table_name,
            df=aug_df,
            fkey_col_to_pkey_table=dict(target_table.fkey_col_to_pkey_table),
            pkey_col=aug_pk_col,
            time_col=target_time_col,
        )
        self._tables[new_table_name] = new_table

        pk_capacity = int(pd.Series(aug_df[aug_pk_col]).nunique())
        pk_schema = DBBColumnSchema(
            name=aug_pk_col,
            dtype=DBBColumnDType.primary_key,
            capacity=pk_capacity,
        )
        new_columns = [pk_schema]
        for column_schema in target_schema.columns:
            new_columns.append(DBBColumnSchema.model_validate(column_schema.model_dump()))
        new_schema = DBBTableSchema(
            name=new_table_name,
            source=f"augmented/{new_table_name}.parquet",
            format=DBBTableDataFormat.PARQUET,
            columns=new_columns,
            time_column=target_schema.time_column,
        )
        self._metadata.tables.append(new_schema)
        schema_cache[new_table_name] = new_schema

        fk_col_name = f"{target_table_name}_aug_ref_{neighbor_name}"
        if fk_col_name in neighbor_table.df.columns:
            return
        neighbor_table.df[fk_col_name] = neighbor_table.df[neighbor_pk]
        neighbor_table.fkey_col_to_pkey_table[fk_col_name] = new_table_name
        fk_schema = DBBColumnSchema(
            name=fk_col_name,
            dtype=DBBColumnDType.foreign_key,
            link_to=f"{new_table_name}.{aug_pk_col}",
            capacity=pk_capacity,
        )
        neighbor_schema.columns.append(fk_schema)
        schema_cache[neighbor_name] = neighbor_schema

    def subsample_train_split(self, task_name: str, sample_size: int, seed: int | None = None) -> None:
        """Subsample the training split for a given task deterministically."""
        if sample_size is None or sample_size <= 0:
            return
        target_task = None
        for task in self.tasks:
            candidate_name = getattr(task.metadata, "name", None) or getattr(task.task, "name", None)
            if candidate_name == task_name:
                target_task = task
                break
        if target_task is None:
            raise ValueError(f"Unknown task {task_name}.")
        train_split = target_task.train_set
        if hasattr(train_split, "df"):
            num_rows = len(train_split.df)
        elif isinstance(train_split, dict) and train_split:
            first_key = next(iter(train_split))
            num_rows = len(train_split[first_key])
        else:
            num_rows = len(train_split) if hasattr(train_split, "__len__") else 0
        if num_rows == 0 or sample_size >= num_rows:
            return
        subset_size = min(sample_size, num_rows)
        cached_idx = self._train_sample_indices.get(task_name)
        if (
            cached_idx is None
            or len(cached_idx) != subset_size
            or (len(cached_idx) > 0 and cached_idx.max() >= num_rows)
        ):
            base_seed = seed if seed is not None else 42
            rng = np.random.default_rng(base_seed)
            cached_idx = np.sort(rng.choice(num_rows, size=subset_size, replace=False))
            self._train_sample_indices[task_name] = cached_idx
        self._apply_train_indices(target_task, cached_idx)

    def _apply_train_indices(self, task: DBBRDBTask, indices: np.ndarray) -> None:
        train_split = task.train_set
        if hasattr(train_split, "df"):
            task.train_set.df = train_split.df.iloc[indices].reset_index(drop=True)
            return
        if isinstance(train_split, dict):
            for key, values in train_split.items():
                task.train_set[key] = np.asarray(values)[indices]
            return
        raise ValueError(f"Unsupported train set type: {type(train_split)}")

    def get_task(self, name: str) -> DBBRDBTask:
        for task in self.tasks:
            if task.metadata.name == name:
                return task
        raise ValueError(f"Unknown task {name}.")

    @property
    def sqlalchemy_metadata(self) -> sqlalchemy.MetaData:
        """Get metadata in sqlalchemy structure."""
        metadata = MetaData()
        pks, referred_pks = {}, {}
        for tbl_meta in self.metadata.tables:
            tbl_name = tbl_meta.name
            cols = []
            for col_meta in tbl_meta.columns:
                col_name = col_meta.name
                col_data = self.tables[tbl_name][col_name]
                if col_meta.dtype == DBBColumnDType.float_t:
                    if col_data.ndim == 1:
                        col = Column(col_name, Float)
                    else:
                        col = Column(col_name, ARRAY(Float))
                elif col_meta.dtype == DBBColumnDType.category_t:
                    col = Column(col_name, VARCHAR)
                elif col_meta.dtype == DBBColumnDType.datetime_t:
                    col = Column(col_name, DateTime)
                elif col_meta.dtype == DBBColumnDType.text_t:
                    col = Column(col_name, String)
                elif col_meta.dtype == DBBColumnDType.foreign_key:
                    col = Column(col_name, None, ForeignKey(col_meta.link_to))
                    link_tbl, link_col = col_meta.link_to.split('.')
                    referred_pks[link_tbl] = link_col
                elif col_meta.dtype == DBBColumnDType.primary_key:
                    col = Column(col_name, Uuid, primary_key=True)
                    pks[tbl_name] = col_name
                else:
                    col = Column(col_name, VARCHAR)
                cols.append(col)
            alchemy_tbl = Table(tbl_name, metadata, *cols)
        # Create missing tables.
        for tbl, col in referred_pks.items():
            if tbl not in pks:
                alchemy_tbl = Table(tbl, metadata, Column(col, Uuid, primary_key=True))
            elif col != pks[tbl]:
                raise ValueError(f"Detect two primary keys ({col} and {pks[tbl]}) for table '{tbl}'!")

        return metadata

    @property
    @lru_cache(maxsize=None)
    def min_timestamp(self) -> pd.Timestamp:
        r"""Return the earliest timestamp in the database."""

        return min(
            table.min_timestamp
            for table in self.table_dict.values()
            if table.time_col is not None
        )

    @property
    @lru_cache(maxsize=None)
    def max_timestamp(self) -> pd.Timestamp:
        r"""Return the latest timestamp in the database."""

        return max(
            table.max_timestamp
            for table in self.table_dict.values()
            if table.time_col is not None
        )

    def upto(self, timestamp: pd.Timestamp):
        r"""Return a database with all rows upto timestamp."""

        table_dict={
            name: table.upto(timestamp) for name, table in self._tables.items()
        }

        return table_dict



    def from_(self, timestamp: pd.Timestamp) -> Self:
        r"""Return a database with all rows from timestamp."""

        return {
            name: table.from_(timestamp) for name, table in self._tables.items()
        }

    def reindex_pkeys_and_fkeys(self) -> None:
        r"""Map primary and foreign keys into indices according to the ordering in the
        primary key tables."""
        # Get pkey to idx mapping:
        index_map_dict: Dict[str, pd.Series] = {}
        for table_name, table in self._tables.items():
            if table.pkey_col is not None:
                if table.time_col is not None:
                    table.df = table.df.sort_values(table.time_col).reset_index(
                        drop=True
                    )

                ser = table.df[table.pkey_col]

                if ser.nunique() != len(ser):
                    raise RuntimeError(
                        f"The primary key '{table.pkey_col}' "
                        f"of table '{table_name}' contains "
                        "duplicated elements"
                    )
                arange_ser = pd.RangeIndex(len(ser)).astype("Int64")
                index_map_dict[table_name] = pd.Series(
                    index=ser,
                    data=arange_ser,
                    name="index",
                )
                table.df[table.pkey_col] = arange_ser

        # Replace fkey_col_to_pkey_table with indices.
        for table in self._tables.values():
            for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
                out = pd.merge(
                    table.df[fkey_col],
                    index_map_dict[pkey_table_name],
                    how="left",
                    left_on=fkey_col,
                    right_index=True,
                )
                table.df[fkey_col] = out["index"]


def load_rdb_data(name_or_path: str) -> DBBRDBDataset:
    path = name_or_path
    return DBBRDBDataset(path)

class DBBRDBTaskCreator:

    def __init__(self, name: str):
        self.task_fields = {
            'name': name,
            'columns': {},
        }

    def set_task_type(self, task_type: DBBTaskType):
        return self.add_task_field("task_type", task_type)

    def set_evaluation_metric(self, metric: DBBTaskEvalMetric):
        return self.add_task_field("evaluation_metric", metric)

    def set_target_table(self, tbl: str):
        return self.add_task_field("target_table", tbl)

    def set_target_column(self, col: str):
        return self.add_task_field("target_column", col)

    def set_target_time_column(self, col: str):
        return self.add_task_field("time_column", col)

    def set_key_prediction_label_column(self, col: str):
        return self.add_task_field("key_prediction_label_column", col)

    def set_key_prediction_query_idx_column(self, col: str):
        return self.add_task_field("key_prediction_query_idx_column", col)

    def add_task_field(self, key: str, val: Any):
        self.task_fields[key] = val
        return self

    def add_task_data(
        self,
        name: str,
        train_data: Optional[np.ndarray],
        validation_data: Optional[np.ndarray],
        test_data: Optional[np.ndarray],
        dtype: DBBColumnDType,
        **extra_meta
    ):
        assert train_data is not None or validation_data is not None or test_data is not None
        self.task_fields['columns'][name] = {
            'name': name,
            'data': (train_data, validation_data, test_data),
            'dtype': dtype,
        }
        self.task_fields['columns'][name].update(extra_meta)
        return self

    def copy_fields_from(self, task_meta: DBBTaskMeta):
        task_meta_dict = task_meta.model_dump()
        self.task_fields = task_meta_dict
        self.task_fields['columns'] = {}
        return self

    def done(
        self,
        path: Path,
        table_format: DBBTableDataFormat = DBBTableDataFormat.NUMPY
    ) -> DBBTaskMeta:
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError(f"Provided path {path} must be a directory.")
        task_path = path / self.task_fields['name']
        task_path.mkdir(parents=True, exist_ok=True)
        # Save task table.
        train_table = {}
        val_table = {}
        test_table = {}
        col_schemas = []
        for key, col_schema in self.task_fields['columns'].items():
            train_data, val_data, test_data = col_schema.pop('data')
            if train_data is not None:
                train_table[key] = train_data
            if val_data is not None:
                val_table[key] = val_data
            if test_data is not None:
                test_table[key] = test_data

            if train_data is not None:
                # NOTE: Only write schema of training columns. Skip val/test -only
                #   columns such as `key_prediction_label_column` and
                #   `key_prediction_query_idx_column` used by retrieval tasks.
                col_schemas.append(col_schema)
        table_writer = get_table_data_writer(table_format)
        table_writer.write(task_path, "train", train_table)
        table_writer.write(task_path, "validation", val_table)
        table_writer.write(task_path, "test", test_table)
        task_meta = dict(self.task_fields)
        task_meta['format'] = table_format
        task_meta['source'] = str(table_writer.filename(task_path, "{split}").relative_to(path))
        task_meta['columns'] = col_schemas
        return DBBTaskMeta.model_validate(task_meta)

class DBBRDBDatasetCreator:

    def __init__(self, name: str):
        self.name = name
        self.tasks = []
        self.tables = {}
        self.column_groups = None

    def add_table(self, table_name: str):
        if table_name in self.tables:
            raise ValueError(f"Table {table_name} has already been added.")
        self.tables[table_name] = {"columns": {}}
        return self

    def add_column(
        self,
        table_name: str,
        column_name: str,
        data: np.ndarray,
        dtype: DBBColumnDType,
        **extra_meta
    ):
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist. Please add_table first.")

        if column_name in self.tables[table_name]:
            raise ValueError(f"Column {column_name} already exists.")

        self.tables[table_name]["columns"][column_name] = {
            'name': column_name,
            'data': data,
            'dtype': dtype,
        }
        self.tables[table_name]["columns"][column_name].update(extra_meta)

        return self

    def set_time_column(self, table: str, time_col: str):
        other_time_col = self.tables[table].get("time_column")
        if other_time_col is not None and time_col != other_time_col:
            raise ValueError(f"A table can only have one time column but got {time_col} and {other_time_col}.")
        self.tables[table]["time_column"] = time_col

    def add_task(
        self,
        task_creator: DBBRDBTaskCreator
    ):
        self.tasks.append(task_creator)
        return self

    def add_column_group(
        self,
        col_group: List[Tuple[str, str]]
    ):
        if self.column_groups is None:
            self.column_groups = []
        col_group = [DBBColumnID(table=tbl, column=col) for tbl, col in col_group]
        self.column_groups.append(col_group)
        return self

    def replace_tables_from(
        self,
        other: DBBRDBDataset, 
        augmented_tables: Dict[str, pd.DataFrame] = None
    ):
        self.tables = {}
        for table_schema in other.metadata.tables:
            table_name = table_schema.name
            self.add_table(table_name)
            self.set_time_column(table_name, table_schema.time_column)
            for col_schema in table_schema.columns:
                col_name = col_schema.name
                col_schema = col_schema.model_dump()
                if augmented_tables is not None and table_name in augmented_tables \
                    and col_name in augmented_tables[table_name].df:
                    col_data = augmented_tables[table_name].df[col_name].values
                else:
                    continue
                self.add_column(
                    table_name,
                    col_name,
                    col_data,
                    **col_schema
                )
        return self

    def _validate(self):
        for table_name, table_info in self.tables.items():
            table_size = None
            for col_name, col_info in table_info["columns"].items():
                if table_size is None:
                    table_size = len(col_info['data'])
                elif len(col_info['data']) != table_size:
                    raise ValueError(
                        f"Expect all columns to have the same length."
                        f" But got {col_info['data']} and {table_size}."
                    )

    def done(
        self,
        path: Path,
        table_format: DBBTableDataFormat = DBBTableDataFormat.NUMPY
    ):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError(f"Provided path {path} must be a directory.")

        self._validate()

        # Write task data.
        tasks = [task_ctor.done(path, table_format) for task_ctor in self.tasks]

        # Write table data.
        table_writer = get_table_data_writer(table_format)
        schemas = []
        for table_name, table_info in self.tables.items():
            data_dir = path / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)
            table_data = {}
            col_schemas = []
            for col_name, col_info in table_info["columns"].items():
                table_data[col_name] = col_info.pop("data")
                col_schemas.append(col_info)
            table_writer.write(data_dir, table_name, table_data)
            source = str(table_writer.filename(data_dir, table_name).relative_to(path))
            schema = DBBTableSchema.model_validate({
                'name': table_name,
                'source': source,
                'format': table_format,
                'columns': col_schemas,
                'time_column': table_info.get('time_column')
            })
            schemas.append(schema)

        metadata = DBBRDBDatasetMeta(
            dataset_name=self.name,
            tables=schemas,
            tasks=tasks,
        )

        yaml_utils.save_pyd(metadata, path / 'metadata.yaml')
