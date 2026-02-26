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


from typing import Tuple, Dict, Optional, List
from collections import defaultdict
import numpy as np
import pandas as pd
import pydantic
from dbinfer_bench import DBBColumnDType
from loguru import logger
from ...device import DeviceInfo
from .base import (
    RDBTransform,
    get_column_transform_class,
    ColumnData,
    RDBData,
    is_task_table,
    unmake_task_table_name,
)
from pathlib import Path


def _get_train_mask(
    rdb_data: RDBData,
    tbl_name: str,
    num_rows: int
) -> np.ndarray:
    """
    Get a boolean mask indicating which rows are in the training set (before test_timestamp_cutoff).
    Note: In this setting, validation data is visible, so we use test_timestamp_cutoff.

    Returns:
        train_mask: Boolean array where True indicates training+validation data
    """
    # Check if we have a test timestamp cutoff
    if rdb_data.test_timestamp_cutoff is None:
        logger.warning(
            f"No test_timestamp_cutoff found. Fitting on ENTIRE dataset. "
            "This will cause data leakage!"
        )
        return np.ones(num_rows, dtype=bool)

    # Check if this table has a time column
    if rdb_data.time_column_name is None or tbl_name not in rdb_data.time_column_name:
        logger.warning(
            f"No time column found for table {tbl_name}. Fitting on ENTIRE table. "
            "Please double check if this is the desired behavior. Only do this if you are sure that the table has no time column."
        )
        return np.ones(num_rows, dtype=bool)

    time_col_name = rdb_data.time_column_name[tbl_name]

    # Get the time column data
    if time_col_name not in rdb_data.tables[tbl_name]:
        logger.warning(f"Time column {time_col_name} not found in table {tbl_name}")
        return np.ones(num_rows, dtype=bool)

    time_data = rdb_data.tables[tbl_name][time_col_name].data
    test_cutoff = rdb_data.test_timestamp_cutoff

    # Normalize both time_data and test_cutoff to the same type for comparison
    try:
        # Check if time_data is pandas datetime or numpy datetime64
        if hasattr(time_data, 'dtype') and np.issubdtype(time_data.dtype, np.datetime64):
            # time_data is datetime64, convert cutoff to datetime64
            if isinstance(test_cutoff, str):
                test_cutoff = pd.Timestamp(test_cutoff)
            elif isinstance(test_cutoff, (int, np.integer)):
                test_cutoff = pd.Timestamp(test_cutoff)
        else:
            # time_data is numeric (int64), convert cutoff to int64
            if isinstance(test_cutoff, str):
                test_cutoff = pd.Timestamp(test_cutoff).value
            elif hasattr(test_cutoff, 'value'):
                test_cutoff = test_cutoff.value

        # Create the training+validation data mask (before test)
        train_mask = time_data < test_cutoff

        num_train = np.sum(train_mask)
        logger.info(f"Table {tbl_name}: {num_train}/{num_rows} rows in train+val set (before {rdb_data.test_timestamp_cutoff})")

        if num_train == 0:
            logger.warning(
                f"No training data found for {tbl_name} before cutoff {test_cutoff}. "
                "Falling back to full dataset."
            )
            return np.ones(num_rows, dtype=bool)

        return train_mask

    except Exception as e:
        logger.error(f"Could not parse timestamps in {tbl_name}.{time_col_name}: {e}")
        logger.warning("Falling back to fitting on full dataset")
        return np.ones(num_rows, dtype=bool)


class RDBTransformWrapper(RDBTransform):

    def __init__(
        self,
        col_xform_name : str,
        col_xform_config : Dict,
        target_columns : Optional[Dict[str, str]] = None,
        path_to_save_scalers : Optional[Path] = None,
    ):
        self.col_xform_class = get_column_transform_class(col_xform_name)
        self.col_xform_config = self.col_xform_class.config_class.model_validate(col_xform_config)
        self.col_groups = []
        self.col_xforms = []
        self.target_columns = target_columns or {}  # task_name -> target_column mapping
        self.path_to_save_scalers = path_to_save_scalers


    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        if self.col_xform_class.input_dtype == DBBColumnDType.text_t:
            self.col_groups = self.get_text_column_groups(rdb_data)
        else:
            self.col_groups = self.get_column_groups(rdb_data)

        if len(self.col_groups) == 0:
            return

        # Create & fit transforms.
        for cg in self.col_groups:
            logger.info(f"Fitting transform {self.col_xform_class.name} for column group {cg}.")
            # Pass target_columns to norm_numeric transform
            if self.col_xform_class.name == "norm_numeric":
                col_xform = self.col_xform_class(self.col_xform_config, self.target_columns)
            else:
                col_xform = self.col_xform_class(self.col_xform_config)
            cg_meta = _get_cg_meta(cg, rdb_data)
            # Add column and table info for caching and target column identification
            if len(cg) == 1:  # Single column group
                tbl_name, col_name = cg[0]
                cg_meta['table_name'] = tbl_name
                cg_meta['column_name'] = col_name
                cg_meta['target_columns'] = self.target_columns
                if self.path_to_save_scalers:
                    cg_meta['path_to_save_scalers'] = self.path_to_save_scalers

            # Apply temporal filtering: only use training data for fitting
            cg_data_list = []
            for tbl_name, col_name in cg:
                full_data = rdb_data.tables[tbl_name][col_name].data

                # Get training mask for this table
                train_mask = _get_train_mask(rdb_data, tbl_name, len(full_data))

                # Filter to only training data
                train_data = full_data[train_mask]
                cg_data_list.append(train_data)

            cg_data = np.concatenate(cg_data_list, axis=0)
            col_xform.fit(ColumnData(cg_meta, cg_data), device)
            self.col_xforms.append(col_xform)

    def get_text_column_groups(self, rdb_data : RDBData):
        text_cg = []
        for tbl_name, table in rdb_data.tables.items():
            for col_name, col in table.items():
                if col.metadata['dtype'] == DBBColumnDType.text_t:
                    text_cg.append((tbl_name, col_name))
        if len(text_cg) == 0:
            return []
        else:
            return [text_cg]

    def get_column_groups(self, rdb_data : RDBData):
        # Get column group.
        col_groups = []
        scanned = set()
        # Add column groups explicitly defined.
        if rdb_data.column_groups is not None:
            for cg in rdb_data.column_groups:
                cg_meta = _get_cg_meta(cg, rdb_data)
                if cg_meta['dtype'] == self.col_xform_class.input_dtype:
                    col_groups.append(cg)
                    for tbl_name, col_name in cg:
                        scanned.add((tbl_name, col_name))
        # Scan rest of the columns.
        for tbl_name, table in rdb_data.tables.items():
            for col_name, col in table.items():
                if (tbl_name, col_name) in scanned:
                    continue
                if col.metadata['dtype'] == self.col_xform_class.input_dtype:
                    col_groups.append([(tbl_name, col_name)])
        return col_groups

    def _is_target_column(self, tbl_name: str, col_name: str) -> bool:
        """Check if a column is a target column for any task."""
        if is_task_table(tbl_name):
            task_name, target_table = unmake_task_table_name(tbl_name)
            target_col = self.target_columns.get(task_name)
            return target_col is not None and col_name == target_col
        else:
            # For regular tables, check if this column is a target column for any task
            for task_name, target_col in self.target_columns.items():
                if col_name == target_col:
                    return True
            return False

    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo,
        metadata : Dict[str, str] = {}
    ) -> RDBData:
        to_xform = {}
        for cg, xform in zip(self.col_groups, self.col_xforms):
            for tbl_name, col_name in cg:
                to_xform[(tbl_name, col_name)] = xform
        new_tables = {tbl_name : {} for tbl_name in rdb_data.tables}
        col_name_mapping = defaultdict(list)
        for tbl_name, table in rdb_data.tables.items():
            for col_name, col in table.items():
                # Get transform.
                xform = to_xform.get((tbl_name, col_name), None)
                if xform is None and is_task_table(tbl_name):
                    _, target_table = unmake_task_table_name(tbl_name)
                    xform = to_xform.get((target_table, col_name), None)
                # Conduct transformation.
                if xform is not None:
                    logger.info(f"Transforming table/column {tbl_name}/{col_name}.")
                    new_cols = xform.transform(col, device, metadata)
                    assert len(new_cols) == len(xform.output_name_formatters)
                    for formatter, new_col in zip(xform.output_name_formatters, new_cols):
                        if len(new_col.data) != len(col.data):
                            raise ValueError(
                                "Column transform is not allowed to change the length of columns."
                                f" {len(col.data)} -> {len(new_col.data)}."
                            )
                        new_col_name = formatter.format(name=col_name)
                        new_tables[tbl_name][new_col_name] = new_col
                        col_name_mapping[(tbl_name, col_name)].append((tbl_name, new_col_name))
                else:
                    new_tables[tbl_name][col_name] = col
                    col_name_mapping[(tbl_name, col_name)].append((tbl_name, col_name))
        col_name_mapping = dict(col_name_mapping)
        # Re-create column groups.
        column_groups = None
        if rdb_data.column_groups is not None:
            column_groups = []
            for cg in rdb_data.column_groups:
                mapped_cg = [col_name_mapping[name_pair] for name_pair in cg]
                for i in range(len(mapped_cg[0])):
                    column_groups.append([mapped_cg[j][i] for j in range(len(mapped_cg))])
        return RDBData(new_tables, column_groups, rdb_data.relationships, rdb_data.val_timestamp_cutoff, rdb_data.test_timestamp_cutoff, rdb_data.time_column_name)

def _get_cg_meta(cg, rdb_data):
    tbl_name, col_name = cg[0]
    return rdb_data.tables[tbl_name][col_name].metadata
