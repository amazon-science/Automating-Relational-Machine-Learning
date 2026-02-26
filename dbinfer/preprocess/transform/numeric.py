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


import copy
from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import logging
import pickle
from pathlib import Path
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)
from loguru import logger
import os

class NormNumericTransformConfig(pydantic.BaseModel):
    impute_strategy : Optional[str] = "median"
    skew_threshold : Optional[float] = 0.99

@column_transform
class NormNumericTransform(ColumnTransform):
    config_class = NormNumericTransformConfig
    name = "norm_numeric"
    input_dtype = DBBColumnDType.float_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : NormNumericTransformConfig, target_columns=None):
        super().__init__(config)
        self.target_columns = target_columns or {}
        self.target_scalers = {}  # Store scalers for target columns only

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        # Note: temporal filtering is handled by RDBTransformWrapper.fit()
        # which only passes training data (before val_timestamp_cutoff) to this method
        self.new_meta = {
            'dtype' : self.output_dtypes[0],
            'in_size' : 1 if column.data.ndim == 1 else column.data.shape[1]
        }

        if column.data.ndim > 1:
            # Ignore vector embeddings.
            return

        # Check if this is a target column
        table_name = column.metadata.get('table_name')
        column_name = column.metadata.get('column_name')
        target_columns = column.metadata.get('target_columns')
        if column_name in list(target_columns.values()) and '__task__' in table_name:
            # For target columns, always use StandardScaler and save it
            logger.debug(f'\tTarget column detected: {column_name}. Using StandardScaler.')
            self.transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.config.impute_strategy, add_indicator=True)),
                ('scaler', StandardScaler())
            ])
            self.transformer.fit(column.data.reshape(-1, 1))
            scaler_key = f"{table_name}:{column_name}" if table_name and column_name else column_name
            self.target_scalers[scaler_key] = self.transformer
            path_to_save_scalers = column.metadata.get('path_to_save_scalers')
            if path_to_save_scalers:
                self.save_target_scalers(path_to_save_scalers)
        else:
            # Regular logic for non-target columns
            skew_score = pd.Series(column.data).skew()
            if np.abs(skew_score) > self.config.skew_threshold:
                logger.debug(f'\tSkew values detected. Skew score: {skew_score:.6f}.')
                self.transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=self.config.impute_strategy)),
                    ('scaler', QuantileTransformer(output_distribution='normal'))])
            else:
                self.transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy=self.config.impute_strategy)),
                    ('scaler', StandardScaler())])
            try:
                self.transformer.fit(column.data.reshape(-1, 1))
            except Exception as e:
                logger.error(f'Error fitting column {column_name} in table {table_name}: {e}')
                column.data = np.zeros_like(column.data)

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo,
        metadata : Dict[str, str] = {}
    ) -> List[ColumnData]:
        if column.data.size == 0 or np.all(np.isnan(column.data)):
            return [ColumnData(self.new_meta, column.data)]
        if column.data.ndim > 1:
            new_data = column.data.astype('float32')
        else:
            # Check if this is a target column and use saved scaler if available
            table_name = column.metadata.get('table_name')
            column_name = column.metadata.get('column_name')
            scaler_key = f"{table_name}:{column_name}" if table_name and column_name else None
            
            if scaler_key and scaler_key in self.target_scalers:
                transformer = self.target_scalers[scaler_key]
                logger.debug(f'\tUsing saved target scaler for: {scaler_key}')
            else:
                transformer = self.transformer
            
            try:
                new_data = transformer.transform(column.data.reshape(-1, 1)).reshape(-1)
            except Exception as e:
                logger.error(f'Error transforming column {column_name} in table {table_name}: {e}')
                new_data = np.zeros_like(column.data)
            new_data = new_data.astype('float32')
        return [ColumnData(self.new_meta, new_data)]

    def save_target_scalers(self, output_path: Path) -> None:
        """Save only the target column scalers."""
        if self.target_scalers:
            scalers_path = Path(output_path) / 'target_scalers.pkl'
            os.makedirs(output_path, exist_ok=True)
            with open(scalers_path, 'wb') as f:
                pickle.dump(self.target_scalers, f)
            logger.debug(f'Saved target scalers to: {scalers_path}')

    def load_target_scalers(self, input_path: Path) -> None:
        """Load target column scalers."""
        scalers_path = input_path / 'target_scalers.pkl'
        if scalers_path.exists():
            with open(scalers_path, 'rb') as f:
                self.target_scalers = pickle.load(f)
            logger.debug(f'Loaded target scalers from: {scalers_path}')
        else:
            logger.debug(f'No target scalers found at: {scalers_path}')

    def _is_target_column(self, table_name: str, column_name: str) -> bool:
        """Check if a column is a target column for any task."""
        if not table_name or not column_name:
            return False
            
        from .base import is_task_table, unmake_task_table_name
        
        if is_task_table(table_name):
            task_name, target_table = unmake_task_table_name(table_name)
            target_col = self.target_columns.get(task_name)
            return target_col is not None and column_name == target_col
        else:
            # For regular tables, check if this column is a target column for any task
            for task_name, target_col in self.target_columns.items():
                if column_name == target_col:
                    return True
            return False
