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
import logging
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)
from loguru import logger

class RemapCategoryTransformConfig(pydantic.BaseModel):
    pass

@column_transform
class RemapCategoryTransform(ColumnTransform):
    config_class = RemapCategoryTransformConfig
    name = "remap_category"
    input_dtype = DBBColumnDType.category_t
    output_dtypes = [DBBColumnDType.category_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : RemapCategoryTransformConfig):
        super().__init__(config)

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        # Note: temporal filtering is handled by RDBTransformWrapper.fit()
        # which only passes training data (before val_timestamp_cutoff) to this method
        if column.data.ndim > 1:
            raise ValueError("RemapCategoryTransform only supports 1D data.")
        _, self.categories = pd.factorize(column.data, use_na_sentinel=True)
        self.unseen_category = len(self.categories)

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo,
        metadata : Dict[str, str] = {}
    ) -> List[ColumnData]:
        if column.data.ndim > 1:
            raise ValueError("RemapCategoryTransform only supports 1D data.")

        new_data = pd.Categorical(column.data, categories=self.categories).codes.copy()
        new_data[new_data == -1] = self.unseen_category
        new_data = new_data.astype('int64')
        new_meta = copy.deepcopy(column.metadata)
        new_meta['num_categories'] = len(self.categories) + 1

        return [ColumnData(new_meta, new_data)]
