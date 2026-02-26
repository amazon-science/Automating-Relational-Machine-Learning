from typing import Optional

import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from copy import deepcopy
import os

from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    macro_f1,
    mae,
    micro_f1,
    mrr,
    r2,
    rmse,
    roc_auc,
)

from .database import Database
from .dataset import Dataset
from .table import Table
from .task_base import TaskType
from .task_entity import EntityTask


class AutoCompleteTask(EntityTask):
    r"""Auto complete column task on a dataset. Predict all values in the target column.

    The task is constructed by specifying the entity table, entity column, time column, and target column.
    The target column is removed from the entity table and saved to `db.table_dict[entity_table].removed_cols`,
    which is used to construct the table for the predict column task.

    The entity table needs to have a time column by which the data is split into training and validation set.

    Args:
        dataset: The dataset object.
        task_type: The type of the task.
        entity_table: The name of the entity table.
        entity_col: The name of the entity (id) column. Can be None if the entity table has no id column.
        time_col: The name of the time column by which the data is split into training and validation set.
        target_col: The name of the target column to be predicted.
        cache_dir: The directory to cache the task tables.
    """

    timedelta = pd.Timedelta(days=1)
    entity_col: str
    auto: bool = True

    def __init__(
        self,
        dataset: Dataset,
        task_type: TaskType,
        entity_table: str,
        target_col: str,
        cache_dir: Optional[str] = None,
        remove_columns: list[tuple[str, str]] = [],
        pre_defined = False,
        pre_defined_table_path = None,
        pre_defined_task_name = "",
        time_column = "timestamp_first_active",
        fk_column = "id",

    ):
        super().__init__(dataset, cache_dir=cache_dir)

        self.task_type = task_type
        self.entity_table = entity_table
        self.target_col = target_col
        self.remove_columns = remove_columns
        ## if pre_defined is True, will load a pre-defined table for the task
        self.pre_defined = pre_defined
        self.pre_defined_table_path = os.path.join(os.getenv('XDG_CACHE_HOME', 'cache_data'), pre_defined_table_path) if pre_defined_table_path is not None else None
        self.pre_defined_task_name = pre_defined_task_name

        # If using pre-defined tables, skip database modification
        # We'll get metadata from the database without modifying it
        self.dataset.target_col = target_col
        self.dataset.entity_table = entity_table
        self.dataset.remove_columns = remove_columns
        self.dataset.get_db.cache_clear()
        db = self.dataset.get_db()

        entity_col = db.table_dict[entity_table].pkey_col
        self.entity_col = entity_col if entity_col is not None else "primary_key"
        self.time_col = db.table_dict[entity_table].time_col
        self.transform = {'train': False, 'val': False, 'test': False}
        self.num_classes = None  # FIXME HACK
        if self.task_type == TaskType.REGRESSION:
            self.metrics = [r2, mae, rmse]
        else:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                self.metrics = [average_precision, accuracy, f1, roc_auc]
            elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                self.metrics = [accuracy, macro_f1, micro_f1]

            # Only initialize encoder for non-predefined tasks
            # For pre-defined tasks, encoder will be initialized lazily when needed
            if not self.pre_defined:
                removed_cols = db.table_dict[self.entity_table].removed_cols
                db = db.upto(self.dataset.val_timestamp)
                train_ids = db.table_dict[self.entity_table].df[self.entity_col].values
                # Get the training targets as a Pandas Series first
                train_target_series = removed_cols.loc[
                    removed_cols[self.entity_col].isin(train_ids), self.target_col
                ]

                clean_train_targets = train_target_series.dropna().values

                clean_train_targets = clean_train_targets[clean_train_targets != -1]

                self.target_encoder = OrdinalEncoder(
                    unknown_value=-1, handle_unknown="use_encoded_value", dtype="int64"
                )
                self.target_encoder.fit(clean_train_targets.reshape(-1, 1))
                num_classes = (
                    self.target_encoder.categories_[0].shape[0] + 1
                )  # +1 for unknown class
                self.num_classes = num_classes
            else:
                # For pre-defined tasks, encoder will be initialized from training data
                self.target_encoder = None

    def filter_dangling_entities(self, db: Database, table: Table) -> Table:
        num_entities = len(db.table_dict[self.entity_table])
        filter_mask = table.df[self.entity_col] >= num_entities

        if filter_mask.any():
            table.df = table.df[~filter_mask]

        return table

    def _get_table(self, split: str) -> Table:
        r"""Helper function to get a table for a split.

        This function overrides the `_get_table` method in `EntityTask`.

        If pre_defined is True, loads the table from pre-defined parquet files.
        Otherwise, generates the table using time-aware SQL queries.
        """

        # Load from pre-defined parquet file if pre_defined is True
        if self.pre_defined:
            return self._load_predefined_table(split)

        # Otherwise, use the original time-aware SQL generation
        db = self.dataset.get_db(upto_test_timestamp=split != "test")

        if split == "train":
            start = self.dataset.val_timestamp - self.timedelta
            end = db.min_timestamp
            freq = -self.timedelta

        elif split == "val":
            if self.dataset.val_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "val timestamp + timedelta is larger than max timestamp! "
                    "This would cause val labels to be generated with "
                    "insufficient aggregation time."
                )

            start = self.dataset.test_timestamp - self.timedelta
            end = self.dataset.val_timestamp
            freq = -self.timedelta

        elif split == "test":
            if self.dataset.test_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )

            start = db.max_timestamp
            end = self.dataset.test_timestamp
            freq = -self.timedelta

        timestamps = pd.date_range(start=start, end=end, freq=freq)

        if split == "train" and len(timestamps) < 3:
            raise RuntimeError(
                f"The number of training time frames is too few. "
                f"({len(timestamps)} given)"
            )

        table = self.make_table(db, timestamps)
        table = self.filter_dangling_entities(db,table)

        return table

    def _convert_predefined_table(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Return a relbench-compatible table (time, fk, target only)."""
        # Determine which column in the pre-defined table corresponds to time.
        if self.time_col in df.columns:
            time_column = self.time_col
        elif "timestamp" in df.columns:
            time_column = "timestamp"
        else:
            raise ValueError(
                f"Pre-defined table must contain the provided time column '{self.time_col}' "
                "or a generic 'timestamp' column."
            )

        required_cols = [self.entity_col, time_column, target_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Pre-defined table missing required columns: {missing_cols}. "
                f"Found columns: {df.columns.tolist()}"
            )

        relbench_df = df[[self.entity_col, time_column, target_column]].copy()

        rename_map = {}
        if time_column != self.time_col:
            rename_map[time_column] = self.time_col
        if target_column != self.target_col:
            rename_map[target_column] = self.target_col
        if rename_map:
            relbench_df = relbench_df.rename(columns=rename_map)

        return relbench_df

    def _load_predefined_table(self, split: str) -> Table:
        r"""Load pre-defined task table from parquet file.

        The table should contain:
        1. Foreign key to the entity table (self.entity_col)
        2. Time column ('timestamp' - will be used for self.time_col)
        3. Label column (self.target_col)

        Args:
            split: One of 'train', 'val', or 'test'

        Returns:
            Table object with the loaded data and appropriate metadata
        """
        # Map 'val' to 'validation' for file naming convention
        file_split = 'validation' if split == 'val' else split

        # Construct the path to the parquet file
        parquet_path = os.path.join(
            self.pre_defined_table_path,
            self.pre_defined_task_name,
            f"{file_split}.pqt"
        )

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Pre-defined task table not found at {parquet_path}"
            )

        # Load the parquet file
        df = pd.read_parquet(parquet_path)

        # Pre-defined tables use 'timestamp' as the time column name.
        # Auto-detect target column if needed, then drop all other columns.
        actual_target_col = self.target_col
        if self.target_col not in df.columns:
            candidate_cols = [col for col in df.columns if col not in [self.entity_col, 'timestamp']]
            if len(candidate_cols) == 0:
                raise ValueError(
                    f"Pre-defined table has no candidate target column. "
                    f"Found columns: {df.columns.tolist()}"
                )
            actual_target_col = candidate_cols[0]
            print(f"Note: Target column '{self.target_col}' not found. Using '{actual_target_col}' instead.")

        df_filtered = self._convert_predefined_table(df, actual_target_col)

        # Create foreign key mapping (entity_col -> entity_table)
        fkey_col_to_pkey_table = {self.entity_col: self.entity_table}

        # Create and return the Table object
        return Table(
            df=df_filtered,
            fkey_col_to_pkey_table=fkey_col_to_pkey_table,
            pkey_col=None,
            time_col=self.time_col,
        )

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        entity_table = db.table_dict[self.entity_table].df  # noqa: F841
        entity_table_removed_cols = db.table_dict[  # noqa: F841
            self.entity_table
        ].removed_cols

        if entity_table_removed_cols is None:
            entity_table_removed_cols = db.table_dict[self.entity_table].df[
                [self.target_col]
            ].copy()

        time_col = db.table_dict[self.entity_table].time_col
        entity_col = db.table_dict[self.entity_table].pkey_col
        
        # SOLUTION 1 FIX: Get ALL foreign keys from the original entity table
        entity_table_obj = db.table_dict[self.entity_table]
        original_fkeys = entity_table_obj.fkey_col_to_pkey_table.copy()

        # Calculate minimum and maximum timestamps from timestamp_df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        min_timestamp = timestamp_df["timestamp"].min()
        max_timestamp = timestamp_df["timestamp"].max()

        # SOLUTION 1 FIX: Include ALL foreign key columns in SELECT
        fkey_columns = list(original_fkeys.keys())
        if fkey_columns:
            # Filter out foreign key columns that might have been removed
            available_fkey_columns = []
            for col in fkey_columns:
                if col in entity_table.columns:
                    available_fkey_columns.append(col)
                else:
                    print(f"Warning: Foreign key column '{col}' not found in entity table. Skipping.")
            
            if available_fkey_columns:
                # Build SELECT clause for available foreign key columns
                fkey_select_columns = [f"entity_table.{col}" for col in available_fkey_columns]
                fkey_select_str = ",\n                " + ",\n                ".join(fkey_select_columns)
                # Update original_fkeys to only include available columns
                original_fkeys = {k: v for k, v in original_fkeys.items() if k in available_fkey_columns}
            else:
                fkey_select_str = ""
                original_fkeys = {}
        else:
            fkey_select_str = ""

        df = duckdb.sql(
            f"""
            SELECT
                entity_table.{time_col},
                entity_table.{entity_col}{fkey_select_str},
                entity_table_removed_cols.{self.target_col}
            FROM
                entity_table
            LEFT JOIN
                entity_table_removed_cols
            ON
                entity_table.{entity_col} = entity_table_removed_cols.{entity_col}
            WHERE
                entity_table.{time_col} > '{min_timestamp}' AND
                entity_table.{time_col} <= '{max_timestamp}' 
                AND (CASE 
                    WHEN typeof(entity_table_removed_cols.{self.target_col}) = 'integer' 
                    THEN entity_table_removed_cols.{self.target_col} != -1
                    ELSE 1
                END)
            """
        ).df()

        # remove rows where self.target_col is nan
        df = df.dropna(subset=[self.target_col])

        # SOLUTION 1 FIX: Include ALL foreign key relationships in task table
        complete_fkey_mapping = original_fkeys.copy()
        complete_fkey_mapping[entity_col] = self.entity_table

        return Table(
            df=df,
            fkey_col_to_pkey_table=complete_fkey_mapping,  # Preserve ALL relationships
            pkey_col=None,
            time_col=time_col,
        )

    def transform_target(self, target_col: pd.Series) -> pd.Series:
        # 1. Transform the original column. Known categories are converted to
        # integers, and unknown categories become -1.
        transformed_values = self.target_encoder.transform(
            target_col.values.reshape(-1, 1)
        ).flatten()

        # 2. Create a new Series with the transformed values, preserving the
        # original index from target_col.
        transformed_series = pd.Series(
            transformed_values, index=target_col.index, name=target_col.name
        )

        # 3. Filter the Series. Keep only the rows where the transformed value
        # is NOT -1. This effectively drops all rows that contained an
        # unknown category.
        filtered_series = transformed_series[transformed_series != -1]

        return filtered_series

    def get_table(self, split, mask_input_cols=None):
        table = super().get_table(split, mask_input_cols=False)
        table = deepcopy(table)
        if self.task_type in (TaskType.MULTICLASS_CLASSIFICATION, TaskType.BINARY_CLASSIFICATION, 'binary_classification', 'multiclass_classification'):
            # Initialize encoder lazily for pre-defined tasks
            if self.pre_defined and self.target_encoder is None and split == 'train':
                if self.target_col in table.df:
                    clean_train_targets = table.df[self.target_col].dropna().values
                    clean_train_targets = clean_train_targets[clean_train_targets != -1]
                    self.target_encoder = OrdinalEncoder(
                        unknown_value=-1, handle_unknown="use_encoded_value", dtype="int64"
                    )
                    self.target_encoder.fit(clean_train_targets.reshape(-1, 1))
                    self.num_classes = (
                        self.target_encoder.categories_[0].shape[0] + 1
                    )  # +1 for unknown class

            if self.target_col in table.df:
                transformed_target = self.transform_target(
                    table.df[self.target_col]
                )
                table.df = table.df.loc[transformed_target.index]
                table.df[self.target_col] = transformed_target
        if mask_input_cols != None:
            need_mask = mask_input_cols
        else:
            need_mask = (split == 'test')
        if need_mask:
            ## remove target column here
            table = self._mask_input_cols(table)
        return table
