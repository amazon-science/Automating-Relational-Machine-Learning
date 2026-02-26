import logging
from typing import Tuple, Dict, Optional, List, Any, Literal
from pathlib import Path
import wandb
from enum import Enum
import warnings
import pydantic

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dbinfer_bench import DBBRDBDataset, DBBColumnDType, DBBTaskType
from tabpfn import TabPFNClassifier, TabPFNRegressor

from .base import (
    TabularMLSolution,
    FitSummary,
    tabml_solution,
)
from .tabular_dataset_config import TabularDatasetConfig
from ..device import DeviceInfo
from .. import yaml_utils
from ..evaluator import get_metric_fn
from loguru import logger
from ..evaluator import get_metrics
import pickle

__all__ = ['TabPFNSolution', 'TabPFNSolutionConfig']

class TabPFNSolutionConfig(pydantic.BaseModel):
    # Number of estimators in the TabPFN model.
    n_estimators: int = 8
    # Maximum number of training samples to use. None means no limit.
    max_train_samples: Optional[int] = None
    # Strategy for selecting training samples.
    training_sample_selection_strategy: Literal['graph', 'random', 'stratified'] = 'random'
    # Whether to use foreign keys as features.
    use_foreign_key_feature: bool = True
    # Batch size for evaluation
    eval_batch_size: int = 1024
    # Use auto TabPFN classifier/regressor
    use_auto_tabpfn: bool = False
    # Number of trials for auto TabPFN classifier/regressor
    num_trials: int = 60
    # Whether to remove add_feature columns
    remove_add_feature: bool = True
    # Whether to remove categorical temporal information
    remove_cat_temporal_information: bool = True
    # whether to use the no-text datasets or normal datasets
    use_no_text_datasets: bool = False


@tabml_solution
class TabPFNSolution(TabularMLSolution):
    """TabPFN solution class."""
    config_class = TabPFNSolutionConfig
    name = "tabpfn"

    def __init__(
        self,
        solution_config: TabPFNSolutionConfig,
        data_config: TabularDatasetConfig
    ):
        super().__init__(solution_config, data_config)
        self.predictor = None
        self.label_categories: Optional[np.ndarray] = None

    def fit(
        self,
        dataset: DBBRDBDataset,
        task_name: str,
        ckpt_path: Path,
        device: DeviceInfo,
        scaler_path: Path
    ) -> FitSummary:
        _dataset = dataset.get_task(task_name)
        train_feat_store, valid_feat_store = \
            _dataset.train_set, _dataset.validation_set

        logger.info(f"Number of columns before adjustment: {len(train_feat_store.keys())}")
        train_feat_dict, feat_meta = self.adjust_features(train_feat_store, self.solution_config.remove_add_feature, self.solution_config.remove_cat_temporal_information, dataset.get_task(task_name).metadata.entity_column)
        self.entity_id_column = dataset.get_task(task_name).metadata.entity_column
        logger.info(f"Number of columns after adjustment: {len(train_feat_dict.keys())}")
        
        # Fix 2D arrays to 1D for pandas DataFrame
        fixed_feat_dict = {}
        for key, value in train_feat_dict.items():
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 1:
                fixed_feat_dict[key] = value.squeeze()
            else:
                fixed_feat_dict[key] = value
        
        train_df = pd.DataFrame(fixed_feat_dict, copy=False)

        task_type = self.data_config.task.task_type

        # Prepare features and labels
        X_train, y_train = self.extract_features_and_label(train_df, task_type, scaler_path)

        self.X_train = X_train
        self.y_train = y_train

        self.select_and_fit(target_set=train_df)

        self.checkpoint(ckpt_path)

        # Calculate metrics
        #train_metric = self.calculate_metric(X_train, y_train, train_df)
        train_metric = 0.0

        # Log to wandb
        logger.info(f"Training metric: {train_metric}")

        summary = FitSummary()
        summary.val_metric = 0.0  # Placeholder, as we don't calculate validation metric here
        summary.train_metric = train_metric

        return summary

    def select_and_fit(
        self,
        target_set: pd.DataFrame,
    ):
        # Initialize the appropriate model based on the task
        if self.data_config.task.task_type == DBBTaskType.classification:
            self.predictor = TabPFNClassifier(
                    n_estimators=self.solution_config.n_estimators,
                    n_jobs=-1,
                    random_state=42,
                    inference_config={"SUBSAMPLE_SAMPLES": min(10000, len(target_set))},
                    ignore_pretraining_limits=True,
                )
        elif self.data_config.task.task_type == DBBTaskType.regression:
            self.predictor = TabPFNRegressor(
                    n_estimators=self.solution_config.n_estimators,
                    n_jobs=-1,
                    random_state=42,
                    inference_config={"SUBSAMPLE_SAMPLES": min(10000, len(target_set))},
                    ignore_pretraining_limits=True,
                )
        else:
            raise ValueError(f"Unsupported task type: {self.data_config.task.task_type}")


        if self.solution_config.max_train_samples:
            # Select training samples
            selected_X_train, selected_y_train, _ = self.select_training_samples(
                self.X_train, self.y_train,
                target_set=target_set,
                max_samples=10000 if self.solution_config.max_train_samples == 'None' else int(self.solution_config.max_train_samples),
                strategy=self.solution_config.training_sample_selection_strategy)
        else:
            # Use all training samples
            selected_X_train = self.X_train
            selected_y_train = self.y_train

        logger.info(f"Fitting on {len(selected_X_train)} samples. # of features: {selected_X_train.shape[1]}")
        logger.info(f"Label distribution in training set: {np.unique(selected_y_train, return_counts=True)}")

        # Fit the model
        self.predictor.fit(selected_X_train, selected_y_train)

    def evaluate(
        self,
        table: Dict[str, np.ndarray],
        device: DeviceInfo,
        scaler_path: Path,
        is_regression: bool = False,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        return_predictions: bool = False,
    ):
        feat_dict, _ = self.adjust_features(table, self.solution_config.remove_add_feature, 
                                            self.solution_config.remove_cat_temporal_information, 
                                            self.entity_id_column)
        
        # Fix 2D arrays to 1D for pandas DataFrame
        fixed_feat_dict = {}
        for key, value in feat_dict.items():
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 1:
                fixed_feat_dict[key] = value.squeeze()
            else:
                fixed_feat_dict[key] = value
        
        test_df = pd.DataFrame(fixed_feat_dict, copy=False)

        task_type = self.data_config.task.task_type

        X_test, y_test = self.extract_features_and_label(test_df, task_type, scaler_path)

        is_multi_class = task_type == DBBTaskType.classification and y_test.max() != 1.

        if task_type == DBBTaskType.regression:
            self.data_config.task.num_classes = 1

        metric_output = self.calculate_metric(
            X_test,
            y_test,
            test_df,
            scaler_path,
            is_regression,
            is_multi_class=is_multi_class,
            num_classes=self.data_config.task.num_classes,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            return_predictions=return_predictions,
        )
        if return_predictions:
            metrics, predictions, ground_truth = metric_output
            return metrics, predictions, ground_truth
        return metric_output

    def checkpoint(self, ckpt_path: Path) -> None:
        # Minimally implemented - just saving configuration
        ckpt_path = Path(ckpt_path)
        yaml_utils.save_pyd(self.solution_config, ckpt_path / 'solution_config.yaml')
        yaml_utils.save_pyd(self.data_config, ckpt_path / 'data_config.yaml')

        # Note: For production, you'd want to save the model using joblib or pickle
        # but we're skipping that per requirements

    def load_from_checkpoint(self, ckpt_path: Path) -> None:
        # Minimally implemented - just loading configuration
        ckpt_path = Path(ckpt_path)
        self.solution_config = yaml_utils.load_pyd(
            self.config_class, ckpt_path / 'solution_config.yaml')
        self.data_config = yaml_utils.load_pyd(
            TabularDatasetConfig, ckpt_path / 'data_config.yaml')

        # Note: For production, you'd want to load the model using joblib or pickle
        # but we're skipping that per requirements

    def adjust_features(
        self,
        feat_dict: Dict[str, np.ndarray],
        remove_add_feature: bool = True, 
        remove_cat_temporal_information: bool = True,
        entity_id_column: str = ""
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Adjust features suitable for sklearn models."""
        logger.info("Adapting features ...")
        new_feat_dict = {}
        feat_meta = {}  # Simple metadata placeholder

        for name, feat in feat_dict.items():
            if name == entity_id_column:
                continue
            if name in [self.data_config.task.target_column]:
                new_feat_dict[name] = feat
            elif remove_add_feature and 'add_feature' in name:
                continue
            elif remove_cat_temporal_information and ('YEAR' in name or 'MONTH' in name or 'DAY' in name or 'DAYOFWEEK' in name):
                continue
            else:
                dtype = self.data_config.features[name].dtype
                if dtype in [DBBColumnDType.primary_key]:
                    continue
                elif dtype in [DBBColumnDType.category_t, DBBColumnDType.foreign_key]:
                    new_feat_dict[name] = feat
                elif dtype == DBBColumnDType.timestamp_t:
                    continue
                elif dtype == DBBColumnDType.float_t:
                    in_size = self.data_config.features[name].extra_fields.get('in_size', 1)
                    if in_size == 1:
                        new_feat_dict[name] = feat
                    else:
                        continue  # Skip multi-dimensional features for now
                else:
                    logger.info(f"Ignore feature '{name}' of type {dtype}")

        return new_feat_dict, feat_meta

    def extract_features_and_label(self, df: pd.DataFrame, task_type: DBBTaskType, scaler_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract features and labels from the dataframe."""
        label_name = self.get_label_name()

        # Get feature names excluding target/label columns
        feature_cols = []
        for col in df.columns:
            if col in [
                self.data_config.task.target_column,
                f'{self.data_config.task.target_table}.{self.data_config.task.target_column}'
            ]:
                continue


            # Skip primary keys
            if col in self.data_config.features and self.data_config.features[col].dtype == DBBColumnDType.primary_key:
                continue

            # Skip foreign keys if configured
            if (not self.solution_config.use_foreign_key_feature and
                col in self.data_config.features and
                self.data_config.features[col].dtype == DBBColumnDType.foreign_key):
                continue

            feature_cols.append(col)

        X = df[feature_cols].copy()
        label_series = df[label_name]

        valid_mask = label_series.notna()
        if not valid_mask.all():
            X = X.loc[valid_mask].reset_index(drop=True)
            label_series = label_series[valid_mask]
        label_series = label_series.reset_index(drop=True)

        if self.data_config.task.task_type == DBBTaskType.classification:
            unique_labels = pd.Index(label_series.unique())
            if self.label_categories is None:
                self.label_categories = np.array(sorted(unique_labels))
            else:
                combined = pd.Index(self.label_categories).union(unique_labels)
                if len(combined) != len(self.label_categories):
                    self.label_categories = np.array(sorted(combined))
            label_map = {val: idx for idx, val in enumerate(self.label_categories)}
            encoded = label_series.map(label_map)
            if encoded.isnull().any():
                missing = label_series[pd.isna(encoded)].unique()
                raise ValueError(f"Encountered unknown labels {missing} after encoding.")
            y = encoded.to_numpy(dtype=np.int64, copy=False)
        else:
            y = label_series.to_numpy()

        # if task_type == DBBTaskType.classification and y.max() != 1.:
        #     scaler_path = Path(scaler_path) / 'target_scalers.pkl'
        #     scaler = list(pickle.load(open(scaler_path, 'rb')).values())[0]
        #     y = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1).astype(int)

        return X, y

    def get_label_name(self) -> str:
        """Get the appropriate label column name based on task type."""
        return self.data_config.task.target_column

    def calculate_metric(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        df: pd.DataFrame,
        scaler_path: Path,
        is_regression: bool = False,
        is_multi_class: bool = False,
        num_classes: int = 2,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        return_predictions: bool = False,
    ):
        """Calculate the appropriate metric based on task type using batch processing."""
        # metric_fn = get_metric_fn(self.data_config.task)
        task_type = self.data_config.task.task_type
        if task_type == 'classification' and num_classes > 2:
            task_type = 'multiclass_classification'
        elif task_type == 'classification' and num_classes == 2:
            task_type = 'binary_classification'
        metric_fns = get_metrics(task_type, num_classes)
        batch_size = self.solution_config.eval_batch_size

        # Process data in batches to avoid memory issues
        num_samples = len(X)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

        all_preds = []
        if is_regression:
            scaler_path = Path(scaler_path) / 'target_scalers.pkl'
            scaler = list(pickle.load(open(scaler_path, 'rb')).values())[0]
        else:
            scaler = None

        # Process each batch
        for i in tqdm(range(num_batches), desc="Computing predictions in batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            X_batch = X[start_idx:end_idx]

            #self.select_and_fit(target_set=X_batch)

            # Get predictions based on task type
            if self.data_config.task.task_type == DBBTaskType.classification:
                # For classification tasks
                batch_pred = self.predictor.predict_proba(X_batch)
                all_preds.append(batch_pred)
            else:
                # For regression tasks
                batch_pred = self.predictor.predict(X_batch).reshape(-1, 1)
                batch_pred = scaler.inverse_transform(batch_pred).reshape(-1)
                all_preds.append(batch_pred)

        # Combine all batch predictions
        if self.data_config.task.task_type == DBBTaskType.classification:
            # For multi-class classification, need to handle the 2D array
            pred = np.vstack(all_preds)
        else:
            # For binary classification or regression
            pred = np.concatenate(all_preds)

        # Apply inverse transform for regression
        if self.data_config.task.task_type == DBBTaskType.regression:
            y = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)

        # Apply clamping if specified
        if clamp_min is not None and clamp_max is not None:
            pred = np.clip(pred, clamp_min, clamp_max)

        final_pred = np.array(pred, copy=True)

        pred, label = torch.tensor(pred), torch.tensor(y)

        index = None

        metric = {
            metric_name: metric_fn(index, pred, label).item()
            for metric_name, metric_fn in metric_fns.items()
        }
        if return_predictions:
            return metric, final_pred, np.array(y, copy=True)
        return metric
        # return metric_fn(index, pred, label).item()

    def select_training_samples(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        target_set: pd.DataFrame,
        max_samples: int,
        strategy: Literal['graph', 'random', 'stratified'] = 'graph'
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Select a subset of training samples based on the task type and max_samples.

        Args:
            X: Feature DataFrame
            y: Target labels/values
            max_samples: Maximum number of samples to retain

        Returns:
            Tuple of (selected_X, selected_y, index)
        """
        return self.downsample_training_set(
                X, y, max_samples, stratified_sampling=True)



    def downsample_training_set(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_samples: int,
        stratified_sampling=False
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Downsample data to max_samples while preserving class balance for classification tasks.

        Args:
            X: Feature DataFrame
            y: Target labels/values
            max_samples: Maximum number of samples to retain
            stratified_sampling: If True, maintain class balance. If False, sample randomly
                                 but ensure at least one sample per class for classification tasks.

        Returns:
            Tuple of (downsampled_X, downsampled_y, idx)
        """
        if len(X) <= max_samples:
            return X, y, None

        logger.info(f"Downsampling training set from {len(X)} to {max_samples} samples.")
        idx = np.random.choice(len(X), max_samples, replace=False)
        return X.iloc[idx], y[idx], idx
