"""Fetch and combine experiment runs from WandB and MongoDB."""
import ast
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import wandb
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from utils.information import (
    NEEDED_TASKS,
    rdl_cared_columns,
    relbench_ft_transformer_cared_columns,
    relbench_lgbm_cared_columns,
    relbench_tabpfn_cared_columns,
    task_name_to_metric_no_prefix,
    task_name_to_metric_prefix_rdl,
    task_name_to_task_type,
)
from utils.misc import lift_and_remove_nested_keys
from utils.type import seed_everything

try:
    from utils.database import SwapDB
except ImportError:
    SwapDB = None

load_dotenv()

DB_LOCATION = os.environ.get("MONGODB_URI", "")

# Cache file name (fixed typo from legacy "valaidated")
_COMBINED_CACHE = "validated_search_results.csv"
_LEGACY_CACHE = "valaidated_search_results.csv"


# ---------------------------------------------------------------------------
# Preprocessing helpers (top-level, parameterized)
# ---------------------------------------------------------------------------

def _parse_dbs_field(dbs) -> tuple:
    """Extract (dataset_name, task_name) from a dbs config field."""
    if isinstance(dbs, str):
        dbs = ast.literal_eval(dbs)
    return dbs[0]['db_name'], dbs[0]['task_name'][0]


def _preprocess_run_dataframe(
    df: pd.DataFrame,
    *,
    model_type: str,
    metric_lookup: dict,
    cared_columns: list,
    use_full_name_metric: bool = False,
    model_label: Optional[str] = None,
    column_renames: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Unified preprocessing for WandB run CSVs.

    Parameters
    ----------
    model_type : 'rdl' or 'dfs'
    metric_lookup : dict mapping task_name -> metric name
    cared_columns : columns to copy from raw df
    use_full_name_metric : if True, metric col = val_{metric}_{dataset}_{task} (RDL style)
    model_label : optional model name column (e.g. 'ft_transformer', 'lgbm', 'tabpfn')
    column_renames : optional column renames to apply before copying (e.g. hidden_size -> hidden_channels)
    """
    if column_renames:
        for old_name, new_name in column_renames.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]

    new_df = pd.DataFrame()
    new_df['dataset_name'], new_df['task_name'] = zip(
        *df['dbs'].apply(_parse_dbs_field)
    )
    new_df['model_type'] = model_type
    new_df['task_type'] = new_df['task_name'].map(task_name_to_task_type)
    new_df['main_metric_name'] = new_df['task_name'].map(metric_lookup)
    df['main_metric_name'] = new_df['main_metric_name']

    if use_full_name_metric:
        df['full_name'] = new_df['dataset_name'] + '_' + new_df['task_name']
        new_df['val_metric'] = df.apply(
            lambda x: x[f"val_{x['main_metric_name']}_{x['full_name']}"], axis=1
        )
        new_df['test_metric'] = df.apply(
            lambda x: x[f"test_{x['main_metric_name']}_{x['full_name']}"], axis=1
        )
    else:
        if 'val_auroc' in df.columns:
            df['val_roc_auc'] = df['val_auroc']
            df['test_roc_auc'] = df['test_auroc']
        new_df['val_metric'] = df.apply(
            lambda x: x[f"val_{x['main_metric_name']}"], axis=1
        )
        new_df['test_metric'] = df.apply(
            lambda x: x[f"test_{x['main_metric_name']}"], axis=1
        )

    if model_label is not None:
        new_df['model'] = model_label

    for column in cared_columns:
        if column in df.columns:
            new_df[column] = df[column].values
        else:
            new_df[column] = pd.NA

    return new_df[new_df['val_metric'].notna() & new_df['test_metric'].notna()]


# Dispatch config: base_name -> preprocessing kwargs + optional post_filter
_DFS_RENAMES = {'hidden_size': 'hidden_channels', 'normalization': 'norm'}

_PREPROCESS_CONFIG = {
    'all_rdl': dict(
        model_type='rdl',
        metric_lookup=task_name_to_metric_prefix_rdl,
        cared_columns=rdl_cared_columns,
        use_full_name_metric=True,
    ),
    'rdl-constructor': dict(
        model_type='rdl',
        metric_lookup=task_name_to_metric_prefix_rdl,
        cared_columns=rdl_cared_columns,
        use_full_name_metric=True,
    ),
    'dfs-constructor': dict(
        model_type='dfs',
        metric_lookup=task_name_to_metric_no_prefix,
        cared_columns=relbench_ft_transformer_cared_columns,
        column_renames=_DFS_RENAMES,
    ),
    'old_dfs': dict(
        model_type='dfs',
        metric_lookup=task_name_to_metric_no_prefix,
        cared_columns=relbench_ft_transformer_cared_columns,
        column_renames=_DFS_RENAMES,
    ),
    'relbench_ft_transformer': dict(
        model_type='dfs',
        metric_lookup=task_name_to_metric_no_prefix,
        cared_columns=relbench_ft_transformer_cared_columns,
        model_label='ft_transformer',
        column_renames=_DFS_RENAMES,
    ),
    'relbench_lgbm': dict(
        model_type='dfs',
        metric_lookup=task_name_to_metric_no_prefix,
        cared_columns=relbench_lgbm_cared_columns,
        model_label='lgbm',
    ),
    'relbench_tabpfn': dict(
        model_type='dfs',
        metric_lookup=task_name_to_metric_no_prefix,
        cared_columns=relbench_tabpfn_cared_columns,
        model_label='tabpfn',
    ),
}

# Post-filters applied after preprocessing
_CONSTRUCTOR_TASKS = ['constructor-scores-points', 'constructor-scores-tasks']

_POST_FILTERS = {
    'rdl-constructor': lambda df: df[df['task_name'].isin(_CONSTRUCTOR_TASKS)],
    'dfs-constructor': lambda df: df[df['task_name'].isin(_CONSTRUCTOR_TASKS)],
    'old_dfs': lambda df: df[df['dataset_name'].isin(['rel-avs', 'rel-ieeecis'])],
}

# Files that already have standardized columns from MongoDB (no WandB preprocessing needed)
_ALREADY_STANDARDIZED = {'new_rdl', 'dfs'}


# ---------------------------------------------------------------------------
# MongoDB metric extraction helper
# ---------------------------------------------------------------------------

def _extract_metrics(container: dict, key: str) -> dict:
    """Extract metrics dict, handling singular/plural key names and scalar values."""
    metrics = container.get(key, None)
    if not metrics:
        alt_key = key[:-1] if key.endswith('s') else f"{key}s"
        metrics = container.get(alt_key, None)
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return metrics
    return {"value": metrics}


# ---------------------------------------------------------------------------
# WandBAnalyzer
# ---------------------------------------------------------------------------

class WandBAnalyzer:
    """Fetch and combine WandB/MongoDB experiment results."""

    def __init__(self, entity: str):
        self.api = wandb.Api()
        self.entity = entity
        self.runs = None
        self.expected_data_path = "results"
        self.expected_files = [
            'all_rdl', 'dfs', 'new_rdl', 'old_dfs',
            'rdl-constructor', 'dfs-constructor',
        ]
        self.file_to_run_mapping = {
            'all_rdl': ('rdl-search', 'wandb'),
            'dfs': ('dfs-search-2-refined', 'mongodb'),
            'new_rdl': ('cr-search-refined', 'mongodb'),
            'old_dfs': ('dfs-search', 'wandb'),
            'rdl-constructor': ('rdl-search-refined', 'wandb'),
            'dfs-constructor': ('dfs-search-refined', 'wandb'),
        }
        seed_everything(42)
        self.column_formats = [
            'model_type', 'dataset_name', 'task_name', 'task_type',
            'main_metric_name', 'val_metric', 'test_metric', 'lr', 'gamma',
            'weight_decay', 'num_layers', 'dropout', 'hidden_channels',
            'batch_size', 'scheduler', 'dfs_layer', 'max_steps_per_epoch',
            'num_heads', 'attn_dropout', 'norm', 'loss_fn', 'torch_frame_model_cls',
            'num_neighbors', 'pre_sf', 'post_sf', 'aggregation', 'mpnn_type',
            'channels', 'temporal', 'skip_connection', 'loader_type',
        ]
        self.check_and_download_datasets()

    # -- Download / fetch --------------------------------------------------

    def check_and_download_datasets(self, replace: bool = False):
        """Download missing dataset CSVs from WandB or MongoDB."""
        os.makedirs(self.expected_data_path, exist_ok=True)
        existing = set(os.listdir(self.expected_data_path))

        for file in self.expected_files:
            target = f"{file}.csv"
            if target in existing and not replace:
                logger.debug(f"Skip fetching '{file}': file already exists")
                continue
            if file not in self.file_to_run_mapping:
                logger.warning(f"No mapping found for '{file}', skipping")
                continue
            _, source_type = self.file_to_run_mapping[file]
            try:
                if source_type == 'wandb':
                    self.fetch_wandb_runs(file)
                elif source_type == 'mongodb':
                    rows = self.fetch_mongodb_runs(file)
                    if rows:
                        self.save_mongodb_runs_to_raw_dataframe(rows, file)
                else:
                    logger.warning(f"Unknown source '{source_type}' for '{file}', skipping")
            except Exception as e:
                logger.error(f"Failed to fetch '{file}' from {source_type}: {e}")

    @lru_cache(maxsize=1000)
    def fetch_wandb_runs(self, file, filters: Optional[Dict[str, Any]] = None,
                         state: str = "finished") -> List[wandb.apis.public.Run]:
        project_name, _ = self.file_to_run_mapping[file]
        try:
            runs = self.api.runs(f"{self.entity}/{project_name}")
            logger.info(f"Fetched {len(runs)} runs from {project_name}")
        except Exception as e:
            logger.error(f"Error fetching runs for {project_name}: {e}")
            return
        self.save_runs_to_raw_dataframe(runs, file)

    def fetch_mongodb_runs(self, file) -> Optional[list]:
        if file not in self.file_to_run_mapping:
            logger.warning(f"No mapping for '{file}' when fetching from mongodb")
            return None
        collection_name, _ = self.file_to_run_mapping[file]
        try:
            db = SwapDB(
                path_to_db=DB_LOCATION,
                collection_name=collection_name,
                db_name="AutoGML_test",
            )
            rows = db.get_all_rows()
        except Exception as e:
            logger.error(f"Error fetching from MongoDB '{collection_name}': {e}")
            return None
        if not rows:
            logger.debug(f"No rows returned from MongoDB for '{collection_name}'")
            return None
        return rows

    # -- Save raw runs to CSV ----------------------------------------------

    def save_runs_to_raw_dataframe(self, runs: List[wandb.apis.public.Run], filename: str):
        data = []
        for run in tqdm(runs, desc=f"Saving runs to {filename}"):
            row = {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "runtime": run.summary.get("_runtime", 0),
            }
            flattened_config = lift_and_remove_nested_keys(run.config)
            row.update(flattened_config)
            for key, value in run.summary.items():
                if not key.startswith("_"):
                    row[key] = value
            data.append(row)
        pd.DataFrame(data).to_csv(
            f"{self.expected_data_path}/{filename}.csv", index=False
        )

    def save_mongodb_runs_to_raw_dataframe(
        self, rows: List[Dict[str, Any]], filename: str
    ) -> pd.DataFrame:
        """Convert MongoDB documents to a standardized DataFrame and save as CSV."""
        data = []
        for row in tqdm(rows, desc=f"Processing MongoDB runs for {filename}"):
            try:
                config = row.get('config', {})
                results = row.get('results', {})
                model_params = config.get('model_params', {})

                dbs = config.get('dbs', [])
                if isinstance(dbs, str):
                    dbs = ast.literal_eval(dbs)
                if not dbs:
                    continue

                dataset_name = dbs[0].get('db_name', '')
                task_names = dbs[0].get('task_name', [])
                task_name = (
                    task_names[0] if isinstance(task_names, list) and task_names
                    else (task_names if isinstance(task_names, str) else '')
                )
                if not task_name:
                    continue

                task_type = task_name_to_task_type.get(task_name, 'unknown')
                main_metric_name = task_name_to_metric_no_prefix.get(task_name, 'unknown')

                val_metrics = _extract_metrics(results, 'val_metrics')
                test_metrics = _extract_metrics(results, 'test_metrics')

                if main_metric_name == 'roc_auc':
                    val_metric = val_metrics.get('roc_auc', val_metrics.get('auroc'))
                    test_metric = test_metrics.get('roc_auc', test_metrics.get('auroc'))
                else:
                    val_metric = val_metrics.get(main_metric_name, val_metrics.get('value'))
                    test_metric = test_metrics.get(main_metric_name, test_metrics.get('value'))

                model_type_raw = model_params.get('model_type', '')
                if model_type_raw in ('rdl', 'RDL'):
                    model_type = 'rdl'
                elif model_type_raw in ('fs', 'dfs', 'FS', 'DFS'):
                    model_type = 'dfs'
                else:
                    model_type = 'dfs'

                optimizer_config = model_params.get('optimizer_config', {})
                gnn_config = model_params.get('gnn_config', {})
                sampler_config = model_params.get('sampler_config', {})
                tab_nn_config = model_params.get('tab_nn_config', {})

                data.append({
                    'model_type': model_type,
                    'dataset_name': dataset_name,
                    'task_name': task_name,
                    'task_type': task_type,
                    'main_metric_name': main_metric_name,
                    'val_metric': val_metric,
                    'test_metric': test_metric,
                    'lr': optimizer_config.get('lr'),
                    'gamma': optimizer_config.get('gamma'),
                    'weight_decay': optimizer_config.get('weight_decay'),
                    'scheduler': optimizer_config.get('scheduler'),
                    'batch_size': model_params.get('batch_size'),
                    'max_steps_per_epoch': config.get('max_steps_per_epoch'),
                    'dfs_layer': model_params.get('dfs_layer'),
                    'torch_frame_model_cls': model_params.get('torch_frame_model_cls'),
                    'num_layers': tab_nn_config.get('num_layers', gnn_config.get('num_layers', sampler_config.get('num_layers'))),
                    'dropout': tab_nn_config.get('dropout', gnn_config.get('dropout')),
                    'hidden_channels': tab_nn_config.get('hidden_channels', gnn_config.get('hidden_channels')),
                    'num_heads': tab_nn_config.get('num_heads', gnn_config.get('num_heads')),
                    'attn_dropout': tab_nn_config.get('attn_dropout'),
                    'norm': tab_nn_config.get('normalization', gnn_config.get('norm')),
                    'loss_fn': gnn_config.get('loss_fn'),
                    'num_neighbors': sampler_config.get('num_neighbors'),
                    'pre_sf': gnn_config.get('pre_sf'),
                    'post_sf': gnn_config.get('post_sf'),
                    'aggregation': gnn_config.get('aggregation'),
                    'mpnn_type': gnn_config.get('mpnn_type'),
                    'channels': gnn_config.get('channels', model_params.get('channels')),
                    'temporal': sampler_config.get('temporal'),
                    'skip_connection': gnn_config.get('skip_connection'),
                    'loader_type': sampler_config.get('loader_type'),
                })
            except Exception as e:
                logger.warning(f"Error processing MongoDB row: {e}")

        if not data:
            logger.warning(f"No valid data found for {filename}")
            return pd.DataFrame(columns=self.column_formats)

        df = pd.DataFrame(data)
        for col in self.column_formats:
            if col not in df.columns:
                df[col] = None
        df = df[self.column_formats]

        output_path = f"{self.expected_data_path}/{filename}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} MongoDB runs to {output_path}")
        return df

    # -- Combine & preprocess ----------------------------------------------

    def _find_available_csv_files(self) -> dict:
        """Return {base_name: filename} for expected CSVs that exist on disk."""
        available = {}
        for filename in os.listdir(self.expected_data_path):
            if not filename.endswith(".csv"):
                continue
            base = Path(filename).stem
            if base in self.expected_files:
                available[base] = filename

        missing = [b for b in self.expected_files if b not in available]
        if missing:
            logger.warning(f"Missing CSV files for {missing}. These sources will be skipped.")
        return available

    def _preprocess_file(self, base_name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preprocess a single CSV file based on its base_name, or return None to skip."""
        if base_name in _ALREADY_STANDARDIZED:
            # MongoDB-sourced files already have standardized columns
            return df

        if base_name not in _PREPROCESS_CONFIG:
            logger.warning(f"Unknown results file prefix '{base_name}', skipping")
            return None

        processed = _preprocess_run_dataframe(df, **_PREPROCESS_CONFIG[base_name])

        if base_name in _POST_FILTERS:
            processed = _POST_FILTERS[base_name](processed)
            logger.info(f"Post-filtered '{base_name}': {len(processed)} rows remain")

        return processed

    def combine_and_generate_a_large_dataframe(self) -> pd.DataFrame:
        """Combine all downloaded runs into a single standardized DataFrame."""
        # Check for cached combined results (handle legacy typo filename)
        if os.path.exists(_COMBINED_CACHE):
            logger.info(f"Loading existing combined results from {_COMBINED_CACHE}")
            return pd.read_csv(_COMBINED_CACHE)
        if os.path.exists(_LEGACY_CACHE):
            logger.warning(f"Found legacy cache '{_LEGACY_CACHE}', renaming to '{_COMBINED_CACHE}'")
            os.rename(_LEGACY_CACHE, _COMBINED_CACHE)
            return pd.read_csv(_COMBINED_CACHE)

        logger.info("Combined results file not found, generating from individual CSV files...")
        available_files = self._find_available_csv_files()
        if not available_files:
            raise FileNotFoundError(
                f"No cached run CSVs found under '{self.expected_data_path}'. "
                "Run check_and_download_datasets() first."
            )

        frames = []
        for base_name, filename in available_files.items():
            df = pd.read_csv(os.path.join(self.expected_data_path, filename))
            processed = self._preprocess_file(base_name, df)
            if processed is not None:
                frames.append(processed)

        if not frames:
            raise ValueError(
                "No result CSVs could be processed. "
                "Ensure expected files exist or re-run data collection."
            )

        combined = pd.concat(frames, axis=0, join='outer', ignore_index=True)
        combined.to_csv(_COMBINED_CACHE, index=False)
        return combined


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_validated_runs(entity: str, refresh: bool = False) -> pd.DataFrame:
    """Load, combine, and filter validated experiment runs."""
    analyzer = WandBAnalyzer(entity=entity)
    if refresh:
        analyzer.check_and_download_datasets(replace=True)

    df = analyzer.combine_and_generate_a_large_dataframe()
    df = df[df["task_name"].isin(NEEDED_TASKS)].copy()

    is_rdl = df['model_type'].astype(str).str.lower() == 'rdl'
    is_dfs = df['model_type'].astype(str).str.lower() == 'dfs'
    dfs_valid = (
        is_dfs
        & df['torch_frame_model_cls'].notna()
        & (df['torch_frame_model_cls'] != 'lightgbm')
        & (df['dfs_layer'] >= 2)
    )
    return df[is_rdl | dfs_valid].copy()


# ---------------------------------------------------------------------------
# Backward-compatible re-exports from utils.analysis
# (prefer importing directly from utils.analysis in new code)
# ---------------------------------------------------------------------------
from utils.analysis import (  # noqa: E402, F401
    BetterIs,
    TaskType,
    add_taskwise_rank,
    get_best_k_trials,
    result_stats,
    summarize_best_trials,
    summarize_successful_runs,
)
