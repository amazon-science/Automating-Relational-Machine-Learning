"""
Unified task embedding computation using multiple methods.

This module computes various task-level features including:
- Homophily metrics (edge, insensitive, adjacency, aggregation)
- Time homophily metrics
- Duplication rates
- Heuristic-based features
- Metapath-induced degree statistics
- Auto-transfer embeddings
- Hit rate / Random feature regressor performance

Features can be selectively computed via command-line arguments.
"""
import argparse
import gc
import os
import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from loguru import logger
from relbench.base import Table
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from tqdm import tqdm
from models.regressor import RandomFeatureRegressor

# Local imports
from data.mtaskdataset import (
    prepare_transductive_rdb_datasets_save_memory,
    MultiTabularDataset,
)
from models.automl.autotransfer import (
    DFS_ANCHOR_CONFIG,
    SIX_ANCHOR_CONFIG,
    generate_model_task_embedding,
)
from swap.execution import (
    TaskType,
    configure_rdl,
    generate_representation,
    get_auto_model,
)
from models.dfs import generate_dfs_data_pre_post
from utils.heuristics import gini_degree, get_simple_stats, heuristics, time_homophily, compute_lag_autocorr
from utils.heuristics_gpu import homophily_from_table_db, rff_embed_scalar_1d
from utils.misc import load_yaml, timer as time_it
from utils.type import seed_everything
from utils.information import NEEDED_TASKS
from utils.retrieve_search_data import load_validated_runs
from swap.same_class_analysis import (
    SameClassAnalyzer,
    REGRESSION_TASKS as SAME_CLASS_REGRESSION_TASKS,
    _same_class_cache_path,
    _same_class_cache_complete,
    _feature_payload_from_result as _same_class_payload,
)

# Machine-specific settings
os.environ.setdefault('XDG_CACHE_HOME', 'cache_data')
warnings.filterwarnings("ignore")

DEFAULT_DATASET_CONFIG = "configs/pyg/hpo/all-nodes.yaml"

# Features that require the full database loaded into memory
_HEAVY_FEATURES = {'homophily', 'hit', 'autotransfer', 'autotransfer_dfs'}
# Features that only need task tables (lightweight)
_LIGHTWEIGHT_FEATURES = {'temporal', 'stats', 'same_class'}


# ============================================================================
# Memory Management
# ============================================================================

def clear_task_memory() -> None:
    """
    Clear memory after processing each task to prevent memory accumulation.
    """
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()
    gc.collect()

    # Log memory usage if available
    if torch.cuda.is_available() and hasattr(torch.cuda, 'memory_allocated'):
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logger.debug(
            f"GPU Memory after cleanup - Allocated: {memory_allocated:.2f}GB, "
            f"Reserved: {memory_reserved:.2f}GB"
        )


# ============================================================================
# Helpers
# ============================================================================

def _summarize_metapath_degrees(metapath_degrees: List[Tuple[Any, torch.Tensor, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
    """
    Convert raw metapath degree tensors into summary statistics per metapath.
    """
    stats: Dict[str, Dict[str, float]] = {}
    if not metapath_degrees:
        return stats
    for i, (metapath, source_degrees, target_degrees) in enumerate(metapath_degrees):
        metapath_str = f"{metapath[0]}-{metapath[1]}"
        source_avg_deg = float(source_degrees.mean().item())
        target_avg_deg = float(target_degrees.mean().item())
        source_max_deg = float(source_degrees.max().item())
        target_max_deg = float(target_degrees.max().item())
        min_source_deg = float(source_degrees.min().item())
        min_target_deg = float(target_degrees.min().item())
        avg_deg = (source_avg_deg + target_avg_deg) / 2.0
        max_deg = (source_max_deg + target_max_deg) / 2.0
        min_deg = (min_source_deg + min_target_deg) / 2.0
        gini_deg = (
            gini_degree(source_degrees.cpu().numpy())
            + gini_degree(target_degrees.cpu().numpy())
        ) / 2.0

        logger.info(
            f"  Metapath {i}: {metapath_str}: "
            f"Avg degree={avg_deg:.2f}, Max degree={max_deg:.0f}, "
            f"Gini degree={gini_deg:.3f}"
        )
        stats[metapath_str] = {
            'avg_deg': avg_deg,
            'max_deg': max_deg,
            'min_deg': min_deg,
            'gini_deg': gini_deg,
        }
    return stats


# ============================================================================
# Temporal Axis Features
# ============================================================================

def _compute_temporal_offsets(entity_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given sorted entity codes, return:
      - start_positions: first index of each entity segment
      - entity_ord: per-row entity ordinal (0..num_entities-1)
      - prev_count: number of historical rows for the entity before each row
    """
    n = entity_codes.numel()
    device = entity_codes.device
    is_new = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        is_new[1:] = entity_codes[1:] != entity_codes[:-1]
    entity_ord = torch.cumsum(is_new.to(torch.long), dim=0) - 1
    start_positions = torch.nonzero(is_new, as_tuple=False).squeeze(1)
    row_idx = torch.arange(n, device=device)
    prev_count = row_idx - start_positions[entity_ord]
    return start_positions, entity_ord, prev_count


def _prefix_by_entity(prefix: torch.Tensor, start_positions: torch.Tensor, entity_ord: torch.Tensor) -> torch.Tensor:
    """
    Convert a global prefix sum tensor into per-entity prefix values (exclude current row).
    Works with 1D or 2D prefix tensors.
    """
    prefix_before = torch.zeros_like(prefix)
    prefix_before[1:] = prefix[:-1]
    start_prefix = prefix_before[start_positions]
    return prefix_before - start_prefix[entity_ord]


@torch.no_grad()
def compute_temporal_axis_gpu(
    task_table: pd.DataFrame,
    entity_col: str,
    time_col: str,
    label_col: str,
    *,
    task_type: str,
    is_autocomplete: bool = False,
    lags: Tuple[int, ...] = (1, 2, 3),
) -> Dict[str, float]:
    """
    Compute lag-specific temporal autocorrelation using GPU acceleration.

    For each lag k in {1, 2, 3}, computes the similarity between labels at time t
    and labels at time t-k for the same entity. Also computes the old temporal
    homophily metrics for backward compatibility.

    Returns a dict with:
      - temporal_homophily_gaussian (legacy, all-history average)
      - temporal_homophily_corr (legacy, all-history correlation)
      - lag{k}_autocorr_gaussian for each k in lags (lag-specific Gaussian similarity)
      - lag{k}_autocorr_corr for each k in lags (lag-specific correlation)
    """
    zero_result = {
        'temporal_homophily_gaussian': 0.0,
        'temporal_homophily_corr': 0.0,
    }
    for k in lags:
        zero_result[f'lag{k}_autocorr_gaussian'] = 0.0
        zero_result[f'lag{k}_autocorr_corr'] = 0.0

    if task_table.empty or label_col not in task_table:
        return zero_result

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = task_table[[entity_col, time_col, label_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values([entity_col, time_col], kind='mergesort').reset_index(drop=True)

    has_duplicate_entity = df[entity_col].duplicated().any()
    if is_autocomplete and not has_duplicate_entity:
        return zero_result

    entity_codes = pd.factorize(df[entity_col], sort=False)[0].astype(np.int64)
    entity_t = torch.as_tensor(entity_codes, device=device)
    start_positions, entity_ord, prev_count = _compute_temporal_offsets(entity_t)
    if prev_count.max().item() == 0:
        return zero_result

    task_type_str = task_type.lower()
    is_regression = 'regression' in task_type_str

    result = {}

    if is_regression:
        label_np = pd.to_numeric(df[label_col], errors='coerce').to_numpy(dtype=np.float64)
        label_t = torch.as_tensor(label_np, device=device, dtype=torch.float64)
        valid_label = torch.isfinite(label_t)
        if not valid_label.any():
            return zero_result

        # Legacy: Gaussian-style similarity via RFF aggregation (all history)
        sigma = float(label_t[valid_label].std(unbiased=True).item() or 1.0)
        rff = rff_embed_scalar_1d(label_t, dim_half=32, sigma=sigma)  # [N, 64]
        rff_prefix = torch.cumsum(rff, dim=0)
        rff_hist = _prefix_by_entity(rff_prefix, start_positions, entity_ord)
        valid_mask = valid_label & (prev_count > 0)
        temporal_h = 0.0
        if valid_mask.any():
            denom = prev_count[valid_mask].unsqueeze(1).to(rff.dtype).clamp_min(1.0)
            mean_hist = rff_hist[valid_mask] / denom
            sim = (rff[valid_mask] * mean_hist).sum(dim=1)
            temporal_h = float(sim.mean().item())
        result['temporal_homophily_gaussian'] = temporal_h

        # Legacy: Correlation-style similarity using aggregated pair statistics (all history)
        y_norm = label_t.clone()
        mu = y_norm[valid_label].mean()
        std = y_norm[valid_label].std(unbiased=True)
        if (not torch.isfinite(std)) or std.item() == 0.0:
            std = torch.tensor(1.0, device=device, dtype=y_norm.dtype)
        y_norm = torch.where(valid_label, (y_norm - mu) / std, torch.zeros_like(y_norm))

        y_prefix = torch.cumsum(y_norm, dim=0)
        y2_prefix = torch.cumsum(y_norm * y_norm, dim=0)
        y_hist = _prefix_by_entity(y_prefix, start_positions, entity_ord)
        y2_hist = _prefix_by_entity(y2_prefix, start_positions, entity_ord)

        num_prev = prev_count.to(torch.float64)
        M_edges = num_prev.sum()
        temporal_corr = 0.0
        if M_edges.item() > 0:
            sum_yu = (y_norm * num_prev).sum()
            sum_yu2 = (y_norm * y_norm * num_prev).sum()
            sum_yv = y_hist.sum()
            sum_yv2 = y2_hist.sum()
            sum_yuyv = (y_norm * y_hist).sum()

            mu_u = sum_yu / M_edges
            mu_v = sum_yv / M_edges
            var_u = (sum_yu2 / M_edges) - mu_u * mu_u
            var_v = (sum_yv2 / M_edges) - mu_v * mu_v
            cov = (sum_yuyv / M_edges) - mu_u * mu_v
            denom = (var_u * var_v).clamp_min(1e-12).sqrt()
            temporal_corr = float((cov / denom).item())
        result['temporal_homophily_corr'] = temporal_corr

        # NEW: Lag-specific autocorrelation
        n = len(label_t)
        for k in lags:
            # Shift labels by k within each entity
            lagged_labels = torch.full_like(label_t, float('nan'))
            for i in range(n):
                ent = entity_ord[i].item()
                if prev_count[i].item() >= k:
                    # Find the k-th previous row for this entity
                    # prev_count[i] tells us there are that many rows before i with same entity
                    # We need to go back k rows within the same entity
                    start_pos = start_positions[ent].item()
                    idx_within_entity = i - start_pos
                    if idx_within_entity >= k:
                        lagged_idx = i - k
                        lagged_labels[i] = label_t[lagged_idx]

            # Mask valid pairs
            mask = torch.isfinite(lagged_labels) & valid_label
            if not mask.any():
                result[f'lag{k}_autocorr_gaussian'] = 0.0
                result[f'lag{k}_autocorr_corr'] = 0.0
                continue

            cur = label_t[mask]
            prev = lagged_labels[mask]

            # Gaussian kernel similarity
            diffs_sq = (cur - prev) ** 2
            gauss_sim = torch.exp(-diffs_sq / (2.0 * sigma ** 2))
            result[f'lag{k}_autocorr_gaussian'] = float(gauss_sim.mean().item())

            # Pearson correlation
            if cur.numel() > 1:
                cur_std = cur.std(unbiased=True)
                prev_std = prev.std(unbiased=True)
                if cur_std > 0 and prev_std > 0:
                    cur_centered = cur - cur.mean()
                    prev_centered = prev - prev.mean()
                    corr = (cur_centered * prev_centered).mean() / (cur_std * prev_std)
                    result[f'lag{k}_autocorr_corr'] = float(corr.item())
                else:
                    result[f'lag{k}_autocorr_corr'] = 0.0
            else:
                result[f'lag{k}_autocorr_corr'] = 0.0

        return result

    # Classification path
    label_codes = pd.factorize(df[label_col], sort=False)[0].astype(np.int64)
    labels_t = torch.as_tensor(label_codes, device=device, dtype=torch.long)
    num_classes = int(labels_t.max().item()) + 1 if labels_t.numel() > 0 else 0
    if num_classes <= 0:
        return zero_result

    # Legacy: one-hot encoding with all history
    one_hot = F.one_hot(labels_t, num_classes=num_classes).to(torch.float32)
    prefix = torch.cumsum(one_hot, dim=0)
    hist = _prefix_by_entity(prefix, start_positions, entity_ord)
    valid_mask = prev_count > 0
    temporal_h = 0.0
    if valid_mask.any():
        same_counts = hist.gather(1, labels_t.view(-1, 1)).squeeze(1)
        temporal_h = float((same_counts[valid_mask] / prev_count[valid_mask].to(torch.float32)).mean().item())
    result['temporal_homophily_gaussian'] = temporal_h

    # Legacy: correlation-style metric using numeric label codes
    y_float = labels_t.to(torch.float64)
    y_prefix = torch.cumsum(y_float, dim=0)
    y2_prefix = torch.cumsum(y_float * y_float, dim=0)
    y_hist = _prefix_by_entity(y_prefix, start_positions, entity_ord)
    y2_hist = _prefix_by_entity(y2_prefix, start_positions, entity_ord)
    num_prev = prev_count.to(torch.float64)
    M_edges = num_prev.sum()
    temporal_corr = 0.0
    if M_edges.item() > 0:
        sum_yu = (y_float * num_prev).sum()
        sum_yu2 = (y_float * y_float * num_prev).sum()
        sum_yv = y_hist.sum()
        sum_yv2 = y2_hist.sum()
        sum_yuyv = (y_float * y_hist).sum()

        mu_u = sum_yu / M_edges
        mu_v = sum_yv / M_edges
        var_u = (sum_yu2 / M_edges) - mu_u * mu_u
        var_v = (sum_yv2 / M_edges) - mu_v * mu_v
        cov = (sum_yuyv / M_edges) - mu_u * mu_v
        denom = (var_u * var_v).clamp_min(1e-12).sqrt()
        temporal_corr = float((cov / denom).item())
    result['temporal_homophily_corr'] = temporal_corr

    # NEW: Lag-specific autocorrelation for classification
    n = len(labels_t)
    for k in lags:
        # Shift labels by k within each entity
        lagged_labels = torch.full((n,), -1, dtype=torch.long, device=device)
        for i in range(n):
            ent = entity_ord[i].item()
            if prev_count[i].item() >= k:
                start_pos = start_positions[ent].item()
                idx_within_entity = i - start_pos
                if idx_within_entity >= k:
                    lagged_idx = i - k
                    lagged_labels[i] = labels_t[lagged_idx]

        # Mask valid pairs
        mask = lagged_labels >= 0
        if not mask.any():
            result[f'lag{k}_autocorr_gaussian'] = 0.0
            result[f'lag{k}_autocorr_corr'] = 0.0
            continue

        cur = labels_t[mask]
        prev = lagged_labels[mask]

        # Equality rate (fraction where current == lagged)
        eq_rate = (cur == prev).float().mean()
        result[f'lag{k}_autocorr_gaussian'] = float(eq_rate.item())

        # For correlation, use numeric encoding
        cur_float = cur.to(torch.float64)
        prev_float = prev.to(torch.float64)
        if cur_float.numel() > 1:
            cur_std = cur_float.std(unbiased=True)
            prev_std = prev_float.std(unbiased=True)
            if cur_std > 0 and prev_std > 0:
                cur_centered = cur_float - cur_float.mean()
                prev_centered = prev_float - prev_float.mean()
                corr = (cur_centered * prev_centered).mean() / (cur_std * prev_std)
                result[f'lag{k}_autocorr_corr'] = float(corr.item())
            else:
                result[f'lag{k}_autocorr_corr'] = 0.0
        else:
            result[f'lag{k}_autocorr_corr'] = 0.0

    return result


# ============================================================================
# Homophily Features (from heuristics3.py)
# ============================================================================

@time_it
def compute_homophily_features(
    data: Any,
    task: Any,
    train_task_table: pd.DataFrame,
    val_task_table: pd.DataFrame,
    db_obj: Any,
    raw_train_task_obj: Any,
    raw_val_task_obj: Any,
    entity_col: str,
    entity_table_name: str,
    existing_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate homophily-based task embedding features.

    Args:
        data: The HeteroData graph object
        task: The task object with metadata
        train_task_table: Training task table as DataFrame
        val_task_table: Validation task table as DataFrame
        db_obj: Database object containing table metadata
        raw_train_task_obj: Raw training task object
        raw_val_task_obj: Raw validation task object
        entity_col: Name of the entity column (foreign key)
        entity_table_name: Name of the entity table
        existing_result: Optional cached homophily features to reuse regression-mode metrics

    Returns:
        Dictionary containing graph-based homophily features only
    """
    # Compute heuristic-based features
    heuristics_train_result_dict, heuristics_val_result_dict = heuristics(
        task, raw_train_task_obj, raw_val_task_obj
    )

    # Get database statistics
    num_of_tables, num_of_rows, num_of_columns = get_simple_stats(db_obj)

    # Aggregate train and validation tables for homophily computation
    aggregated_train_val_table = pd.concat([train_task_table, val_task_table])
    aggregated_train_val_table = aggregated_train_val_table.reset_index(drop=True)
    aggregated_table_obj = deepcopy(raw_train_task_obj)
    aggregated_table_obj.df = aggregated_train_val_table

    is_autocomplete = task.auto
    task_type = (
        task.task_type if isinstance(task.task_type, str) else task.task_type.value
    )

    homophily_kwargs = dict(
        data=data,
        labels_table_obj=aggregated_table_obj,
        db_obj=db_obj,
        task_entity=task.entity_table,
        label_col=task.target_col,
        task_type='classification' if 'classification' in task_type else 'regression',
        is_autocomplete=is_autocomplete,
        max_group=300,
    )
    is_regression_task = 'regression' in task_type
    existing_result = existing_result or {}

    label_soft: Optional[torch.Tensor] = None
    labeled_mask: Optional[torch.Tensor] = None

    base_metric_keys = ['h_edges', 'h_insensitives', 'h_adjs', 'h_aggs', 'sparsities']
    metapath_degree_stats: Dict[str, Dict[str, float]] = {}
    base_metrics: Dict[str, Any] = {}

    if is_regression_task:
        gaussian_required = base_metric_keys + ['metapath_degrees']
        gaussian_cached = all(key in existing_result for key in gaussian_required)
        if gaussian_cached:
            logger.info("Using cached gaussian homophily metrics.")
            base_metrics = {key: existing_result[key] for key in base_metric_keys}
            metapath_degree_stats = existing_result['metapath_degrees']
        else:
            logger.info("Computing gaussian homophily metrics.")
            homophily_result, label_soft, labeled_mask = homophily_from_table_db(
                **homophily_kwargs,
                regression_mode='gaussian',
            )
            (
                h_edges,
                h_insensitives,
                h_adjs,
                h_aggs,
                sparsities,
                metapath_degrees,
            ) = homophily_result
            logger.info(f"Homophily computation completed for {len(metapath_degrees)} metapaths")
            metapath_degree_stats = _summarize_metapath_degrees(metapath_degrees)
            base_metrics = {
                'h_edges': h_edges,
                'h_insensitives': h_insensitives,
                'h_adjs': h_adjs,
                'h_aggs': h_aggs,
                'sparsities': sparsities,
            }
    else:
        homophily_result, label_soft, labeled_mask = homophily_from_table_db(**homophily_kwargs)
        (
            h_edges,
            h_insensitives,
            h_adjs,
            h_aggs,
            sparsities,
            metapath_degrees,
        ) = homophily_result
        logger.info(f"Homophily computation completed for {len(metapath_degrees)} metapaths")
        metapath_degree_stats = _summarize_metapath_degrees(metapath_degrees)
        base_metrics = {
            'h_edges': h_edges,
            'h_insensitives': h_insensitives,
            'h_adjs': h_adjs,
            'h_aggs': h_aggs,
            'sparsities': sparsities,
        }

    regression_modes_info: Dict[str, Dict[str, Any]] = {}
    if is_regression_task:
        fake_keys = [
            'h_edges_fake_cls',
            'h_insensitives_fake_cls',
            'h_adjs_fake_cls',
            'h_aggs_fake_cls',
            'sparsities_fake_cls',
        ]
        if all(key in existing_result for key in fake_keys):
            logger.info("Using cached fake_cls homophily metrics.")
            regression_modes_info['fake_cls'] = {key: existing_result[key] for key in fake_keys}
        else:
            logger.info("Computing fake_cls homophily metrics.")
            fake_cls_result, _, _ = homophily_from_table_db(
                **homophily_kwargs,
                regression_mode='fake_cls',
                regression_bins=[4, 8, 12, 16],
            )
            (
                h_edges_fake,
                h_ins_fake,
                h_adjs_fake,
                h_aggs_fake,
                sparsities_fake,
                _,
            ) = fake_cls_result
            regression_modes_info['fake_cls'] = {
                'h_edges_fake_cls': h_edges_fake,
                'h_insensitives_fake_cls': h_ins_fake,
                'h_adjs_fake_cls': h_adjs_fake,
                'h_aggs_fake_cls': h_aggs_fake,
                'sparsities_fake_cls': sparsities_fake,
            }

        corr_keys = [
            'h_edges_corr',
            'h_insensitives_corr',
            'h_adjs_corr',
            'h_aggs_corr',
            'sparsities_corr',
        ]
        if all(key in existing_result for key in corr_keys):
            logger.info("Using cached corr homophily metrics.")
            regression_modes_info['corr'] = {key: existing_result[key] for key in corr_keys}
        else:
            logger.info("Computing corr homophily metrics.")
            corr_result, _, _ = homophily_from_table_db(
                **homophily_kwargs,
                regression_mode='corr',
            )
            (
                h_edges_corr,
                h_ins_corr,
                h_adjs_corr,
                h_aggs_corr,
                sparsities_corr,
                _,
            ) = corr_result
            regression_modes_info['corr'] = {
                'h_edges_corr': h_edges_corr,
                'h_insensitives_corr': h_ins_corr,
                'h_adjs_corr': h_adjs_corr,
                'h_aggs_corr': h_aggs_corr,
                'sparsities_corr': sparsities_corr,
            }

    # Cleanup
    del aggregated_train_val_table, aggregated_table_obj
    if label_soft is not None:
        del label_soft
    if labeled_mask is not None:
        del labeled_mask

    result = {
        'heuristics_train_result_dict': heuristics_train_result_dict,
        'heuristics_val_result_dict': heuristics_val_result_dict,
        'num_of_tables': num_of_tables,
        'num_of_rows': num_of_rows,
        'num_of_columns': num_of_columns,
        'metapath_degrees': metapath_degree_stats,
        'h_edges': base_metrics['h_edges'],
        'h_insensitives': base_metrics['h_insensitives'],
        'h_adjs': base_metrics['h_adjs'],
        'h_aggs': base_metrics['h_aggs'],
        'sparsities': base_metrics['sparsities'],
    }

    if is_regression_task:
        result.update(regression_modes_info.get('fake_cls', {}))
        result.update(regression_modes_info.get('corr', {}))
        result['regression_modes'] = ['gaussian', 'fake_cls', 'corr']

    return result


# ============================================================================
# Temporal Features
# ============================================================================

@time_it
def compute_temporal_features(
    task: Any,
    raw_train_task_obj: Any,
    raw_val_task_obj: Any,
    entity_col: str,
    entity_table_name: str,
    existing_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute temporal features including lag-specific autocorrelation.

    This function computes temporal characteristics of the task data:
    - Lag-specific autocorrelation for lags {1, 2, 3}
    - Entity duplication rates
    - Legacy temporal homophily metrics (for backward compatibility)

    Args:
        task: Task object
        raw_train_task_obj: Raw training task object
        raw_val_task_obj: Raw validation task object
        entity_col: Name of the entity column (foreign key)
        entity_table_name: Name of the entity table
        existing_result: Optional cached features to reuse

    Returns:
        Dictionary containing temporal features
    """
    logger.info("Computing temporal features")

    # Get task tables
    train_task_table = raw_train_task_obj.df
    val_task_table = raw_val_task_obj.df

    combined_task_table = pd.concat([train_task_table, val_task_table])
    combined_task_table = combined_task_table.reset_index(drop=True)

    # Determine task type
    task_type_str = (
        task.task_type if isinstance(task.task_type, str) else task.task_type.value
    )
    is_regression = 'regression' in task_type_str


    # Compute entity duplication rates
    duplication_rate = len(combined_task_table) / len(
        combined_task_table[entity_col].drop_duplicates()
    )

    # Compute GPU-accelerated temporal axis metrics (includes lag-specific autocorrelation)
    temporal_axis = compute_temporal_axis_gpu(
        train_task_table,
        entity_col=entity_col,
        time_col=task.time_col,
        label_col=task.target_col,
        task_type=task_type_str,
        lags=(1, 2, 3),
    )

    result = {
        # Legacy temporal homophily metrics (all-history)
        'temporal_homophily_gaussian': temporal_axis['temporal_homophily_gaussian'],
        'temporal_homophily_corr': temporal_axis['temporal_homophily_corr'],

            # GPU-computed lag-specific autocorrelation
        'lag1_autocorr_gaussian': temporal_axis.get('lag1_autocorr_gaussian', 0.0),
        'lag2_autocorr_gaussian': temporal_axis.get('lag2_autocorr_gaussian', 0.0),
        'lag3_autocorr_gaussian': temporal_axis.get('lag3_autocorr_gaussian', 0.0),
        'lag1_autocorr_corr': temporal_axis.get('lag1_autocorr_corr', 0.0),
        'lag2_autocorr_corr': temporal_axis.get('lag2_autocorr_corr', 0.0),
        'lag3_autocorr_corr': temporal_axis.get('lag3_autocorr_corr', 0.0),
        'duplication_rate': duplication_rate,
    }

    logger.info("Temporal features computation completed")
    return result


# ============================================================================
# Auto-Transfer Features (from heuristics_at.py)
# ============================================================================

def compute_autotransfer_features(
    database: Any,
    task_id: int,
    basic_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute auto-transfer task embeddings.

    Args:
        database: Database object
        task_id: Task index
        basic_config: Basic configuration dictionary

    Returns:
        Dictionary containing auto-transfer embeddings
    """
    logger.info("Computing auto-transfer features (RDL)")
    ta_th_at = generate_model_task_embedding(
        database,
        task_id,
        basic_config,
        method_type='training-head',
        model_name='auto-transfer',
        anchor_config=SIX_ANCHOR_CONFIG
    )

    clear_task_memory()

    return {'ta_th_at': ta_th_at}


def compute_autotransfer_dfs_features(
    database: Any,
    task: Any,
    task_id: int,
    basic_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute auto-transfer DFS (Deep Feature Synthesis) task embeddings.

    Args:
        database: Database object
        task: Task object
        task_id: Task index
        basic_config: Basic configuration dictionary

    Returns:
        Dictionary containing DFS-based auto-transfer embeddings
    """
    logger.info("Computing auto-transfer DFS features")

    # Generate DFS data if not exists
    one_layer_scaler_path_name = f"scaler_{database.database.name}_{task.name}_1"
    two_layer_scaler_path_name = f"scaler_{database.database.name}_{task.name}_2"

    # Generate 2-layer and 1-layer DFS datasets
    two_layer_rdb_dataset = generate_dfs_data_pre_post(
        "datasets/dfs",
        database,
        max_depth=2,
        load_scalers_from=two_layer_scaler_path_name,
        lte=task.auto,
        text_to_pca=True,
        text_to_pca_dim=3,
        task_idx=task_id
    )

    one_layer_rdb_dataset = generate_dfs_data_pre_post(
        "datasets/dfs",
        database,
        max_depth=1,
        load_scalers_from=one_layer_scaler_path_name,
        lte=task.auto,
        text_to_pca=True,
        text_to_pca_dim=3,
        task_idx=task_id
    )

    # Configure model and compute embeddings
    ta_th_at_dfs_2 = generate_model_task_embedding(
        two_layer_rdb_dataset,
        task_id,
        basic_config,
        method_type='training-head',
        model_name='auto-transfer',
        anchor_config=DFS_ANCHOR_CONFIG,
        model_type='dfs'
    )

    ta_th_at_dfs_1 = generate_model_task_embedding(
        one_layer_rdb_dataset,
        task_id,
        basic_config,
        method_type='training-head',
        model_name='auto-transfer',
        anchor_config=DFS_ANCHOR_CONFIG,
        model_type='dfs'
    )

    clear_task_memory()

    return {
        'ta_th_at_dfs_2': ta_th_at_dfs_2,
        'ta_th_at_dfs_1': ta_th_at_dfs_1,
    }


def compute_split_row_stats(task) -> Dict[str, int]:
    """Compute the number of rows in each task split."""
    stats = {}
    for split in ("train", "val", "test"):
        if split == "test":
            table = task.get_table(split, mask_input_cols=False)
        else:
            table = task.get_table(split)
        length = len(table.df) if hasattr(table, "df") else len(table)
        stats[f"{split}_rows"] = int(length)
    return stats


# ============================================================================
# Same-Class Analysis Features
# ============================================================================

# Manual overrides for mapping (db_name, task_name) -> (table_name, target_col)
# when SameClassAnalyzer needs a different table/column than the relbench task name
_SAME_CLASS_MANUAL_OVERRIDES: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("rel-avs", "repeater"): ("history", "repeater"),
    ("rel-f1", "constructor-scores-points"): ("constructor-scores-points", "scored_points"),
    ("rel-f1", "driver-position-change"): ("driver-position-change", "position_change"),
    ("rel-ieeecis", "transaction-fraud"): ("transaction-fraud", "isFraud"),
    ("rel-stack", "user-comment-count"): ("user-comment-count", "comment_count"),
    ("rel-arxiv", "paper-citation"): ("paper-citation", "cited"),
    ("rel-arxiv", "author-publication"): ("author-publication", "publication_count"),
}


def _resolve_same_class_tuple(db_name: str, task_name: str) -> Tuple[str, str, str, str]:
    """
    Resolve (db_name, table_name, target_col, task_type) for SameClassAnalyzer.

    Maps from relbench (db_name, task_name) to the tuple expected by
    data.rt.data.RelationalDataset, handling manual overrides for tasks
    whose table/column names differ from the standard pattern.
    """
    if (db_name, task_name) in _SAME_CLASS_MANUAL_OVERRIDES:
        table_name, target_col = _SAME_CLASS_MANUAL_OVERRIDES[(db_name, task_name)]
    else:
        table_name = task_name
        if '-' in task_name:
            _, target_hint = task_name.split('-', 1)
            target_col = target_hint.replace('-', '_')
        else:
            target_col = task_name

    task_type = 'regression' if (db_name, table_name) in SAME_CLASS_REGRESSION_TASKS else 'classification'
    return (db_name, table_name, target_col, task_type)


def compute_same_class_features(
    db_name: str,
    task_name: str,
    split: str = 'train',
    num_samples: int = 10000,
    batch_size: int = 32,
    seq_len: int = 512,
    max_bfs_width: int = 256,
    seed: int = 42,
    use_rfm_preprocessing: bool = False,
    embedding_model: str = 'all-MiniLM-L12-v2',
    d_text: int = 384,
) -> Dict[str, Any]:
    """
    Compute same-class analysis features for a single task.

    Uses SameClassAnalyzer to sample sequences from data.rt.data.RelationalDataset
    and compute same-class ratio statistics (classification) or label correlation
    (regression).

    Returns a feature payload dict ready for caching, containing keys like:
      sparsity_ratio, adjusted_mean_same_class_ratio, mean_past_task_nodes,
      mean_same_class_ratio_ignore (classification) / mean_label_correlation (regression),
      variance_same_class_ratio (classification) / variance_label_correlation (regression).
    """
    task_tuple = _resolve_same_class_tuple(db_name, task_name)
    logger.info(f"Computing same-class features for {db_name}/{task_name} -> {task_tuple}")

    analyzer = SameClassAnalyzer(
        task_tuple=task_tuple,
        split=split,
        num_samples=num_samples,
        batch_size=batch_size,
        seq_len=seq_len,
        max_bfs_width=max_bfs_width,
        seed=seed,
        use_rfm_preprocessing=use_rfm_preprocessing,
        empty_handling='ignore',
        embedding_model=embedding_model,
        d_text=d_text,
    )

    result = analyzer.analyze()
    payload = _same_class_payload(result)
    return payload




@torch.no_grad()
def test_rfr_model(
    model, task, num_dst_nodes_dict, task_meta, meta_graph, loader,
    res_task: Union[pd.DataFrame, Table], device, stage='val', loader_type='node'
):
    """Evaluate RandomFeatureRegressor model."""
    if len(loader) == 0:
        logger.debug(f"No data for {stage} {task.name}")
        return {}, -1

    model.eval()
    model.to(device)

    if task.task_type == TaskType.LINK_PREDICTION or task.task_type == "link_prediction":
        entity_table = task.src_entity_table
        target_table = task.dst_entity_table
    else:
        entity_table = task.entity_table
        target_table = task.entity_table

    if isinstance(loader, tuple):
        src_loader, dst_loader = loader

    pred_list = []
    if loader_type == 'node':
        pbar = tqdm(loader, total=len(loader), desc=f"{stage} Test")
    else:
        pbar = tqdm(dst_loader, total=len(dst_loader), desc=f"{stage} Test")

    if loader_type == 'node':
        for batch in pbar:
            batch = batch.to(device)

            target_output = generate_representation(
                model, device, batch, entity_table, target_table, meta_graph
            )

            seed_time = batch[entity_table].seed_time
            if target_output.ndim == 2 and target_output.size(1) == 1:
                pred = target_output.view(-1)[:seed_time.size(0)]
            else:
                pred = target_output[:seed_time.size(0)]

            if task.task_type in (TaskType.REGRESSION, 'regression'):
                if pred.ndim == 2 and pred.size(1) == 1:
                    pred = pred.view(-1)
                if task_meta.get('clamp_min') is not None and task_meta.get('clamp_max') is not None:
                    pred = torch.clamp(pred, task_meta['clamp_min'], task_meta['clamp_max'])
            else:
                if pred.ndim == 2 and pred.size(1) > 1:
                    pred = torch.softmax(pred, dim=1)
                else:
                    pred = torch.sigmoid(pred.view(-1))

            pred_list.append(pred.detach().cpu())

        pred_list = torch.cat(pred_list, dim=0)
        metric = task.evaluate(pred_list.numpy(), res_task)
        return metric
    else:
        # Link/edge loader path
        model.eval()

        dst_embs: list[Tensor] = []
        for batch in tqdm(dst_loader):
            batch = batch.to(device)
            emb = model(batch, task.dst_entity_table,
                        task.dst_entity_table, read_out_table='dst')[:batch[task.dst_entity_table]['seed_time'].size(0)]
            dst_embs.append(emb)
        dst_emb = torch.cat(dst_embs, dim=0)
        del dst_embs

        pred_index_mat_list: list[Tensor] = []
        for batch in tqdm(src_loader):
            batch = batch.to(device)
            emb = model(batch, task.src_entity_table, task.dst_entity_table, read_out_table='src')[:batch[task.src_entity_table]['seed_time'].size(0)]
            _, pred_index_mat = torch.topk(emb @ dst_emb.t(), k=task.eval_k, dim=1)
            pred_index_mat_list.append(pred_index_mat.cpu())
        pred = torch.cat(pred_index_mat_list, dim=0).numpy()
        metric = task.evaluate(pred, res_task)
        return metric


def random_rdl_embedding(
    configs, database, task, task_id, basic_config, data_config, device, wanted_metric
):
    """Run random feature regressor on multiple model configurations."""
    randomsage = []
    for search_params in configs:
        model_config = {**basic_config, **data_config}
        model_config['model_params'] = search_params

        base_ctor = get_auto_model(model_config['model_params']["model_type"])
        model_init = {**model_config['model_params']}
        model_init['col_stats_dict'] = database.get_col_stats_dict()

        task_info = task
        task_info.task_type = task_info.task_type.value if not isinstance(task_info.task_type, str) else task_info.task_type

        if task_info.task_type in ("recommendation", "link_prediction"):
            model_init['src_table'] = task_info.src_entity_table
            model_init['dst_table'] = task_info.dst_entity_table
        else:
            model_init['src_table'] = task_info.entity_table
            model_init['dst_table'] = task_info.entity_table

        if model_config['model_params']["model_type"] == "rdl":
            model_init = configure_rdl(model_init, database, model_config, task_info)

        base_model = base_ctor(**model_init)

        loader_dict, num_dst_nodes_dict, task_meta, meta_graph, sparse_tensor, val_res_task = \
            database.initialize_data(
                task_id, 0, 1,
                model_config['model_params']['sampler_config']['loader_type'],
                model_config
            )

        if model_config['model_params']['sampler_config']['loader_type'] != 'node':
            logger.info("RFR currently supports node loader; skipping this config.")
            continue

        # Wrap with RFR and fit
        if task_info.task_type in ('regression', TaskType.REGRESSION):
            rfr_model = RandomFeatureRegressor(
                base_model, task_kind='regression',
                ridge_lambda=float(model_config['model_params'].get('ridge_lambda', 1.0)),
                standardize=True
            ).to(device)
            clamp_min = task_meta.get('clamp_min', None)
            clamp_max = task_meta.get('clamp_max', None)
            train_loss = rfr_model.fit_from_loader(
                loader_dict['train'],
                entity_table=task_info.entity_table,
                dst_table=task_info.entity_table,
                device=torch.device(device),
                clamp_min=clamp_min, clamp_max=clamp_max
            )
            logger.info(f"[{database.database.name}/{task.name}] RFR (regression) train MAE: {train_loss:.6f}")
        else:
            rfr_model = RandomFeatureRegressor(
                base_model, task_kind='binary',
                lda_shrinkage=float(model_config['model_params'].get('lda_shrinkage', 1e-3)),
                standardize=True
            ).to(device)
            train_loss = rfr_model.fit_from_loader(
                loader_dict['train'],
                entity_table=task_info.entity_table,
                dst_table=task_info.entity_table,
                device=torch.device(device)
            )
            logger.info(f"[{database.database.name}/{task.name}] RFR (binary) train BCE: {train_loss:.6f}")

        # Evaluate on val
        val_loader = loader_dict['val']
        res_table = val_res_task
        val_metric = test_rfr_model(
            rfr_model, task, num_dst_nodes_dict, task_meta, meta_graph,
            val_loader, res_table, torch.device(device),
            stage='val',
            loader_type=model_config['model_params']['sampler_config']['loader_type']
        )[wanted_metric]
        randomsage.append(val_metric)
    return randomsage


def compute_hit_features(
    database: MultiTabularDataset,
    task: Any,
    task_id: int,
    basic_config: Dict[str, Any],
    data_config: Dict[str, Any],
    device: str,
    validated_results_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute hit rate features using heuristics and random feature regressors.

    Args:
        database: Database object
        task: Task object
        task_id: Task index
        basic_config: Basic configuration dictionary
        data_config: Data configuration dictionary
        device: Device to use ('cuda' or 'cpu')
        validated_results_path: Path to validated runs CSV (deprecated, computed online now)

    Returns:
        Dictionary containing hit rate features
    """
    logger.info("Computing hit rate features")

    anchor_model_embedding = {}

    # Determine the metric to use based on task type
    wanted_metric = 'mae' if task.task_type in ['regression', TaskType.REGRESSION] else 'roc_auc'

    dataset_name = database.database.name
    task_name = database.tasks[task_id].name

    try:
        validated_runs = load_validated_runs(entity=os.getenv("WANDB_ENTITY", "msuczk"), refresh=False)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(f"Failed to load validated runs via load_validated_runs: {exc}")
        if validated_results_path and os.path.exists(validated_results_path):
            validated_runs = pd.read_csv(validated_results_path)
        else:
            validated_runs = pd.DataFrame()

    tabpfn_df = validated_runs.copy()
    if 'torch_frame_model_cls' in tabpfn_df.columns:
        tabpfn_df['torch_frame_model_cls'] = tabpfn_df['torch_frame_model_cls'].astype(str)
    else:
        tabpfn_df['torch_frame_model_cls'] = ''
    tabpfn_df = tabpfn_df[
        (tabpfn_df['dataset_name'] == dataset_name) &
        (tabpfn_df['task_name'] == task_name) &
        (tabpfn_df['torch_frame_model_cls'].str.lower() == 'tabpfn')
    ]
    if 'main_metric_name' in tabpfn_df.columns:
        tabpfn_df = tabpfn_df[tabpfn_df['main_metric_name'] == wanted_metric]

    def _select_tabpfn_perf(hop: int) -> float:
        if tabpfn_df.empty:
            return 0.0
        layer_df = tabpfn_df[tabpfn_df['num_layers'] == hop] if 'num_layers' in tabpfn_df else pd.DataFrame()
        if layer_df.empty and 'dfs_layer' in tabpfn_df:
            layer_df = tabpfn_df[tabpfn_df['dfs_layer'] == hop]
        if layer_df.empty:
            return 0.0
        metric_series = layer_df['val_metric'].dropna()
        if metric_series.empty:
            return 0.0
        best_val = metric_series.min() if wanted_metric == 'mae' else metric_series.max()
        return float(best_val)

    tabpfn_1hop_perf = _select_tabpfn_perf(1)
    tabpfn_2hop_perf = _select_tabpfn_perf(2)

    anchor_model_embedding['tabpfn'] = {
        '1hop': float(tabpfn_1hop_perf),
        '2hop': float(tabpfn_2hop_perf)
    }

    # Compute heuristics baselines
    train_task_table = task.get_table('train')
    val_task_table = task.get_table('val')

    heuristics_train_result_dict, heuristics_val_result_dict = heuristics(
        task, train_task_table, val_task_table
    )

    # Convert numpy scalars to python floats
    for heur_dict in [heuristics_train_result_dict, heuristics_val_result_dict]:
        for heuristic_name, results_list in heur_dict.items():
            for result_dict in results_list:
                for metric_name, value in list(result_dict.items()):
                    if hasattr(value, 'item'):
                        result_dict[metric_name] = float(value.item())
                    elif isinstance(value, np.ndarray) and value.shape == ():
                        result_dict[metric_name] = float(value)

    # wanted_metric already defined at the beginning of the function
    train_baseline = heuristics_train_result_dict['global_zero'][0][wanted_metric]
    val_baseline = heuristics_val_result_dict['global_zero'][0][wanted_metric]

    train_heuristics = {
        key: heuristics_train_result_dict[key][0][wanted_metric]
        for key in heuristics_train_result_dict.keys()
    }
    val_heuristics = {
        key: heuristics_val_result_dict[key][0][wanted_metric]
        for key in heuristics_val_result_dict.keys()
    }

    anchor_model_embedding['heuristics'] = {
        'train_baseline': float(train_baseline),
        'val_baseline': float(val_baseline),
        'train_heuristics': train_heuristics,
        'val_heuristics': val_heuristics
    }

    # Random-feature GNN runs
    randomnbfnet_configs = [load_yaml(f"configs/model/randomsage{i}.yaml") for i in range(1, 4)]
    randomsage_configs = [load_yaml(f"configs/model/sage{i}.yaml") for i in range(1, 4)]

    graph_construct_sentinel = object()
    prev_graph_construct = getattr(database, "graph_construct", graph_construct_sentinel)
    database.graph_construct = 'r2n'
    if hasattr(database, "clear_cached_graph"):
        database.clear_cached_graph()
    try:
        randomnbfnet = random_rdl_embedding(
            randomnbfnet_configs, database, database.tasks[task_id], task_id, basic_config, data_config, device, wanted_metric
        )
        randomsage = random_rdl_embedding(
            randomsage_configs, database, database.tasks[task_id], task_id, basic_config, data_config, device, wanted_metric
        )
    finally:
        if prev_graph_construct is graph_construct_sentinel:
            delattr(database, "graph_construct")
        else:
            database.graph_construct = prev_graph_construct

    anchor_model_embedding['rfr_randomsage'] = randomsage
    anchor_model_embedding['rfr_randomnbfnet'] = randomnbfnet

    return anchor_model_embedding


# ============================================================================
# Unified Task Embedding Execution
# ============================================================================

def _is_regression(task: Any) -> bool:
    task_type = task.task_type if isinstance(task.task_type, str) else str(task.task_type)
    return 'regression' in task_type.lower()


def _required_keys_for_feature(feature: str, task: Optional[Any] = None) -> Iterable[Any]:
    if feature == 'homophily':
        keys = {
            'heuristics_train_result_dict',
            'heuristics_val_result_dict',
            'num_of_tables',
            'num_of_rows',
            'num_of_columns',
            'metapath_degrees',
            'h_edges',
            'h_insensitives',
            'h_adjs',
            'h_aggs',
            'sparsities',
        }
        if task is not None and _is_regression(task):
            keys.update(
                {
                    'h_edges_fake_cls',
                    'h_insensitives_fake_cls',
                    'h_adjs_fake_cls',
                    'h_aggs_fake_cls',
                    'sparsities_fake_cls',
                    'h_edges_corr',
                    'h_insensitives_corr',
                    'h_adjs_corr',
                    'h_aggs_corr',
                    'sparsities_corr',
                    'regression_modes',
                }
            )
        return keys
    if feature == 'temporal':
        return {
            'train_lag1_autocorr',
            'train_lag2_autocorr',
            'train_lag3_autocorr',
            'val_lag1_autocorr',
            'val_lag2_autocorr',
            'val_lag3_autocorr',
            'train_duplication_rate',
            'val_duplication_rate',
            'temporal_homophily_gaussian',
            'temporal_homophily_corr',
            'lag1_autocorr_gaussian',
            'lag2_autocorr_gaussian',
            'lag3_autocorr_gaussian',
            'lag1_autocorr_corr',
            'lag2_autocorr_corr',
            'lag3_autocorr_corr',
        }
    if feature == 'autotransfer':
        return {'ta_th_at'}
    if feature == 'autotransfer_dfs':
        return {'ta_th_at_dfs_1', 'ta_th_at_dfs_2'}
    if feature == 'stats':
        return {'train_rows', 'val_rows', 'test_rows'}
    if feature == 'hit':
        return {
            ('tabpfn', '1hop'),
            ('tabpfn', '2hop'),
            'heuristics',
            'rfr_randomsage',
            'rfr_randomnbfnet',
        }
    if feature == 'same_class':
        return {
            'sparsity_ratio',
            'adjusted_mean_same_class_ratio',
            'mean_past_task_nodes',
        }
    return set()


def _has_required_keys(data: Dict[str, Any], required_keys: Iterable[Any]) -> bool:
    for key in required_keys:
        if isinstance(key, tuple):
            current = data
            missing = False
            for part in key:
                if not isinstance(current, dict) or part not in current:
                    missing = True
                    break
                current = current[part]
            if missing:
                return False
        else:
            if key not in data:
                return False
    return True


def _feature_is_complete(path: str, feature: str, task: Optional[Any] = None) -> bool:
    if not os.path.exists(path):
        return False
    existing = torch.load(path, map_location='cpu')
    if feature == 'homophily' and task is not None and _is_regression(task):
        legacy_keys = {
            'h_edges_fake_cls',
            'h_insensitives_fake_cls',
            'h_adjs_fake_cls',
            'h_aggs_fake_cls',
            'sparsities_fake_cls',
        }
        has_legacy_fake = legacy_keys.issubset(existing.keys())
        if has_legacy_fake and 'regression_modes' not in existing:
            existing['regression_modes'] = ['gaussian', 'fake_cls']
            torch.save(existing, path)
        corr_keys = {
            'h_edges_corr',
            'h_insensitives_corr',
            'h_adjs_corr',
            'h_aggs_corr',
            'sparsities_corr',
        }
        if corr_keys.issubset(existing.keys()):
            modes = list(existing.get('regression_modes', []))
            if 'corr' not in modes:
                modes.append('corr')
                existing['regression_modes'] = modes
                torch.save(existing, path)
    required_keys = _required_keys_for_feature(feature, task)
    if not required_keys:
        return True
    if _has_required_keys(existing, required_keys):
        logger.info(f"Existing {feature} cache at {path} is complete; skipping recomputation.")
        return True
    logger.info(f"{feature} cache at {path} missing required keys; recomputing.")
    return False


class _LightweightTable:
    """Minimal table wrapper loaded directly from a cached parquet file."""
    def __init__(self, df: pd.DataFrame, fkey_col_to_pkey_table: Dict[str, str],
                 time_col: Optional[str] = None):
        self.df = df
        self.fkey_col_to_pkey_table = fkey_col_to_pkey_table
        self.time_col = time_col


class _LightweightTask:
    """Minimal task wrapper that loads tables directly from cached parquet files.

    Bypasses relbench's AutoCompleteTask.__init__() which forces a full database
    load (~2-3 GB for rel-ieeecis) just to extract schema metadata.  Instead,
    we read the parquet schema (zero data) for metadata, and load only the
    3-column task tables when get_table() is called.
    """

    def __init__(self, db_name: str, task_name: str, cache_dir: str):
        from utils.information import task_name_to_task_type
        import pyarrow.parquet as pq

        self.name = task_name
        self._cache_dir = cache_dir
        self._task_dir = os.path.join(cache_dir, db_name, 'tasks', task_name)

        # Read metadata from train parquet schema (no data loaded)
        train_path = os.path.join(self._task_dir, 'train.parquet')
        pf = pq.ParquetFile(train_path)
        schema = pf.schema_arrow
        import json
        meta = {
            k.decode(): json.loads(v.decode())
            for k, v in schema.metadata.items()
            if k in [b'fkey_col_to_pkey_table', b'pkey_col', b'time_col']
        }
        fk_map = meta.get('fkey_col_to_pkey_table', {})
        self.time_col = meta.get('time_col', None)

        # Derive entity_col and entity_table from FK mapping
        # The FK column that points to entity_table is the entity_col
        if len(fk_map) == 1:
            self.entity_col, self.entity_table = next(iter(fk_map.items()))
        else:
            # Multiple FKs — pick the first one (typical for relbench tasks)
            self.entity_col, self.entity_table = next(iter(fk_map.items()))

        # Derive target_col: column that is not FK, not time
        all_cols = set(schema.names)
        fk_cols = set(fk_map.keys())
        time_cols = {self.time_col} if self.time_col else set()
        target_cols = all_cols - fk_cols - time_cols
        self.target_col = next(iter(target_cols)) if target_cols else None

        # Task type from information.py
        raw_type = task_name_to_task_type.get(task_name, 'binary_classification')
        if 'regression' in raw_type:
            self.task_type = 'regression'
        else:
            self.task_type = 'binary_classification'

        # Auto-complete flag
        self.auto = raw_type.startswith('auto_')
        self.out_channels = 1

        self._tables: Dict[str, _LightweightTable] = {}
        logger.info(
            f"  Direct parquet task: {task_name} "
            f"(entity={self.entity_col}→{self.entity_table}, "
            f"target={self.target_col}, type={self.task_type})"
        )

    def get_table(self, split: str, mask_input_cols: bool = True) -> _LightweightTable:
        if split not in self._tables:
            import pyarrow.parquet as pq
            import json
            path = os.path.join(self._task_dir, f'{split}.parquet')
            pf = pq.ParquetFile(path)
            df = pf.read().to_pandas()
            meta = {
                k.decode(): json.loads(v.decode())
                for k, v in pf.schema_arrow.metadata.items()
                if k in [b'fkey_col_to_pkey_table', b'pkey_col', b'time_col']
            }
            self._tables[split] = _LightweightTable(
                df=df,
                fkey_col_to_pkey_table=meta.get('fkey_col_to_pkey_table', {}),
                time_col=meta.get('time_col', None),
            )
        return self._tables[split]


class _LightweightDatabase:
    """Minimal wrapper mimicking MultiTabularDataset for lightweight features.

    Only loads task parquet files directly — skips relbench API and full
    database loading, saving GBs of RAM for large datasets like rel-ieeecis.
    """
    def __init__(self, db_name: str, tasks):
        self.tasks = tasks
        self.database = type('_DB', (), {'name': db_name})()


def _prepare_lightweight_datasets(data_config: Dict[str, Any], cache_dir: str):
    """Load task tables directly from cached parquet — no relbench API, no full DB."""
    db_split_key = "dbs"
    if db_split_key not in data_config:
        raise ValueError(f"'{db_split_key}' not found in data_config")

    for entry in data_config[db_split_key]:
        db_name = entry["db_name"]
        task_names = entry["task_name"]
        logger.info(f"Direct parquet load: {db_name} (bypassing relbench API)")
        tasks = [_LightweightTask(db_name, tn, cache_dir) for tn in task_names]
        yield _LightweightDatabase(db_name, tasks)


def task_embedding_execution(
    basic_config: Dict[str, Any],
    data_config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """
    Execute task embedding computation for all databases and tasks.

    This function processes each database and task, computing selected features
    based on args.features, then saves the results to disk.

    Args:
        basic_config: Basic configuration dictionary with global settings
        data_config: Data configuration dictionary specifying databases and tasks
        args: Command-line arguments containing execution parameters
            - features: List of features to compute (homophily, autotransfer, autotransfer_dfs, hit)
            - refresh: Whether to recompute existing embeddings
            - root_dir: Directory to save task embeddings

    Returns:
        None. Results are saved to './taskemb/{database}-{task}-*.pt'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.root_dir, exist_ok=True)

    # Use lightweight loading if only temporal/stats/same_class features are requested
    needs_full_db = bool(set(args.features) & _HEAVY_FEATURES)
    if needs_full_db:
        databases = prepare_transductive_rdb_datasets_save_memory(data_config, basic_config)
    else:
        cache_dir = basic_config.get("cache_dir", "cache_data")
        logger.info("Only lightweight features requested — loading directly from cached parquets")
        databases = _prepare_lightweight_datasets(data_config, cache_dir=cache_dir)

    db_filter = args.db_name if args.db_name else None
    task_filter = args.task_name if args.task_name else None
    compute_needed_only = not db_filter and not task_filter

    for database in tqdm(databases, desc="Databases"):
        if db_filter and database.database.name != db_filter:
            continue
        for task_id, task in enumerate(tqdm(database.tasks, desc="Tasks")):
            if task_filter and task.name != task_filter:
                continue
            if compute_needed_only and task.name not in NEEDED_TASKS:
                continue
            # ============================================================
            # Homophily Features
            # ============================================================
            if 'homophily' in args.features:
                output_path = os.path.join(
                    args.root_dir,
                    f"{database.database.name}-{task.name}-model-free-gpu.pt"
                )

                if args.refresh or not _feature_is_complete(output_path, 'homophily', task):
                    logger.info(f"Computing homophily features for {database.database.name}-{task.name}")

                    # Setup database environment (no DuckDB needed)
                    _graph_construct_sentinel = object()
                    previous_graph_construct = getattr(database, "graph_construct", _graph_construct_sentinel)
                    database.graph_construct = 'r2ne_for_heuristics'

                    # Load task tables
                    raw_train_task_obj = task.get_table('train')
                    raw_val_task_obj = task.get_table('val')
                    raw_test_task_obj = task.get_table('test')
                    train_task_table = raw_train_task_obj.df
                    val_task_table = raw_val_task_obj.df
                    test_task_table = raw_test_task_obj.df

                    # Extract entity information
                    entity_table_name = task.entity_table
                    if task.entity_col not in raw_train_task_obj.fkey_col_to_pkey_table:
                        raise ValueError(
                            f"Expected entity column '{task.entity_col}' to appear in task table foreign keys. "
                            f"Available keys: {list(raw_train_task_obj.fkey_col_to_pkey_table.keys())}"
                        )
                    entity_col = task.entity_col

                    # Get database and graph objects
                    db_obj = database.database.get_db()
                    data, _, _ = database.get_db(task_idx=task_id)

                    existing_homophily = None
                    if (not args.refresh) and os.path.exists(output_path):
                        try:
                            existing_homophily = torch.load(output_path, map_location='cpu')
                        except Exception as exc:
                            logger.warning(
                                f"Failed to load existing homophily cache from {output_path}: {exc}"
                            )

                    # Compute homophily features
                    homophily_features = compute_homophily_features(
                        data, task, train_task_table, val_task_table,
                        db_obj, raw_train_task_obj, raw_val_task_obj,
                        entity_col, entity_table_name,
                        existing_result=existing_homophily,
                    )

                    # Save results
                    torch.save(homophily_features, output_path)
                    logger.info(f"Successfully saved homophily features to {output_path}")

                    # Cleanup
                    del train_task_table, val_task_table, test_task_table
                    del raw_train_task_obj, raw_val_task_obj, raw_test_task_obj
                    del data, db_obj
                    #clear_task_memory()
                    if previous_graph_construct is _graph_construct_sentinel:
                        delattr(database, "graph_construct")
                    else:
                        database.graph_construct = previous_graph_construct
                else:
                    logger.info(f"Homophily features for {database.database.name}-{task.name} already exist, skipping")

            # ============================================================
            # Temporal Features
            # ============================================================
            if 'temporal' in args.features:
                output_path = os.path.join(
                    args.root_dir,
                    f"{database.database.name}-{task.name}-temporal.pt"
                )

                if args.refresh or not _feature_is_complete(output_path, 'temporal', task):
                    logger.info(f"Computing temporal features for {database.database.name}-{task.name}")

                    # Temporal only needs task tables — no graph or DB loading required
                    raw_train_task_obj = task.get_table('train')
                    raw_val_task_obj = task.get_table('val')
                    entity_col = task.entity_col
                    entity_table_name = task.entity_table

                    existing_temporal = None
                    if (not args.refresh) and os.path.exists(output_path):
                        try:
                            existing_temporal = torch.load(output_path, map_location='cpu')
                        except Exception as exc:
                            logger.warning(
                                f"Failed to load existing temporal cache from {output_path}: {exc}"
                            )

                    # Compute temporal features
                    temporal_features = compute_temporal_features(
                        task,
                        raw_train_task_obj, raw_val_task_obj,
                        entity_col, entity_table_name,
                        existing_result=existing_temporal,
                    )

                    # Save results
                    torch.save(temporal_features, output_path)
                    logger.info(f"Successfully saved temporal features to {output_path}")

                    del raw_train_task_obj, raw_val_task_obj
                else:
                    logger.info(f"Temporal features for {database.database.name}-{task.name} already exist, skipping")

            # ============================================================
            # Auto-Transfer Features
            # ============================================================
            if 'autotransfer' in args.features:
                output_path = os.path.join(
                    args.root_dir,
                    f"{database.database.name}-{task.name}-model-atrdl.pt"
                )

                if args.refresh or not _feature_is_complete(output_path, 'autotransfer'):
                    logger.info(f"Computing auto-transfer features for {database.database.name}-{task.name}")
                    at_features = compute_autotransfer_features(database, task_id, basic_config)
                    torch.save(at_features, output_path)
                    logger.info(f"Successfully saved auto-transfer features to {output_path}")
                else:
                    logger.info(f"Auto-transfer features for {database.database.name}-{task.name} already exist, skipping")

            # ============================================================
            # Auto-Transfer DFS Features
            # ============================================================
            if 'autotransfer_dfs' in args.features:
                output_path = os.path.join(
                    args.root_dir,
                    f"{database.database.name}-{task.name}-model-atdfs.pt"
                )

                if args.refresh or not _feature_is_complete(output_path, 'autotransfer_dfs'):
                    logger.info(f"Computing auto-transfer DFS features for {database.database.name}-{task.name}")
                    atdfs_features = compute_autotransfer_dfs_features(database, task, task_id, basic_config)
                    torch.save(atdfs_features, output_path)
                    logger.info(f"Successfully saved auto-transfer DFS features to {output_path}")
                else:
                    logger.info(f"Auto-transfer DFS features for {database.database.name}-{task.name} already exist, skipping")

            # ============================================================
            # Hit Rate Features
            # ============================================================
            if 'hit' in args.features:
                output_path = os.path.join(
                    args.root_dir,
                    f"{database.database.name}-{task.name}-model-heuristics-full.pt"
                )

                if args.refresh or not _feature_is_complete(output_path, 'hit'):
                    logger.info(f"Computing hit rate features for {database.database.name}-{task.name}")
                    hit_features = compute_hit_features(
                        database, task, task_id, basic_config, data_config, device,
                        args.validated_results
                    )
                    torch.save(hit_features, output_path)
                    logger.info(f"Successfully saved hit rate features to {output_path}")
                else:
                    logger.info(f"Hit rate features for {database.database.name}-{task.name} already exist, skipping")

            if 'stats' in args.features:
                output_path = os.path.join(
                    args.root_dir,
                    f"{database.database.name}-{task.name}-table-stats.pt"
                )

                if args.refresh or not _feature_is_complete(output_path, 'stats'):
                    logger.info(f"Computing split row stats for {database.database.name}-{task.name}")
                    stats_features = compute_split_row_stats(task)
                    torch.save(stats_features, output_path)
                    logger.info(f"Successfully saved split row stats to {output_path}")
                else:
                    logger.info(f"Split row stats for {database.database.name}-{task.name} already exist, skipping")

            # ============================================================
            # Same-Class Analysis Features
            # ============================================================
            if 'same_class' in args.features:
                sc_tuple = _resolve_same_class_tuple(database.database.name, task.name)
                sc_db, sc_table, sc_target, sc_task_type = sc_tuple
                output_path = _same_class_cache_path(
                    args.root_dir, sc_db, sc_table, sc_target, 'train'
                )

                if args.refresh or not _same_class_cache_complete(output_path):
                    logger.info(f"Computing same-class features for {database.database.name}-{task.name}")
                    try:
                        sc_features = compute_same_class_features(
                            database.database.name, task.name,
                            split='train',
                        )
                        torch.save(sc_features, output_path)
                        logger.info(f"Successfully saved same-class features to {output_path}")
                    except Exception as exc:
                        logger.error(
                            f"Failed to compute same-class features for "
                            f"{database.database.name}-{task.name}: {exc}"
                        )
                else:
                    logger.info(f"Same-class features for {database.database.name}-{task.name} already exist, skipping")

        # Force garbage collection after each database
        clear_task_memory()

    logger.info("Completed processing all tasks")


# ============================================================================
# Argument Parser Setup
# ============================================================================

def _setup_argparser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the script.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Unified task embedding computation with multiple feature types"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="",
        help="Path to dataset configuration YAML file"
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="",
        help="Database name (used when --dataset-config is not provided). Leave blank to process all databases listed in dataset config."
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="",
        help="Task name (used when --dataset-config is not provided). Leave blank to process tasks from NEEDED_TASKS."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to custom dataset (if not using standard datasets)"
    )

    # Output configuration
    parser.add_argument(
        "--root-dir",
        type=str,
        default="taskemb",
        help="Root directory for saving task embeddings"
    )
    parser.add_argument(
        "--basic-config",
        type=str,
        default="configs/default/task.yaml",
        help="Path to basic configuration YAML file"
    )

    # Feature selection
    parser.add_argument(
        "--features",
        nargs='+',
        default=['homophily', 'temporal', 'hit', 'stats'],
        choices=['homophily', 'temporal', 'autotransfer', 'autotransfer_dfs', 'hit', 'stats', 'same_class', 'all'],
        help="Features to compute. Options: homophily (graph-based), temporal (time-aware), autotransfer, autotransfer_dfs, hit, stats, same_class, all"
    )

    # Additional options
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Force recomputation of existing task embeddings'
    )
    parser.add_argument(
        "--validated-results",
        type=str,
        default="./validated_search_results.csv",
        help="Path to validated runs CSV (for hit rate features)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    return parser


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    """Main execution function."""
    # Setup argument parser
    parser = _setup_argparser()
    args = parser.parse_args()

    # Handle 'all' features option
    if 'all' in args.features:
        args.features = ['homophily', 'temporal', 'autotransfer', 'autotransfer_dfs', 'hit', 'stats', 'same_class']

    # Setup logging
    logger.add("heuristics.log", rotation="1 MB")
    logger.info("=" * 80)
    logger.info("Starting unified task embedding computation")
    logger.info(f"Features to compute: {', '.join(args.features)}")
    logger.info("=" * 80)

    # Ensure output directory exists
    os.makedirs(args.root_dir, exist_ok=True)

    # Load basic configuration
    basic_config = load_yaml(args.basic_config)
    logger.info(f"Loaded basic config from {args.basic_config}")

    # Set random seed if provided
    if args.seed is not None:
        seed_everything(args.seed)
        logger.info(f"Set random seed to {args.seed}")
    elif 'seed' in basic_config:
        seed_everything(basic_config['seed'])
        logger.info(f"Set random seed from config: {basic_config['seed']}")

    # Set deterministic algorithms for reproducibility (needed for some features)
    if 'hit' in args.features:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        logger.info("Enabled deterministic algorithms for reproducibility")

    # Load or construct data configuration
    if args.dataset_config:
        cfg_path = args.dataset_config
    elif args.db_name and args.task_name:
        cfg_path = None
    else:
        cfg_path = DEFAULT_DATASET_CONFIG

    if cfg_path:
        data_config = load_yaml(cfg_path)
        logger.info(f"Loaded dataset config from {cfg_path}")
    else:
        data_config = {
            "dbs": [
                {
                    "db_name": args.db_name,
                    "task_name": [args.task_name],
                    "path": args.path
                }
            ]
        }
        logger.info(f"Using command-line config: db={args.db_name}, task={args.task_name}")

    # Disable wandb logging
    wandb.init(mode="disabled")

    # Execute task embedding computation
    task_embedding_execution(basic_config, data_config, args)
    logger.info("=" * 80)
    logger.info("Task embedding computation completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
