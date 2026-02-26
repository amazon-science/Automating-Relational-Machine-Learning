"""
Analysis module for computing the mean ratio of past nodes from the same task table
that share the same model class with the seed node in sampled sequences.

This module analyzes sequences sampled from relational databases to understand
the homogeneity of class labels within sequences. For classification tasks,
it computes statistics about how many past nodes from the same task table
share the same class label as the seed (target) node.

Usage:
    # Method 1: Analyze all tasks from a YAML config file
    python plot/same_class_analysis.py --config configs/pyg/hpo/all-nodes.yaml \
                                       --split train \
                                       --num_samples 1000

    # Method 2: Analyze all tasks from one database only
    python plot/same_class_analysis.py --config configs/pyg/hpo/all-nodes.yaml \
                                       --db_filter rel-f1 \
                                       --split train

    # Method 3: Analyze a specific task from config
    python plot/same_class_analysis.py --config configs/pyg/hpo/all-nodes.yaml \
                                       --db_filter rel-f1 \
                                       --task_filter driver-dnf \
                                       --split train

    # Method 4: Analyze a single task (legacy mode)
    python plot/same_class_analysis.py --task_id "rel-hm,user-churn,churn" \
                                       --split train \
                                       --num_samples 1000 \
                                       --empty_handling ignore

    # Analyze all nodes in test split for all tasks
    python plot/same_class_analysis.py --config configs/pyg/hpo/all-nodes.yaml \
                                       --split test \
                                       --num_samples -1 \
                                       --empty_handling zero

Config file:
    In addition to listing tasks under `dbs`, the config can optionally
    include an `analysis` block to provide defaults for script arguments:

    analysis:
      split: train
      num_samples: 1000
      batch_size: 32
      seq_len: 1024
      max_bfs_width: 256
      seed: 42
      use_rfm_preprocessing: false
      empty_handling: ignore
      embedding_model: all-MiniLM-L12-v2
      d_text: 384
      plot_per_sample: false
      output_dir: results/same_class_analysis
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from data.rt.data import RelationalDataset, get_column_index, _pre_root
from data.rt import tasks as task_registry
from utils.type import seed_everything
from utils.misc import load_yaml


DEFAULT_ANALYSIS_CONFIG = {
    'split': 'train',
    'num_samples': 1000,
    'batch_size': 32,
    'seq_len': 1024,
    'max_bfs_width': 256,
    'seed': 42,
    'use_rfm_preprocessing': False,
    'empty_handling': 'ignore',
    'embedding_model': 'all-MiniLM-L12-v2',
    'd_text': 384,
    'plot_per_sample': False,
    'output_dir': 'results/same_class_analysis'
}


# Known regression tasks (db, table) pairs
REGRESSION_TASKS = {
    ("rel-f1", "driver-position"),
    ("rel-f1", "driver-position-change"),
    ("rel-hm", "item-sales"),
    ("rel-trial", "site-success"),
    ("rel-trial", "study-adverse"),
    ("rel-event", "user-attendance"),
    ("rel-arxiv", "author-publication"),
    ("rel-avito", "ad-ctr"),
    ("rel-stack", "post-votes"),
    ("rel-stack", "user-comment-count"),
}


def _same_class_cache_path(root_dir: str, db_name: str, table_name: str, target_col: str, split: str) -> str:
    filename = f"{db_name}-{table_name}-{target_col}-{split}-same-class.pt"
    return os.path.join(root_dir, filename)


def _same_class_cache_complete(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        data = torch.load(path, map_location='cpu')
    except Exception as exc:
        logger.warning(f"Failed to load existing same-class cache from {path}: {exc}")
        return False

    task_type = data.get('task_type', 'classification') if isinstance(data, dict) else 'classification'
    if task_type == 'regression':
        required = {"mean_label_correlation", "sparsity_ratio"}
    else:
        required = {"mean_same_class_ratio_ignore", "sparsity_ratio"}
    missing = [k for k in required if k not in data]
    if missing:
        logger.info(f"Same-class cache at {path} missing keys: {missing}")
        return False
    return True


def _feature_payload_from_result(result: Dict) -> Dict:
    if result.get('task_type', 'classification') == 'classification' and result.get('empty_handling') != 'ignore':
        raise ValueError("Cache expects empty_handling='ignore' for classification")

    total_sequences = int(
        result.get('total_sequences_attempted', 0)
        or (result.get('num_sequences_analyzed', 0) + result.get('num_empty_sequences', 0))
    )
    sparsity_ratio = float(result.get('num_empty_sequences', 0) / total_sequences) if total_sequences else 0.0

    payload = {
        'db_name': result['db_name'],
        'table_name': result['table_name'],
        'target_col': result['target_col'],
        'split': result['split'],
        'task_type': result.get('task_type', 'classification'),
        'sparsity_ratio': sparsity_ratio,
        'num_sequences_total': total_sequences,
        'num_sequences_analyzed': int(result.get('num_sequences_analyzed', 0)),
        'num_empty_sequences': int(result.get('num_empty_sequences', 0)),
    }

    if payload['task_type'] == 'regression':
        payload.update(
            mean_label_correlation=float(result.get('mean_label_correlation', 0.0)),
            variance_label_correlation=float(result.get('variance_label_correlation', 0.0)),
            adjusted_mean_same_class_ratio=float(result.get('adjusted_mean_same_class_ratio', result.get('mean_label_correlation', 0.0))),
            mean_past_task_nodes=float(result.get('mean_past_task_nodes', 0.0)),
        )
    else:
        mean_ratio = float(result['mean_same_class_ratio'])
        payload.update(
            mean_same_class_ratio_ignore=mean_ratio,
            variance_same_class_ratio=float(result.get('std_same_class_ratio', 0.0)),
            baseline_label_ratio=float(result.get('baseline_label_ratio', 0.0)),
            adjusted_mean_same_class_ratio=float(result.get('adjusted_mean_same_class_ratio', mean_ratio)),
            mean_past_task_nodes=float(result.get('mean_past_task_nodes', 0.0)),
        )

    return payload


class SameClassAnalyzer:
    """
    Analyzer for computing same-class ratios in sampled sequences.

    This class samples sequences from relational databases and computes statistics
    about how many past nodes from the same task table share the same class label
    as the seed node.
    """

    def __init__(
        self,
        task_tuple: Tuple[str, str, str],
        split: str = "train",
        num_samples: int = 1000,
        batch_size: int = 32,
        seq_len: int = 512,
        max_bfs_width: int = 100,
        seed: int = 42,
        use_rfm_preprocessing: bool = False,
        empty_handling: str = "ignore",
        embedding_model: str = "e5-mistral",
        d_text: int = 4096,
        task_type: str = "classification",
    ):
        """
        Initialize the analyzer.

        Args:
            task_tuple: Tuple of (db_name, table_name, target_col) e.g., ('rel-hm', 'user-churn', 'churn')
            split: Split to analyze ('train', 'val', or 'test')
            num_samples: Number of sequences to sample. Use -1 to sample all nodes in the split
            batch_size: Batch size for dataset
            seq_len: Sequence length
            max_bfs_width: Maximum BFS width for sampling
            seed: Random seed
            use_rfm_preprocessing: Whether to use RFM preprocessing
            empty_handling: How to handle sequences with no past task nodes.
                           Options: 'ignore' (exclude from statistics) or
                                   'zero' (count as 0 ratio)
            embedding_model: Text embedding model name
            d_text: Text embedding dimension
        """
        if len(task_tuple) == 4:
            self.db_name, self.table_name, self.target_col, inferred_type = task_tuple
            task_type = inferred_type
        else:
            self.db_name, self.table_name, self.target_col = task_tuple
        self.task_tuple = task_tuple
        self.split = split
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_bfs_width = max_bfs_width
        self.seed = seed
        self.use_rfm_preprocessing = use_rfm_preprocessing
        self.empty_handling = empty_handling
        self.embedding_model = embedding_model
        self.d_text = d_text
        self.task_type = task_type

        if empty_handling not in ['ignore', 'zero']:
            raise ValueError(f"empty_handling must be 'ignore' or 'zero', got {empty_handling}")

        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        seed_everything(seed)

    def _analyze_batch_classification(
        self,
        batch: Dict[str, torch.Tensor],
        target_col_idx: int
    ) -> Tuple[List[Tuple[Optional[float], int, int]], List[float]]:
        """Analyze a batch for classification tasks and return per-sequence ratios and seed labels."""
        # Extract batch data
        # node_idxs, is_targets, is_task_nodes have shape [batch_size, seq_len]
        # boolean_values has shape [batch_size, seq_len, 1] - contains labels for task nodes
        # col_name_idxs has shape [batch_size, seq_len] - indicates which column each position represents
        node_idxs = batch['node_idxs']
        is_targets = batch['is_targets']
        is_task_nodes = batch['is_task_nodes']
        col_name_idxs = batch['col_name_idxs']
        boolean_values = batch['boolean_values'].squeeze(-1)  # Shape: [batch_size, seq_len]

        batch_size = node_idxs.shape[0]

        # Find seed node for each sequence using the same logic as in rt.py
        # Shape: [batch_size, 1]
        seed_node_idxs = node_idxs.masked_fill(~is_targets, -1).max(dim=1, keepdim=True).values

        # CRITICAL: Filter to only positions corresponding to the target column
        # is_task_nodes includes ALL columns from task tables, but we only want the target column
        # See rustler/src/fly.rs:877 - is_task_nodes marks task table positions OR target column
        is_target_column = (col_name_idxs == target_col_idx)

        # Determine which nodes are seed nodes and which are other task nodes with target column
        # Shape: [batch_size, seq_len]
        is_seed_node = (node_idxs == seed_node_idxs)
        is_other_task_node = is_task_nodes & ~is_seed_node & is_target_column

        results = []
        seed_label_list: List[float] = []
        for seq_idx in range(batch_size):
            seed_node_idx = seed_node_idxs[seq_idx, 0].item()

            if seed_node_idx == -1:
                # No target node found
                results.append((None, 0, 0))
                continue

            # Get seed node label at the TARGET COLUMN position
            # CRITICAL: The seed node appears multiple times with different columns
            # We need the label specifically at the target column position
            is_seed_target = is_seed_node[seq_idx] & is_target_column[seq_idx]
            seed_labels = boolean_values[seq_idx][is_seed_target]


            if len(seed_labels) == 0:
                results.append((None, 0, 0))
                continue

            # Get the label (0 or 1) for the seed node at target column
            seed_label = seed_labels[0].item()
            seed_label_list.append(float(seed_label))

            # Get other task nodes (positions where is_other_task_node is True)
            # These are already filtered to be task nodes only
            is_other = is_other_task_node[seq_idx]

            # Get unique node IDs and their labels for other task nodes
            # Same node can appear multiple times, so we need to deduplicate
            # IMPORTANT: boolean_values at these positions contain the actual labels
            other_node_ids = node_idxs[seq_idx][is_other]
            other_node_labels = boolean_values[seq_idx][is_other]

            if len(other_node_ids) == 0:
                # No past task nodes
                results.append((None, 0, 0))
                continue

            # Deduplicate by node ID - keep first occurrence of each unique node
            unique_nodes = {}
            for i in range(len(other_node_ids)):
                node_id = other_node_ids[i].item()
                if node_id not in unique_nodes:
                    unique_nodes[node_id] = other_node_labels[i].item()

            # Count how many unique nodes share the same label as seed
            unique_labels = list(unique_nodes.values())
            num_same_class = sum(1 for label in unique_labels if abs(label - seed_label) < 0.01)
            num_past_task_nodes = len(unique_labels)

            # Compute ratio
            ratio = num_same_class / num_past_task_nodes
            results.append((ratio, num_past_task_nodes, num_same_class))

        return results, seed_label_list

    def _analyze_batch_regression(
        self,
        batch: Dict[str, torch.Tensor],
        target_col_idx: int
    ) -> Tuple[List[Tuple[float, float]], List[int]]:
        """Collect (seed, other) label pairs for regression tasks."""
        node_idxs = batch['node_idxs']
        is_targets = batch['is_targets']
        is_task_nodes = batch['is_task_nodes']
        col_name_idxs = batch['col_name_idxs']
        number_values = batch['number_values'].squeeze(-1)  # [batch, seq_len]

        batch_size = node_idxs.shape[0]
        seed_node_idxs = node_idxs.masked_fill(~is_targets, -1).max(dim=1, keepdim=True).values
        is_seed_node = (node_idxs == seed_node_idxs)
        is_target_column = (col_name_idxs == target_col_idx)

        pairs: List[Tuple[float, float]] = []
        past_counts: List[int] = []

        # Debug: Check first batch
        if len(pairs) == 0 and batch_size > 0:
            logger.debug(f"First batch - target_col_idx: {target_col_idx}")
            logger.debug(f"  col_name_idxs unique values: {torch.unique(col_name_idxs)[:20].tolist()}")
            logger.debug(f"  is_target_column sum: {is_target_column.sum().item()}")
            logger.debug(f"  number_values where is_target_column (first 20): {number_values[is_target_column][:20].tolist()}")

        for seq_idx in range(batch_size):
            seed_node_idx = seed_node_idxs[seq_idx, 0].item()
            if seed_node_idx == -1:
                continue

            is_seed_target = is_seed_node[seq_idx] & is_target_column[seq_idx]
            seed_labels = number_values[seq_idx][is_seed_target]
            seed_labels = seed_labels[torch.isfinite(seed_labels)]
            if len(seed_labels) == 0:
                continue
            seed_label = float(seed_labels[0].item())

            is_other_task_node = is_task_nodes & ~is_seed_node & is_target_column
            is_other = is_other_task_node[seq_idx]
            other_node_ids = node_idxs[seq_idx][is_other]
            other_labels = number_values[seq_idx][is_other]
            mask_valid = torch.isfinite(other_labels)
            other_node_ids = other_node_ids[mask_valid]
            other_labels = other_labels[mask_valid]
            if len(other_node_ids) == 0:
                past_counts.append(0)
                continue

            unique_nodes = {}
            for i in range(len(other_node_ids)):
                node_id = other_node_ids[i].item()
                if node_id not in unique_nodes:
                    unique_nodes[node_id] = float(other_labels[i].item())

            labels = list(unique_nodes.values())
            past_counts.append(len(labels))
            for lbl in labels:
                pairs.append((seed_label, lbl))

        return pairs, past_counts

    def analyze(self) -> Dict:
        """
        Analyze the task to compute same-class statistics.

        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing: {self.db_name} - {self.table_name}.{self.target_col} (split: {self.split})")

        # Prepare task tuple for dataset
        columns_to_drop = []  # No columns to drop for analysis
        tasks = [(self.db_name, self.table_name, self.target_col, self.split, columns_to_drop)]

        dataset = RelationalDataset(
            tasks=tasks,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            rank=0,
            world_size=1,
            max_bfs_width=self.max_bfs_width,
            embedding_model=self.embedding_model,
            d_text=self.d_text,
            seed=self.seed,
            use_rfm_preprocessing=self.use_rfm_preprocessing
        )

        # Get the target column index from the dataset
        # This is needed to filter positions to only the target column
        target_col_idx = get_column_index(
            column_name=self.target_col,
            table_name=self.table_name,
            db_name=self.db_name,
            use_rfm_preprocessing=self.use_rfm_preprocessing
        )
        logger.info(f"Target column index: {target_col_idx}")

        # Determine actual number of samples
        total_batches = len(dataset)
        if self.num_samples == -1:
            num_batches = total_batches
            logger.info(f"num_samples=-1: analyzing all {total_batches} batches")
        else:
            num_batches = min((self.num_samples + self.batch_size - 1) // self.batch_size, total_batches)
            logger.info(f"Analyzing {num_batches} batches (up to {num_batches * self.batch_size} sequences)")

        # Sample sequences and compute statistics
        ratios: List[float] = []
        past_node_counts: List[int] = []
        same_class_counts: List[int] = []
        seed_labels_all: List[float] = []
        num_empty_sequences = 0

        reg_pairs: List[Tuple[float, float]] = []

        sequences_processed = 0
        for batch_idx in tqdm(range(num_batches), desc=f"Processing {self.table_name}.{self.target_col}"):
            try:
                batch = dataset[batch_idx]
                true_batch_size = batch.get('true_batch_size', self.batch_size)

                if self.task_type == 'regression':
                    batch_pairs, batch_past_counts = self._analyze_batch_regression(batch, target_col_idx)
                    reg_pairs.extend(batch_pairs)
                    past_node_counts.extend(batch_past_counts)
                    num_empty_sequences += sum(1 for c in batch_past_counts if c == 0)
                    sequences_processed += true_batch_size
                else:
                    batch_results, batch_seed_labels = self._analyze_batch_classification(batch, target_col_idx)

                    for seq_idx in range(true_batch_size):
                        if self.num_samples != -1 and sequences_processed >= self.num_samples:
                            break

                        ratio, num_past, num_same = batch_results[seq_idx]

                        if ratio is None:
                            num_empty_sequences += 1
                            if self.empty_handling == 'zero':
                                ratios.append(0.0)
                                past_node_counts.append(0)
                                same_class_counts.append(0)
                        else:
                            ratios.append(ratio)
                            past_node_counts.append(num_past)
                            same_class_counts.append(num_same)
                            seed_label = batch_seed_labels[seq_idx] if seq_idx < len(batch_seed_labels) else None
                            if seed_label is not None:
                                seed_labels_all.append(seed_label)
                        sequences_processed += 1

                if self.num_samples != -1 and sequences_processed >= self.num_samples:
                    break

            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue

        if self.task_type == 'regression':
            if len(reg_pairs) < 2:
                logger.warning(f"No valid regression pairs found for {self.table_name}.{self.target_col}. Using default correlation of 0.0.")
                # Return default values instead of error when no valid sequences
                return {
                    'db_name': self.db_name,
                    'table_name': self.table_name,
                    'target_col': self.target_col,
                    'split': self.split,
                    'task_type': 'regression',
                    'empty_handling': self.empty_handling,
                    'num_sequences_analyzed': sequences_processed,
                    'num_empty_sequences': num_empty_sequences,
                    'total_sequences_attempted': sequences_processed,
                    'mean_label_correlation': 0.0,  # Default value for regression
                    'adjusted_mean_same_class_ratio': 0.0,  # Default value for regression
                    'variance_label_correlation': 0.0,
                    'mean_past_task_nodes': 0.0,
                    'std_past_task_nodes': 0.0,
                    'median_past_task_nodes': 0.0,
                    'sparsity_ratio': float(num_empty_sequences / sequences_processed) if sequences_processed else 1.0
                }

            seeds, others = zip(*reg_pairs)

            # Debug logging
            seeds_array = np.array(seeds)
            others_array = np.array(others)
            logger.info(f"DEBUG {self.table_name}.{self.target_col}: {len(reg_pairs)} pairs")
            logger.info(f"  Seeds - unique: {len(np.unique(seeds_array))}, mean: {seeds_array.mean():.3f}, std: {seeds_array.std():.3f}, min: {seeds_array.min():.3f}, max: {seeds_array.max():.3f}")
            logger.info(f"  Others - unique: {len(np.unique(others_array))}, mean: {others_array.mean():.3f}, std: {others_array.std():.3f}, min: {others_array.min():.3f}, max: {others_array.max():.3f}")
            if seeds_array.std() == 0:
                logger.warning(f"  Seeds array has zero variance! All values are {seeds_array[0]:.3f}")
            if others_array.std() == 0:
                logger.warning(f"  Others array has zero variance! All values are {others_array[0]:.3f}")

            corr = np.corrcoef(seeds, others)[0, 1] if len(reg_pairs) > 1 else 0.0
            results = {
                'db_name': self.db_name,
                'table_name': self.table_name,
                'target_col': self.target_col,
                'split': self.split,
                'task_type': 'regression',
                'empty_handling': self.empty_handling,
                'num_sequences_analyzed': sequences_processed,
                'num_empty_sequences': num_empty_sequences,
                'total_sequences_attempted': sequences_processed,
                'mean_label_correlation': float(corr),
                'adjusted_mean_same_class_ratio': float(corr),
                'variance_label_correlation': 0.0,
                'mean_past_task_nodes': float(np.mean(past_node_counts)) if past_node_counts else 0.0,
                'std_past_task_nodes': float(np.std(past_node_counts)) if past_node_counts else 0.0,
                'median_past_task_nodes': float(np.median(past_node_counts)) if past_node_counts else 0.0,
                'sparsity_ratio': float(num_empty_sequences / sequences_processed) if sequences_processed else 0.0,
            }
            return results

        if len(ratios) == 0:
            logger.warning(f"No valid sequences found for {self.table_name}.{self.target_col}. Using default value of 1.0.")
            # Return default values instead of error when no valid sequences
            return {
                'db_name': self.db_name,
                'table_name': self.table_name,
                'target_col': self.target_col,
                'split': self.split,
                'task_type': 'classification',
                'empty_handling': self.empty_handling,
                'num_sequences_analyzed': sequences_processed,
                'num_empty_sequences': num_empty_sequences,
                'total_sequences_attempted': sequences_processed,
                'mean_same_class_ratio': 1.0,  # Default value
                'variance_same_class_ratio': 0.0,
                'baseline_label_ratio': 0.0,
                'adjusted_mean_same_class_ratio': 1.0,  # Default value
                'mean_past_task_nodes': 0.0,
                'std_past_task_nodes': 0.0,
                'median_past_task_nodes': 0.0,
                'sparsity_ratio': float(num_empty_sequences / sequences_processed) if sequences_processed else 1.0
            }

        baseline_label_ratio = float(np.mean(seed_labels_all)) if seed_labels_all else 0.0
        adjusted_ratios = [r - baseline_label_ratio for r in ratios]
        results = {
            'db_name': self.db_name,
            'table_name': self.table_name,
            'target_col': self.target_col,
            'split': self.split,
            'task_type': 'classification',
            'empty_handling': self.empty_handling,
            'num_sequences_analyzed': len(ratios),
            'num_empty_sequences': num_empty_sequences,
            'total_sequences_attempted': sequences_processed,
            'mean_same_class_ratio': float(np.mean(ratios)),
            'std_same_class_ratio': float(np.std(ratios)),
            'median_same_class_ratio': float(np.median(ratios)),
            'min_same_class_ratio': float(np.min(ratios)),
            'max_same_class_ratio': float(np.max(ratios)),
            'mean_past_task_nodes': float(np.mean(past_node_counts)),
            'std_past_task_nodes': float(np.std(past_node_counts)),
            'median_past_task_nodes': float(np.median(past_node_counts)),
            'baseline_label_ratio': baseline_label_ratio,
            'adjusted_mean_same_class_ratio': float(np.mean(adjusted_ratios)) if adjusted_ratios else 0.0,
            'sparsity_ratio': float(num_empty_sequences / sequences_processed) if sequences_processed else 0.0,
            'ratios': ratios,
            'adjusted_ratios': adjusted_ratios,
            'past_node_counts': past_node_counts,
            'same_class_counts': same_class_counts
        }

        logger.info(f"Results for {self.table_name}.{self.target_col} ({self.split}):")
        logger.info(f"  Mean same-class ratio: {results['mean_same_class_ratio']:.4f} ± {results['std_same_class_ratio']:.4f}")
        logger.info(f"  Median same-class ratio: {results['median_same_class_ratio']:.4f}")
        logger.info(f"  Mean past task nodes: {results['mean_past_task_nodes']:.2f} ± {results['std_past_task_nodes']:.2f}")
        logger.info(f"  Sequences analyzed: {results['num_sequences_analyzed']}")
        logger.info(f"  Empty sequences: {results['num_empty_sequences']}")

        return results

    def save_results(self, result: Dict, output_dir: str):
        """
        Save analysis results to disk.

        Args:
            result: Result dictionary
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        if 'error' in result:
            logger.error(f"Cannot save results due to error: {result['error']}")
            return

        # Save summary CSV
        summary_data = {
            'db_name': result['db_name'],
            'table_name': result['table_name'],
            'target_col': result['target_col'],
            'split': result['split'],
            'empty_handling': result['empty_handling'],
            'num_sequences_analyzed': result['num_sequences_analyzed'],
            'num_empty_sequences': result['num_empty_sequences'],
            'total_sequences_attempted': result.get('total_sequences_attempted', 0),
            'sparsity_ratio': result.get('sparsity_ratio', 0.0),
            'mean_same_class_ratio': result['mean_same_class_ratio'],
            'std_same_class_ratio': result['std_same_class_ratio'],
            'median_same_class_ratio': result['median_same_class_ratio'],
            'mean_past_task_nodes': result['mean_past_task_nodes'],
            'std_past_task_nodes': result['std_past_task_nodes'],
        }

        df = pd.DataFrame([summary_data])
        csv_path = os.path.join(
            output_dir,
            f"{result['db_name']}_{result['table_name']}_{result['target_col']}_{result['split']}_{self.empty_handling}.csv"
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary to {csv_path}")

        # Save full results (including distributions)
        torch_path = os.path.join(
            output_dir,
            f"{result['db_name']}_{result['table_name']}_{result['target_col']}_{result['split']}_{self.empty_handling}.pt"
        )
        torch.save(result, torch_path)
        logger.info(f"Saved full results to {torch_path}")

    def create_visualizations(self, result: Dict, output_dir: str, plot_per_sample: bool = False):
        """
        Create visualization plots for the analysis results.

        Args:
            result: Result dictionary
            output_dir: Directory to save plots
            plot_per_sample: Whether to plot per-sample similarity distribution
        """
        if 'error' in result:
            logger.warning("Cannot create visualizations due to error in result")
            return

        # Set style
        sns.set_style("whitegrid")

        ratios = result['ratios']
        past_node_counts = result['past_node_counts']

        # Plot 1: Histogram of same-class ratios with statistics
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(ratios, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_xlabel('Same-Class Ratio', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        title = f"Same-Class Ratio Distribution\n{result['db_name']} - {result['table_name']}.{result['target_col']} ({result['split']})"
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add mean and median lines
        mean_val = result['mean_same_class_ratio']
        median_val = result['median_same_class_ratio']
        ax.axvline(x=mean_val, color='r', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.3f}')
        ax.axvline(x=median_val, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.3f}')
        ax.axvline(x=0.5, color='g', linestyle='--', alpha=0.5, linewidth=2,
                  label='Random baseline (0.5)')
        ax.legend(fontsize=10)

        plt.tight_layout()
        hist_path = os.path.join(
            output_dir,
            f"{result['db_name']}_{result['table_name']}_{result['target_col']}_{result['split']}_histogram_{self.empty_handling}.png"
        )
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved histogram to {hist_path}")
        plt.close()

        # Plot 2: Scatter plot of ratio vs number of past task nodes
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(past_node_counts, ratios, s=20, alpha=0.5, color='steelblue')

        ax.set_xlabel('Number of Past Task Nodes', fontsize=12)
        ax.set_ylabel('Same-Class Ratio', fontsize=12)
        title = f"Same-Class Ratio vs Past Task Nodes\n{result['db_name']} - {result['table_name']}.{result['target_col']} ({result['split']})"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, linewidth=2, label='Random baseline')
        ax.legend()

        plt.tight_layout()
        scatter_path = os.path.join(
            output_dir,
            f"{result['db_name']}_{result['table_name']}_{result['target_col']}_{result['split']}_scatter_{self.empty_handling}.png"
        )
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {scatter_path}")
        plt.close()

        # Plot 3 (optional): Per-sample similarity rate with average overlay
        if plot_per_sample:
            fig, ax = plt.subplots(figsize=(14, 6))

            # Plot individual ratios
            sample_indices = np.arange(len(ratios))
            ax.plot(sample_indices, ratios, 'o', markersize=2, alpha=0.3, color='steelblue', label='Per-sample ratio')

            # Calculate and plot rolling average
            window_size = max(10, len(ratios) // 50)  # Adaptive window size
            if len(ratios) >= window_size:
                rolling_avg = pd.Series(ratios).rolling(window=window_size, center=True).mean()
                ax.plot(sample_indices, rolling_avg, '-', linewidth=2, color='red',
                       label=f'Rolling average (window={window_size})')

            # Plot overall mean
            ax.axhline(y=mean_val, color='darkred', linestyle='--', linewidth=2,
                      label=f'Overall mean: {mean_val:.3f}')
            ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, linewidth=2,
                      label='Random baseline (0.5)')

            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel('Same-Class Ratio', fontsize=12)
            title = f"Per-Sample Same-Class Ratio\n{result['db_name']} - {result['table_name']}.{result['target_col']} ({result['split']})"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            per_sample_path = os.path.join(
                output_dir,
                f"{result['db_name']}_{result['table_name']}_{result['target_col']}_{result['split']}_per_sample_{self.empty_handling}.png"
            )
            plt.savefig(per_sample_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved per-sample plot to {per_sample_path}")
            plt.close()


def parse_tasks_from_config(config: Dict) -> List[Tuple[str, str, str, str]]:
    """Parse task tuples from config dictionary.

    Uses the task registry in ``data.rt.tasks`` to find the target column for each
    (db, table) pair. Falls back to splitting ``table-target`` if a registry entry
    is missing. Returns tuples of (db, table, target, task_type).
    """

    registry_map: Dict[Tuple[str, str], str] = {}
    for entry in getattr(task_registry, "all_tasks", []):
        if len(entry) >= 3:
            db, table, target = entry[:3]
            registry_map[(db, table)] = target

    manual_overrides: Dict[Tuple[str, str], Tuple[str, str]] = {
        # config task_name -> (table_name, target_col)
        ("rel-avs", "repeater"): ("history", "repeater"),
        ("rel-f1", "constructor-scores-points"): ("constructor-scores-points", "scored_points"),
        ("rel-f1", "driver-position-change"): ("driver-position-change", "position_change"),
        ("rel-ieeecis", "transaction-fraud"): ("transaction-fraud", "isFraud"),
        ("rel-stack", "user-comment-count"): ("user-comment-count", "comment_count"),
        ("rel-arxiv", "paper-citation"): ("paper-citation", "cited"),
        ("rel-arxiv", "author-publication"): ("author-publication", "publication_count"),
    }

    tasks = []
    for db_config in config.get('dbs', []):
        db_name = db_config['db_name']
        task_names = db_config.get('task_name', [])

        # Handle both list and single string formats
        if isinstance(task_names, str):
            task_names = [task_names]

        for task_name in task_names:
            table_name = task_name
            target_col = registry_map.get((db_name, table_name))

            task_type = 'regression' if (db_name, table_name) in REGRESSION_TASKS else 'classification'

            if (db_name, task_name) in manual_overrides:
                table_name, target_col = manual_overrides[(db_name, task_name)]
                task_type = 'regression' if (db_name, table_name) in REGRESSION_TASKS else 'classification'
                logger.info(
                    f"Using manual override for task '{task_name}' -> table '{table_name}', target '{target_col}'."
                )

            if target_col is None:
                if '-' in task_name:
                    # For tasks not in registry, the full task name is the table name in table_info
                    # Try to infer target column from the task name by converting dashes to underscores
                    # and using the part after the first dash as a hint
                    _, target_hint = task_name.split('-', 1)
                    target_col = target_hint.replace('-', '_')
                    # Keep table_name as the full task_name since that's what's in table_info
                    table_name = task_name
                    task_type = 'regression' if (db_name, table_name) in REGRESSION_TASKS else 'classification'
                    logger.warning(
                        f"Task '{task_name}' not found in registry; using table name '{table_name}' and inferring target '{target_col}'."
                    )
                else:
                    logger.warning(
                        f"Task '{task_name}' not found in registry and cannot infer target. Skipping."
                    )
                    continue

            tasks.append((db_name, table_name, target_col, task_type))

    return tasks


def _table_info_available(db_name: str, table_name: str, split: str, use_rfm_preprocessing: bool) -> bool:
    suffix = "_rfm" if use_rfm_preprocessing else ""
    path = _pre_root() / db_name / f"table_info{suffix}.json"
    if not path.exists():
        logger.warning(f"Missing table_info at {path}; skipping {db_name}.{table_name} ({split}).")
        return False
    try:
        info = load_yaml(path)
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning(f"Failed to load table_info at {path}: {exc}; skipping {db_name}.{table_name} ({split}).")
        return False

    key_db = f"{table_name}:Db"
    key_split = f"{table_name}:{split.title()}"
    if key_db in info or key_split in info:
        return True

    logger.warning(
        f"No entry for table '{table_name}' in table_info (looked for '{key_db}'/'{key_split}'); "
        f"skipping {db_name}.{table_name} ({split})."
    )
    return False


def resolve_analysis_params(args: argparse.Namespace, config: Dict) -> Dict:
    """
    Resolve analysis parameters using the YAML config (if provided) with CLI overrides.
    """
    config_block = (config or {}).get('analysis', {})

    params = DEFAULT_ANALYSIS_CONFIG.copy()
    params.update({k: config_block.get(k, v) for k, v in DEFAULT_ANALYSIS_CONFIG.items()})

    overrides = {
        'split': args.split,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'max_bfs_width': args.max_bfs_width,
        'seed': args.seed,
        'use_rfm_preprocessing': args.use_rfm_preprocessing,
        'empty_handling': args.empty_handling,
        'embedding_model': args.embedding_model,
        'd_text': args.d_text,
        'plot_per_sample': args.plot_per_sample,
        'output_dir': args.output_dir
    }

    for key, value in overrides.items():
        if value is not None:
            params[key] = value

    return params


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze same-class ratios in sampled relational sequences. "
                    "Use either --config to analyze multiple tasks from a YAML file, "
                    "or --task_id for a single task."
    )

    # Config file or single task
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--config",
        type=str,
        default="configs/pyg/hpo/all-nodes.yaml",
        help="Path to YAML config file (e.g., configs/pyg/hpo/all-nodes.yaml)"
    )
    task_group.add_argument(
        "--task_id",
        type=str,
        help="Single task identifier in format 'db_name,table_name,target_col' (e.g., 'rel-hm,user-churn,churn')"
    )

    # Analysis parameters
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Split to analyze (train, val, or test)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of sequences to sample. Use -1 to analyze all nodes in the split"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for sampling"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length"
    )
    parser.add_argument(
        "--max_bfs_width",
        type=int,
        default=None,
        help="Maximum BFS width for sampling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--use_rfm_preprocessing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use RFM preprocessing"
    )
    parser.add_argument(
        "--empty_handling",
        type=str,
        default=None,
        choices=["ignore", "zero"],
        help="How to handle sequences with no past task nodes: 'ignore' (exclude) or 'zero' (count as 0)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Text embedding model name"
    )
    parser.add_argument(
        "--d_text",
        type=int,
        default=None,
        help="Text embedding dimension"
    )
    parser.add_argument(
        "--plot_per_sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Create per-sample similarity rate visualization with rolling average"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="taskemb",
        help="Directory to save cached same-class features (e.g., taskemb)"
    )
    parser.add_argument(
        "--refresh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute analysis and overwrite cache even if it exists"
    )
    parser.add_argument(
        "--db_filter",
        type=str,
        default=None,
        help="When using --config, filter to only this database name (e.g., 'rel-f1')"
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default=None,
        help="When using --config, filter to only this task name (e.g., 'driver-dnf')"
    )

    args = parser.parse_args()

    config = load_yaml(args.config) if args.config else {}

    # Parse tasks
    tasks = []
    if args.config:
        # Load config and parse tasks
        logger.info(f"Loading config from {args.config}")
        all_tasks = parse_tasks_from_config(config)

        # Apply filters if specified
        for entry in all_tasks:
            if len(entry) == 4:
                db_name, table_name, target_col, task_type = entry
            else:
                db_name, table_name, target_col = entry
                task_type = 'regression' if (db_name, table_name) in REGRESSION_TASKS else 'classification'

            if args.db_filter and db_name != args.db_filter:
                continue
            task_name = f"{table_name}-{target_col}"
            if args.task_filter and task_name != args.task_filter:
                continue
            tasks.append((db_name, table_name, target_col, task_type))

        if not tasks:
            logger.error("No tasks matched the filters. Check --db_filter and --task_filter.")
            return

        logger.info(f"Found {len(tasks)} task(s) to analyze")
    else:
        # Parse single task_id
        try:
            task_parts = args.task_id.split(',')
            if len(task_parts) != 3:
                raise ValueError("task_id must have exactly 3 parts: db_name,table_name,target_col")
            db_name, table_name, target_col = task_parts
            task_type = 'regression' if (db_name, table_name) in REGRESSION_TASKS else 'classification'
            tasks = [(db_name, table_name, target_col, task_type)]
        except Exception as e:
            parser.error(f"Invalid task_id format: {e}")

    analysis_params = resolve_analysis_params(args, config)

    # Cache directory for task-level features
    os.makedirs(args.root_dir, exist_ok=True)

    # Setup logging
    os.makedirs(analysis_params['output_dir'], exist_ok=True)
    logger.add(
        os.path.join(analysis_params['output_dir'], "same_class_analysis.log"),
        rotation="1 MB"
    )

    logger.info("=" * 80)
    logger.info("Starting Same-Class Ratio Analysis")
    logger.info("=" * 80)
    logger.info(f"Number of tasks: {len(tasks)}")
    logger.info(f"Split: {analysis_params['split']}")
    logger.info(f"Number of samples per task: {analysis_params['num_samples']} (-1 = all nodes)")
    logger.info(f"Empty sequence handling: {analysis_params['empty_handling']}")
    logger.info(f"Plot per-sample: {analysis_params['plot_per_sample']}")
    logger.info(f"Output directory: {analysis_params['output_dir']}")

    # Process each task
    all_results = []
    for task_idx, task_tuple in enumerate(tasks):
        if len(task_tuple) == 4:
            db_name, table_name, target_col, task_type = task_tuple
        else:
            db_name, table_name, target_col = task_tuple
            task_type = 'regression' if (db_name, table_name) in REGRESSION_TASKS else 'classification'
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Task {task_idx + 1}/{len(tasks)}: {db_name} - {table_name}.{target_col}")
        logger.info("=" * 80)

        feature_cache_path = _same_class_cache_path(
            args.root_dir,
            db_name,
            table_name,
            target_col,
            analysis_params['split']
        )

        if not _table_info_available(db_name, table_name, analysis_params['split'], analysis_params['use_rfm_preprocessing']):
            all_results.append({
                'db_name': db_name,
                'table_name': table_name,
                'target_col': target_col,
                'split': analysis_params['split'],
                'error': 'missing table_info entry',
            })
            continue

        if not args.refresh and _same_class_cache_complete(feature_cache_path):
            try:
                cached_payload = torch.load(feature_cache_path, map_location='cpu')
            except Exception as exc:
                logger.warning(
                    f"Cache exists at {feature_cache_path} but failed to load ({exc}); recomputing."
                )
            else:
                if cached_payload.get('task_type', 'classification') == 'regression':
                    logger.info(
                        f"Cache hit for {db_name}-{table_name}-{target_col} ({analysis_params['split']}). "
                        f"corr={cached_payload['mean_label_correlation']:.4f}, "
                        f"sparsity={cached_payload['sparsity_ratio']:.4f}"
                    )
                    all_results.append({
                        'db_name': db_name,
                        'table_name': table_name,
                        'target_col': target_col,
                        'split': analysis_params['split'],
                        'task_type': 'regression',
                        'mean_label_correlation': cached_payload['mean_label_correlation'],
                        'num_sequences_analyzed': cached_payload.get('num_sequences_analyzed', 0),
                        'num_empty_sequences': cached_payload.get('num_empty_sequences', 0),
                        'sparsity_ratio': cached_payload.get('sparsity_ratio', 0.0),
                        'cached': True,
                    })
                else:
                    logger.info(
                        f"Cache hit for {db_name}-{table_name}-{target_col} ({analysis_params['split']}). "
                        f"mean_same_class_ratio(ignore)={cached_payload['mean_same_class_ratio_ignore']:.4f}, "
                        f"sparsity={cached_payload['sparsity_ratio']:.4f}"
                    )
                    all_results.append({
                        'db_name': db_name,
                        'table_name': table_name,
                        'target_col': target_col,
                        'split': analysis_params['split'],
                        'empty_handling': 'ignore',
                        'task_type': 'classification',
                        'mean_same_class_ratio': cached_payload['mean_same_class_ratio_ignore'],
                        'num_sequences_analyzed': cached_payload.get('num_sequences_analyzed', 0),
                        'num_empty_sequences': cached_payload.get('num_empty_sequences', 0),
                        'sparsity_ratio': cached_payload.get('sparsity_ratio', 0.0),
                        'cached': True,
                    })
                continue

        # Create analyzer
        analyzer = SameClassAnalyzer(
            task_tuple=task_tuple,
            split=analysis_params['split'],
            num_samples=analysis_params['num_samples'],
            batch_size=analysis_params['batch_size'],
            seq_len=analysis_params['seq_len'],
            max_bfs_width=analysis_params['max_bfs_width'],
            seed=analysis_params['seed'],
            use_rfm_preprocessing=analysis_params['use_rfm_preprocessing'],
            empty_handling=analysis_params['empty_handling'],
            embedding_model=analysis_params['embedding_model'],
            d_text=analysis_params['d_text'],
            task_type=task_type
        )

        # Run analysis
        try:
            result = analyzer.analyze()
            all_results.append(result)

            # Save results and create visualizations
            if 'error' not in result:
                if result.get('task_type', 'classification') == 'classification':
                    analyzer.save_results(result, analysis_params['output_dir'])
                    analyzer.create_visualizations(
                        result,
                        analysis_params['output_dir'],
                        plot_per_sample=analysis_params['plot_per_sample']
                    )

                feature_result = result
                if feature_result.get('task_type', 'classification') == 'classification' and feature_result.get('empty_handling') != 'ignore':
                    logger.info("Computing additional run with empty_handling='ignore' for cache")
                    ignore_analyzer = SameClassAnalyzer(
                        task_tuple=task_tuple,
                        split=analysis_params['split'],
                        num_samples=analysis_params['num_samples'],
                        batch_size=analysis_params['batch_size'],
                        seq_len=analysis_params['seq_len'],
                        max_bfs_width=analysis_params['max_bfs_width'],
                        seed=analysis_params['seed'],
                        use_rfm_preprocessing=analysis_params['use_rfm_preprocessing'],
                        empty_handling='ignore',
                        embedding_model=analysis_params['embedding_model'],
                        d_text=analysis_params['d_text'],
                        task_type='classification'
                    )
                    feature_result = ignore_analyzer.analyze()

                if 'error' not in feature_result:
                    try:
                        payload = _feature_payload_from_result(feature_result)
                        torch.save(payload, feature_cache_path)
                        logger.info(f"Saved same-class features to {feature_cache_path}")
                    except Exception as exc:
                        logger.error(f"Failed to save same-class cache to {feature_cache_path}: {exc}")
                else:
                    logger.error("Skipping cache save due to error in ignore-handling analysis")

                logger.info("Task analysis complete:")
                if result.get('task_type', 'classification') == 'regression':
                    logger.info(f"  Mean label correlation: {result['mean_label_correlation']:.4f}")
                else:
                    logger.info(f"  Mean same-class ratio: {result['mean_same_class_ratio']:.4f}")
                logger.info(f"  Sequences analyzed: {result['num_sequences_analyzed']}")
                logger.info(f"  Empty sequences: {result['num_empty_sequences']}")
                logger.info(f"  Sparsity ratio: {result.get('sparsity_ratio', 0.0):.4f}")
            else:
                logger.error(f"Task analysis failed: {result['error']}")
        except Exception as e:
            logger.error(f"Error analyzing task {db_name} - {table_name}.{target_col}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Analysis Complete - Summary")
    logger.info("=" * 80)

    successful_results = [r for r in all_results if 'error' not in r]
    failed_results = [r for r in all_results if 'error' in r]

    logger.info(f"Total tasks processed: {len(all_results)}")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(failed_results)}")

    if successful_results:
        logger.info("\nSuccessful tasks:")
        for result in successful_results:
            if result.get('task_type', 'classification') == 'regression':
                logger.info(f"  {result['db_name']} - {result['table_name']}.{result['target_col']}: "
                           f"corr = {result.get('mean_label_correlation', 0.0):.4f}")
            else:
                logger.info(f"  {result['db_name']} - {result['table_name']}.{result['target_col']}: "
                           f"mean ratio = {result.get('mean_same_class_ratio', 0.0):.4f}")

    if failed_results:
        logger.warning("\nFailed tasks:")
        for result in failed_results:
            logger.warning(f"  {result['db_name']} - {result['table_name']}.{result['target_col']}: "
                          f"{result['error']}")


if __name__ == "__main__":
    main()
