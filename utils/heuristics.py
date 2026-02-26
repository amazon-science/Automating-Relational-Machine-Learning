"""
SQL-based heuristics for relbench tasks.
This can't scale well when the dataset is large.
"""

from relbench.base import Dataset, EntityTask, Table, TaskType
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import mode
from typing import Dict
import duckdb as ddb
import pandas as pd
from typing import List
import os
from tqdm import tqdm
from typing import Optional, Literal, Tuple, Iterable, Sequence
import math

def evaluate(task, train_table: Table, pred_table: Table, name: str) -> Dict[str, float]:
    """
        Generate heuristic predictions for a given task, train table, and prediction table.
        Args:
            task: Task object
            train_table: Table object
            pred_table: Table object
            name: Name of the heuristic
        Returns:
            Dictionary with heuristic results
    """
    is_test = task.target_col not in pred_table.df
    num_classes = len(pred_table.df[task.target_col].unique())
    if task.task_type == 'regression' or (not isinstance(task.task_type, str) and task.task_type.value == 'regression'):
        ## regression_task
        num_classes = 1
    if name == "global_zero":
        pred = np.zeros(len(pred_table))
        if num_classes > 2:
            pred = np.zeros((len(pred_table), num_classes))
    elif name == "global_mean":
        mean = train_table.df[task.target_col].astype(float).values.mean()
        pred = np.ones(len(pred_table)) * mean
        if num_classes > 2:
            guessed_label = int(mean)
            ## convert to one-hot encoding
            pred = np.zeros((len(pred_table), num_classes))
            pred[:, guessed_label] = 1
    elif name == "global_median":
        median = np.median(train_table.df[task.target_col].astype(float).values)
        pred = np.ones(len(pred_table)) * median
        if num_classes > 2:
            guessed_label = int(median)
            ## convert to one-hot encoding
            pred = np.zeros((len(pred_table), num_classes))
            pred[:, guessed_label] = 1
    elif name == "entity_mean":
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: "mean"})
        df.rename(columns={task.target_col: "__target__"}, inplace=True)
        df = pred_table.df.merge(df, how="left", on=fkey)
        pred_1d = df["__target__"].fillna(0).astype(float).values
        if num_classes > 2:
            pred = np.zeros((len(pred_table), num_classes))
            for i in range(len(pred_table)):
                pred[i, int(pred_1d[i])] = 1
        else:
            pred = pred_1d
    elif name == "entity_median":
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: "median"})
        df.rename(columns={task.target_col: "__target__"}, inplace=True)
        df = pred_table.df.merge(df, how="left", on=fkey)
        pred_1d = df["__target__"].fillna(0).astype(float).values
        if num_classes > 2:
            pred = np.zeros((len(pred_table), num_classes))
            for i in range(len(pred_table)):
                pred[i, int(pred_1d[i])] = 1
        else:
            pred = pred_1d
    elif name == "random":
        pred = np.random.rand(len(pred_table))
    elif name == "majority":
        past_target = train_table.df[task.target_col].astype(int)
        majority_label = int(past_target.mode().iloc[0])
        pred = torch.full((len(pred_table),), fill_value=majority_label).numpy()
        if num_classes > 2:
            pred = np.zeros((len(pred_table), num_classes))
            pred[:, majority_label] = 1
    elif name == "majority_multilabel":
        past_target = train_table.df[task.target_col]
        majority = mode(np.stack(past_target.values), axis=0).mode[0]
        pred = np.stack([majority] * len(pred_table.df))
    elif name == "random_multilabel":
        num_labels = train_table.df[task.target_col].values[0].shape[0]
        pred = np.random.rand(len(pred_table), num_labels)
    else:
        raise ValueError(f"Unknown eval name called {name}.")
    return task.evaluate(pred, None if is_test else pred_table)

def heuristics(task, train_table, val_table, selected_heuristics = ['global_zero', 'global_mean', 'global_median', 'entity_mean', 'entity_median', 'majority']):
    """
        Generate heuristic predictions for a given task, train table, and validation table.
        Args:
            task: Task object
            train_table: Table object
            val_table: Table object
            selected_heuristics: List of heuristic names
        Returns:
            Dictionary with heuristic results
    """
    train_result_dict = defaultdict(list)
    val_result_dict = defaultdict(list)
    for heuristic in selected_heuristics:
        train_result = evaluate(task, train_table, train_table, heuristic)
        val_result = evaluate(task, train_table, val_table, heuristic)
        train_result_dict[heuristic].append(train_result)
        val_result_dict[heuristic].append(val_result)
    return train_result_dict, val_result_dict

def heuristics_dict_to_vector(heuristics_dict):
    """
    Convert heuristics result dictionary to a 24-dimensional vector.
    
    Args:
        heuristics_dict: Dictionary with heuristic results
        task_type: 'classification' or 'regression' to determine decimal precision
    
    Returns:
        List of 24 float values
    """
    # Define the order of heuristics and metrics
    heuristic_order = ['global_zero', 'global_mean', 'global_median', 'entity_mean', 'entity_median', 'majority']
    metric_order = ['average_precision', 'accuracy', 'f1', 'roc_auc']
    
    vector = []
    
    for heuristic in heuristic_order:
        if heuristic in heuristics_dict:
            # Get the first (and typically only) result dictionary
            result_dict = heuristics_dict[heuristic][0]
            for metric in metric_order:
                if metric in result_dict:
                    value = float(result_dict[metric])
                    vector.append(value)
                else:
                    # If metric is missing, append 0
                    vector.append(0.0)
        else:
            # If heuristic is missing, append 4 zeros (for 4 metrics)
            vector.extend([0.0, 0.0, 0.0, 0.0])
    
    return vector


def get_simple_stats(db_obj):
    num_of_tables = len(db_obj.table_dict)
    num_of_rows = sum(len(table.df) for table in db_obj.table_dict.values())
    num_of_columns = sum(len(table.df.columns) for table in db_obj.table_dict.values())
    return num_of_tables, num_of_rows, num_of_columns



def sample_tables(
    df: pd.DataFrame,
    row_threshold: int,
    k: int,
    n: int,
    *,
    replace: bool = False,
    disjoint: bool = False,
    random_state: int | None = None,
) -> List[pd.DataFrame]:
    """
    If len(df) > row_threshold, return k sampled DataFrames, each with n rows.
    Otherwise, return [df.copy()].

    Args:
        df: Source DataFrame.
        row_threshold: Only sample when len(df) > row_threshold.
        k: Number of sampled tables to generate.
        n: Number of rows per sampled table.
        replace: Sample with replacement within each sample.
        disjoint: Ensure no row overlap across the k samples (requires replace=False
                  and k*n <= len(df)).
        random_state: Seed for reproducibility.

    Returns:
        List of DataFrames (length k if sampling happens, otherwise length 1).
    """
    m = len(df)
    if m <= row_threshold:
        return [df.copy()]

    if not replace and n > m:
        raise ValueError(
            f"Requested n={n} exceeds df size m={m} without replacement. "
            f"Set replace=True or reduce n."
        )

    rng = np.random.default_rng(random_state)

    if disjoint:
        if replace:
            raise ValueError("disjoint=True requires replace=False.")
        if k * n > m:
            raise ValueError(
                f"Not enough rows for disjoint sampling: k*n={k*n} > m={m}."
            )
        # One permutation, sliced into k chunks of size n
        perm = rng.permutation(m)
        return [df.iloc[perm[i*n:(i+1)*n]].reset_index(drop=True) for i in range(k)]

    # Independent samples (may overlap across samples if replace=False)
    samples = []
    for _ in range(k):
        idx = rng.choice(m, size=n, replace=replace)
        samples.append(df.iloc[idx].reset_index(drop=True))
    return samples

def time_homophily(task_table, id_col, time_col, label_col, return_details=False,
                   task="classification", reg_metric="gaussian", eps=None, sigma=None):
    """
    Compute temporal homophily using vectorization.
    
    For classification: For each row, among older rows with the same ID, compute the fraction 
    that share the same label. Return the mean of these fractions across all rows that have history.
    
    For regression: For each row, among older rows with the same ID, compute the average similarity
    using kernel-based measures. Return the mean of these similarities across all rows that have history.

    Args:
        task_table: DataFrame with task data
        id_col: Column name for entity ID
        time_col: Column name for timestamp (any parseable type)
        label_col: Column name for label/target
        return_details: if True, also return a DataFrame with per-row stats
        task: 'classification' (default) or 'regression'
        reg_metric: For regression, similarity metric - 'eps' or 'gaussian'
        eps: For 'eps' metric, absolute tolerance; if None, defaults to 0.5*std of labels
        sigma: For 'gaussian' metric, scale parameter; if None, uses std of labels (fallback=1.0)

    Returns:
        If return_details=False (default): float temporal_homophily
        If return_details=True: (float temporal_homophily, DataFrame details)
    """
    df = task_table.copy()

    # Ensure datetime and stable sort within ties
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values([id_col, time_col], kind='mergesort')
    orig_index = df.index

    if task == "classification":
        # Original classification logic
        # previous rows count within each ID
        prev_count = df.groupby(id_col, sort=False).cumcount()

        # previous rows with the same label within each (ID, label) pair
        same_label_prev = df.groupby([id_col, label_col], sort=False).cumcount()

        # Per-row ratio; first rows in each ID (prev_count==0) -> NaN by design
        denom = prev_count.replace(0, np.nan)
        ratio = same_label_prev / denom

        temporal_homophily = ratio.mean()  # NaNs ignored

        if not return_details:
            return float(temporal_homophily)

        # Optional detailed output aligned to original row order
        out = df.copy()
        out['num_historical_rows'] = prev_count.values
        out['num_same_label'] = same_label_prev.values
        out['temporal_homophily'] = ratio.values
        out = out.set_index(orig_index).sort_index()

        return float(temporal_homophily), out

    elif task == "regression":
        # Convert labels to numeric for regression
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        
        # Set default parameters for regression
        if sigma is None:
            s = df[label_col].std(skipna=True)
            sigma = float(s) if pd.notna(s) and s > 0 else 1.0
        
        if reg_metric == "eps" and eps is None:
            eps = 0.5 * sigma

        # Initialize results
        temporal_homophily_values = []
        num_historical_rows = []
        avg_similarities = []

        # Process each entity separately
        for entity_id in df[id_col].unique():
            entity_data = df[df[id_col] == entity_id].copy()
            
            if len(entity_data) <= 1:
                # No history for this entity
                temporal_homophily_values.extend([np.nan] * len(entity_data))
                num_historical_rows.extend([0] * len(entity_data))
                avg_similarities.extend([np.nan] * len(entity_data))
                continue

            current_label = entity_data[label_col].values
            similarities_per_row = []

            for i in range(len(entity_data)):
                if i == 0:
                    # First row has no history
                    temporal_homophily_values.append(np.nan)
                    num_historical_rows.append(0)
                    avg_similarities.append(np.nan)
                    similarities_per_row.append([])
                    continue

                # Get historical labels (all previous rows)
                historical_labels = current_label[:i]
                current_val = current_label[i]
                
                # Compute similarities with historical labels
                if reg_metric == "eps":
                    # Binary similarity: 1 if within eps, 0 otherwise
                    similarities = (np.abs(historical_labels - current_val) <= eps).astype(float)
                elif reg_metric == "gaussian":
                    # Gaussian kernel similarity
                    similarities = np.exp(-((historical_labels - current_val) ** 2) / (2 * sigma ** 2))
                    
                else:
                    raise ValueError("Unknown reg_metric; choose 'eps' or 'gaussian'")

                # Average similarity
                avg_sim = np.mean(similarities)
                
                temporal_homophily_values.append(avg_sim)
                num_historical_rows.append(len(historical_labels))
                avg_similarities.append(avg_sim)
                similarities_per_row.append(similarities.tolist())

        # Convert to numpy array for consistency
        temporal_homophily_values = np.array(temporal_homophily_values)
        num_historical_rows = np.array(num_historical_rows)
        avg_similarities = np.array(avg_similarities)

        # Overall temporal homophily (mean across all rows with history)
        temporal_homophily = np.nanmean(temporal_homophily_values)

        if not return_details:
            return float(temporal_homophily)

        # Optional detailed output aligned to original row order
        out = df.copy()
        out['num_historical_rows'] = num_historical_rows
        out['avg_similarity'] = avg_similarities
        out['temporal_homophily'] = temporal_homophily_values
        out = out.set_index(orig_index).sort_index()

        return float(temporal_homophily), out

    else:
        raise ValueError("task must be 'classification' or 'regression'")

def find_neighbor_types(data, entity_table):
    """
    Find all neighboring tables for a given entity table based on edge types.
    
    Args:
        data: Graph data object with edge_types attribute
        entity_table: Name of the entity table to find neighbors for
        
    Returns:
        set: Set of neighboring table names
    """
    neighboring_tables = set()
    
    # Iterate through all edge types to find connections to the entity table
    for edge_type in data.edge_types:
        source_table, edge_name, target_table = edge_type
        
        # If entity table is the source, add target as neighbor
        if source_table == entity_table:
            neighboring_tables.add(target_table)
        
        # If entity table is the target, add source as neighbor  
        if target_table == entity_table:
            neighboring_tables.add(source_table)
    
    # Remove the entity table itself if it appears (self-loops)
    neighboring_tables.discard(entity_table)
    return neighboring_tables

def find_edge_relationships(db_obj, target_table):
    result = {}
    for table_name in target_table:
        fkey_col_to_pkey_table = db_obj.table_dict[table_name].fkey_col_to_pkey_table
        result[table_name] = fkey_col_to_pkey_table
    return result

def generate_relation(edge_rel, db_obj, entity_col):
    """
        Given edge_rel, db_obj as input, generate edge relations
    """
    relations = {}
    for key, value in edge_rel.items():
        ## if key is a table with time_dict, then it's easy
        table = db_obj.table_dict[key]
        relations[key] = []
        # Transform foreign key relationships to (entity_col, foreign_key_col, time_col) format
        for fkey_col, referenced_table in value.items():
            if fkey_col == entity_col:
                continue
            if table.time_col:
                relations[key].append((entity_col, fkey_col, table.time_col))
            else:
                # ## try to find time_col through another relationship
                # referenced_table_obj = db_obj.table_dict[referenced_table]
                # if referenced_table_obj.time_col:
                #     relations[key].append((entity_col, fkey_col, referenced_table, referenced_table_obj.time_col))
                # else:
                    ## if still not found, no time constraint
                relations[key].append((entity_col, fkey_col, None))
    return relations
            
        
def gini_degree(degrees, correction=False):
    x = np.asarray(degrees, dtype=float).flatten()
    if np.any(x < 0):
        raise ValueError("Degrees must be nonnegative.")
    s = x.sum()
    if s == 0:
        return 0.0
    x.sort()
    n = x.size
    i = np.arange(1, n+1)
    G = np.sum((2*i - n - 1) * x) / (n * s)
    if correction and n > 1:           # small-sample (unbiased) correction (optional)
        G *= n / (n - 1)
    return float(G)

def adjust_homophily(homophily, labels):
    """
    Adjust homophily values by accounting for class imbalance.
    
    Formula: (homophily - pi) / (1 - pi) where pi is the class proportion
    
    Args:
        homophily: Array-like of homophily values
        labels: Array-like of class labels
        
    Returns:
        numpy.ndarray: Adjusted homophily values
    """
    homophily = np.asarray(homophily)
    labels = np.asarray(labels, dtype=int)
    
    # Get class proportions efficiently
    classes, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    
    # Create mapping for vectorized operations using advanced indexing
    # This is faster than dictionary lookups for large arrays
    max_label = classes.max() if len(classes) > 0 else 0
    if max_label < 10000:  # Use direct indexing for small label ranges
        prop_array = np.zeros(max_label + 1)
        prop_array[classes] = counts / n_total
        pi_values = prop_array[labels]
    else:  # Use searchsorted for large label ranges
        sort_idx = np.argsort(classes)
        sorted_classes = classes[sort_idx]
        sorted_proportions = counts[sort_idx] / n_total
        indices = np.searchsorted(sorted_classes, labels)
        pi_values = sorted_proportions[indices]
    
    # Vectorized computation with proper handling of division by zero
    # When pi == 1, the denominator becomes 0, so we set result to 0
    denominator = 1 - pi_values
    mask = denominator != 0
    
    adj_h = np.zeros_like(homophily, dtype=float)
    adj_h[mask] = (homophily[mask] - pi_values[mask]) / denominator[mask]
    
    return adj_h

def retrieve_entity_key_pairs_from_table(conn, table_name, relation_tuple):
    """
    Given a relation tuple, retrieve (entity, key, time) pairs from a table.

    Args:
        conn: DuckDB connection
        table_name: Name of the source table (e.g., 'results')
        relation_tuple: A tuple describing the relation. Supported forms:
            - (entity_col, key_col, time_col)
            - (entity_col, key_col, None)
            - (entity_col, key_col, referenced_table, time_col)

    Returns:
        DataFrame with columns: ['entity', 'key', 'time'] where 'time' is pandas datetime
    """
    if not isinstance(relation_tuple, (list, tuple)) or len(relation_tuple) < 2:
        raise ValueError("relation_tuple must be a tuple like (entity_col, key_col, [time_col]) or (entity_col, key_col, referenced_table, time_col)")

    entity_col = relation_tuple[0]
    key_col = relation_tuple[1]

    if len(relation_tuple) == 3:
        # Format: (entity_col, key_col, time_col)
        time_col = relation_tuple[2]
        if time_col is not None and not isinstance(time_col, str):
            raise ValueError("Only direct time column in the same table is supported in 3-element relation_tuple")
        
        select_cols = f"{entity_col} AS entity, {key_col} AS key"
        if time_col is not None:
            select_cols += f", {time_col} AS time"
        else:
            select_cols += ", NULL AS time"

        df = conn.execute(f"SELECT {select_cols} FROM {table_name}").df()
        
    elif len(relation_tuple) == 4:
        # Format: (entity_col, key_col, referenced_table, time_col)
        referenced_table = relation_tuple[2]
        time_col = relation_tuple[3]
        
        # Join the current table with the referenced table to get time information
        select_cols = f"t1.{entity_col} AS entity, t1.{key_col} AS key, t2.{time_col} AS time"
        query = f"""
        SELECT {select_cols}
        FROM {table_name} t1
        JOIN {referenced_table} t2 ON t1.{key_col} = t2.{key_col}
        """
        df = conn.execute(query).df()
        
    else:
        raise ValueError(f"Unsupported relation_tuple format: {relation_tuple}")

    # Drop rows with missing entity/key
    df = df.dropna(subset=["entity", "key"]).copy()
    # Ensure datetime
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def compute_degree_and_homophily_from_relation(train_task_table, relation_pairs_df, entity_col, label_col, time_col):
    """
    Compute per-row degree and homophily for an entity in train_task_table induced by a relation.

    The relation is given as (entity, key, time) pairs (e.g., driverId participates in raceId at date).
    For each row (u, t) in train_task_table, we:
      - Consider all keys associated with u at times <= t
      - Neighbors are all other entities that have those keys at times <= t
      - Degree is the number of unique neighbors
      - Homophily is the fraction of neighbors whose most recent label at or before t equals u's label at t

    Future information is masked using t for each row.

    Args:
        train_task_table: DataFrame with at least [time_col, entity_col, label_col]
        relation_pairs_df: DataFrame with columns ['entity', 'key', 'time']
        entity_col: Entity column in train_task_table (e.g., 'driverId')
        label_col: Label column in train_task_table (e.g., 'did_not_finish')
        time_col: Time column in train_task_table (e.g., 'date')

    Returns:
        DataFrame indexed like train_task_table with two columns: ['degree', 'homophily']
    """
    if not {entity_col, time_col, label_col}.issubset(set(train_task_table.columns)):
        raise ValueError("train_task_table must contain entity, time, and label columns")

    required_cols = {"entity", "key", "time"}
    if not required_cols.issubset(set(relation_pairs_df.columns)):
        raise ValueError(f"relation_pairs_df must have columns {required_cols}")

    train_df = train_task_table[[entity_col, time_col, label_col]].copy()
    train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce")
    relation_df = relation_pairs_df.copy()
    relation_df["time"] = pd.to_datetime(relation_df["time"], errors="coerce")

    # Row identifiers to map results back
    train_df = train_df.reset_index().rename(columns={"index": "row_id"})

    # Step 1: For each train row (u, t), collect all keys for u with relation time <= t
    # Join on entity, then filter by time
    tk = train_df.merge(
        relation_df.rename(columns={"entity": entity_col}),
        on=entity_col,
        how="left",
    )
    # Filter to historical keys only (mask future)
    # Only keep historical keys with known timestamps to avoid future leakage
    tk = tk[tk["time"].notna() & (tk["time"] <= tk[time_col])].copy()
    # Keep minimal columns
    tk = tk[["row_id", entity_col, time_col, label_col, "key"]].dropna(subset=["key"])

    if tk.empty:
        # No relations; return zeros/NaNs
        out = pd.DataFrame({
            "degree": np.zeros(len(train_task_table), dtype=int),
            "homophily": np.nan,
        })
        out.index = train_task_table.index
        return out

    # Step 2: For each (row_id, key, t), find neighbors with same key and relation time <= t
    neighbor_candidates = tk.merge(
        relation_df,
        on="key",
        how="left",
        suffixes=("_cur", "_nbr"),
    )
    neighbor_candidates = neighbor_candidates[
        neighbor_candidates["time"].notna() & (neighbor_candidates["time"] <= neighbor_candidates[time_col])
    ].copy()
    neighbor_candidates = neighbor_candidates.rename(columns={"entity": "neighbor_entity"})

    # Exclude self
    neighbor_candidates = neighbor_candidates[neighbor_candidates["neighbor_entity"] != neighbor_candidates[entity_col]]

    # Deduplicate neighbors per row_id
    neighbors_unique = neighbor_candidates[["row_id", "neighbor_entity", time_col, label_col]].drop_duplicates(
        subset=["row_id", "neighbor_entity"]
    )

    # Degree per row
    degree_series = neighbors_unique.groupby("row_id")["neighbor_entity"].nunique()

    # Step 3: Homophily via neighbor labels at or before t
    # Prepare neighbor label lookup using merge_asof
    # --- Prepare labels and neighbors for merge_asof (sorted + aligned dtypes) ---
    # --- Step 3 (robust): homophily via group-wise asof (no `by=`) ---

# Prepare labels (neighbor's last known label at or before t)
    train_labels = (
        train_df[[entity_col, time_col, label_col]]
        .dropna(subset=[time_col])
        .copy()
    )
    # Normalize timezones/dtypes to avoid hidden mismatches
    train_labels[time_col] = pd.to_datetime(train_labels[time_col], errors="coerce").dt.tz_localize(None)
    neighbors_unique[time_col] = pd.to_datetime(neighbors_unique[time_col], errors="coerce").dt.tz_localize(None)
    neighbors_unique["neighbor_entity"] = neighbors_unique["neighbor_entity"].astype(train_df[entity_col].dtype)

    # Build a dict of per-entity right tables, each sorted by time
    labels_by_entity = {
        ent: g[[time_col, label_col]].sort_values(time_col, kind="mergesort")
        for ent, g in train_labels.groupby(entity_col, sort=False)
    }

    # Left table for asof: row_id, neighbor_entity, current row's time
    left = neighbors_unique[["row_id", "neighbor_entity", time_col]].dropna(subset=[time_col])

    parts = []
    for ent, g in tqdm(left.groupby("neighbor_entity", sort=False)):
        right = labels_by_entity.get(ent)
        if right is None or right.empty:
            tmp = g.copy()
            tmp["neighbor_label"] = np.nan
            parts.append(tmp)
            continue

        # Sort left within the group by time (as required by merge_asof)
        g_sorted = g.sort_values(time_col, kind="mergesort")
        merged = pd.merge_asof(
            g_sorted,
            right.rename(columns={label_col: "neighbor_label"}),
            on=time_col,
            direction="backward",
            allow_exact_matches=True,
        )
        parts.append(merged)

    neighbor_labels = pd.concat(parts, ignore_index=True)

    # Join current row's label for comparison
    current_labels = train_df[["row_id", label_col]].rename(columns={label_col: "current_label"})
    neighbor_labels = neighbor_labels.merge(current_labels, on="row_id", how="left")

    # Compare labels
    neighbor_labels["label_match"] = neighbor_labels["neighbor_label"] == neighbor_labels["current_label"]

    # Compute homophily per row: mean over available neighbor labels
    def _homophily_group(g):
        valid = g["neighbor_label"].notna()
        if not valid.any():
            return np.nan
        return g.loc[valid, "label_match"].mean()

    homophily_series = neighbor_labels.groupby("row_id", sort=False).apply(_homophily_group)


    # Assemble output aligned to original index
    out = pd.DataFrame({
        "degree": degree_series,
        "homophily": homophily_series,
    })
    out = out.reindex(train_df["row_id"]).set_index(train_task_table.index)
    out["degree"] = out["degree"].fillna(0).astype(int)
    return out



def compute_degree_and_homophily_duckdb(
    train_task_table: pd.DataFrame,
    relation_pairs_df: pd.DataFrame,
    entity_col: str,
    label_col: str,
    time_col: str,
    conn: Optional[ddb.DuckDBPyConnection] = None,
    *,
    # --- Performance/robustness knobs ---
    threads: Optional[int] = None,            # e.g., os.cpu_count()
    memory_limit: Optional[str | int] = None, # e.g., "16GB" or 16000000000 (bytes) or 16 (GB)
    temp_directory: Optional[str] = None,     # e.g., "/mnt/fastssd/tmp"
    max_key_fanout: Optional[int] = None,     # e.g., 5000 → drop keys with >5000 participants
    chunk_by: Optional[str] = None,           # pandas offset alias, e.g., "MS" (month start), "W", "QS"
) -> pd.DataFrame:
    """
    Scalable DuckDB implementation.

    Parameters
    ----------
    train_task_table : DataFrame
        Must contain [entity_col, time_col, label_col]. Rows are the evaluation points (u, t_row, label(u,t_row)).
    relation_pairs_df : DataFrame
        Must contain ['entity','key','time'] describing participation (entity,key) at 'time'.
        We'll compress to each (entity,key)'s *first* occurrence to avoid fanout explosion.
    entity_col, label_col, time_col : str
        Column names in train_task_table.
    conn : duckdb.DuckDBPyConnection, optional
        Existing connection to reuse. If None, a temporary connection is created and closed.
    threads : int, optional
        Set PRAGMA threads to this exact integer.
    memory_limit : str|int, optional
        Set PRAGMA memory_limit. If int:
          - if >= 1e9 we treat as bytes; else we treat as GB (e.g., 16 -> "16GB").
        If str, passed through (e.g., "0" for unlimited, "16GB").
    temp_directory : str, optional
        PRAGMA temp_directory for spill (use a fast SSD path).
    max_key_fanout : int, optional
        If provided, drop any key whose participant count (over (entity,key) first appearance) exceeds this threshold.
    chunk_by : str, optional
        If provided, process train_task_table in time chunks using a pandas offset alias.
        Recommended values: "MS" (month-start), "W", "QS". Use None to process all at once.

    Returns
    -------
    out : DataFrame
        Index aligned to train_task_table.index with columns:
            - degree : int
            - homophily : float or NaN (if no labeled neighbors)

    Notes
    -----
    - Uses ASOF JOIN to fetch neighbor labels at t_row (<= t_row latest), avoiding interval range joins.
    - Deduplicates relation table to (entity,key)'s *first_time* to control join fanout.
    - Optionally caps extremely large keys via max_key_fanout.
    - Optionally slices the computation by time windows to lower peak memory.
    """
    # ---- Basic validation
    need_train = {entity_col, time_col, label_col}
    if not need_train.issubset(train_task_table.columns):
        raise ValueError(f"train_task_table must contain columns {need_train}.")

    need_rel = {"entity", "key", "time"}
    if not need_rel.issubset(relation_pairs_df.columns):
        raise ValueError(f"relation_pairs_df must have columns {need_rel}")

    # Keep original index to restore order
    orig_index = train_task_table.index

    # Shallow copies & normalize timestamps (tz-naive)
    train_df = train_task_table[[entity_col, time_col, label_col]].copy()
    train_df = train_df.reset_index(drop=False).rename(columns={"index": "row_id"})
    train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce").dt.tz_localize(None)

    rel_df = relation_pairs_df[["entity", "key", "time"]].copy()
    rel_df["time"] = pd.to_datetime(rel_df["time"], errors="coerce").dt.tz_localize(None)

    # Drop rows with null times (not joinable)
    train_df = train_df[train_df[time_col].notna()]
    rel_df = rel_df[rel_df["time"].notna()]
    if len(train_df) == 0:
        # No valid rows: return zeros/NaNs aligned to orig_index
        out = pd.DataFrame({"degree": 0, "homophily": pd.Series([math.nan] * len(orig_index), index=orig_index)})
        out["degree"] = out["degree"].astype(int)
        return out

    # ---- Open/own connection?
    close_when_done = False
    if conn is None:
        conn = ddb.connect()
        close_when_done = True

    # ---- PRAGMA tuning (only if provided)
    try:
        if threads is not None:
            if not isinstance(threads, int) or threads <= 0:
                raise ValueError("threads must be a positive integer")
            conn.execute(f"PRAGMA threads={threads};")

        if memory_limit is not None:
            if isinstance(memory_limit, int):
                # Heuristic: interpret small ints as GB, large ints as bytes
                if memory_limit >= 1_000_000_000:
                    mem_str = str(memory_limit)
                else:
                    mem_str = f"{memory_limit}GB"
            else:
                mem_str = str(memory_limit)
            conn.execute(f"PRAGMA memory_limit='{mem_str}';")

        if temp_directory is not None:
            conn.execute(f"PRAGMA temp_directory='{temp_directory}';")

        # Cache compiled objects & data
        conn.execute("PRAGMA enable_object_cache=true;")

        # ---- Register as sources then materialize as TEMP tables (lets DuckDB manage spill)
        conn.register("train_src", train_df)
        conn.register("rel_src", rel_df)
        conn.execute("CREATE TEMP TABLE train_df AS SELECT * FROM train_src;")
        conn.execute("CREATE TEMP TABLE rel_df   AS SELECT * FROM rel_src;")

        # ---- Build first-appearance (entity,key) → first_time
        conn.execute("""
            CREATE TEMP TABLE rel_first AS
            SELECT CAST(entity AS VARCHAR) AS entity,
                   key,
                   CAST(MIN(time) AS TIMESTAMP) AS first_time
            FROM rel_df
            WHERE entity IS NOT NULL AND key IS NOT NULL
            GROUP BY 1,2;
        """)

        # ---- Optionally cap huge keys
        if max_key_fanout is not None:
            if not isinstance(max_key_fanout, int) or max_key_fanout <= 0:
                raise ValueError("max_key_fanout must be a positive integer")
            conn.execute("""
                CREATE TEMP TABLE key_card AS
                SELECT key, COUNT(*)::BIGINT AS n_entities
                FROM rel_first
                GROUP BY 1;
            """)
            conn.execute("""
                CREATE TEMP TABLE rel_first_use AS
                SELECT rf.*
                FROM rel_first rf
                JOIN key_card kc USING (key)
                WHERE kc.n_entities <= ?;
            """, [max_key_fanout])
        else:
            conn.execute("CREATE TEMP TABLE rel_first_use AS SELECT * FROM rel_first;")

        # ---- Chunk planning
        windows: Iterable[Tuple[pd.Timestamp, pd.Timestamp]] = [(_min_time(train_df[time_col]), _next_tick(_max_time(train_df[time_col])))]
        if chunk_by is not None:
            windows = list(_time_windows(train_df[time_col], chunk_by))

        results = []

        # ---- SQL pipeline (references a pre-made TEMP TABLE `this_train`)
        MAIN_QUERY = """
        WITH
        tk AS (
          -- For each (row_id, u, t_row), collect u's keys whose first_time <= t_row
          SELECT tr.row_id, tr.entity AS u, tr.t_row, tr.current_label, rf.key
          FROM this_train tr
          JOIN rel_first_use rf
            ON rf.entity = tr.entity
           AND rf.first_time <= tr.t_row
        ),
        candidates AS (
          -- Other entities sharing those keys with first_time <= t_row (exclude self)
          SELECT tk.row_id, CAST(r2.entity AS VARCHAR) AS neighbor_entity, tk.t_row, tk.current_label
          FROM tk
          JOIN rel_first_use r2
            ON r2.key = tk.key
           AND r2.first_time <= tk.t_row
          WHERE r2.entity <> tk.u
        ),
        neighbors_dedup AS (
          -- Deduplicate per (row_id, neighbor_entity) to control fanout
          SELECT row_id,
                 neighbor_entity,
                 any_value(t_row)       AS t_row,
                 any_value(current_label) AS current_label
          FROM candidates
          GROUP BY row_id, neighbor_entity
        ),
        degree AS (
          SELECT row_id, COUNT(*)::BIGINT AS degree
          FROM neighbors_dedup
          GROUP BY row_id
        ),
        train_dedup AS (
          -- ensure one label per (entity, t_row) for ASOF
          SELECT entity, t_row, any_value(current_label) AS current_label
          FROM this_train
          GROUP BY entity, t_row
          ORDER BY entity, t_row
        ),
        neighbor_labels AS (
          -- For each neighbor at time t_row, get their latest label <= t_row
          SELECT n.row_id, n.neighbor_entity, t.current_label AS neighbor_label
          FROM neighbors_dedup n
          ASOF JOIN (SELECT entity, t_row, current_label FROM train_dedup ORDER BY entity, t_row) t
            ON n.neighbor_entity = t.entity AND n.t_row >= t.t_row
        ),
        homophily AS (
          SELECT
            n.row_id,
            AVG(CASE
                  WHEN nl.neighbor_label IS NULL THEN NULL         -- ignore unlabeled
                  WHEN nl.neighbor_label = n.current_label THEN 1.0
                  ELSE 0.0
                END) AS homophily
          FROM neighbors_dedup n
          LEFT JOIN neighbor_labels nl
            ON nl.row_id = n.row_id
           AND nl.neighbor_entity = n.neighbor_entity
          GROUP BY n.row_id
        )
        SELECT
          tr.row_id,
          COALESCE(d.degree, 0) AS degree,
          h.homophily
        FROM this_train tr
        LEFT JOIN degree d USING (row_id)
        LEFT JOIN homophily h USING (row_id);
        """

        # ---- Run per chunk
        for (start, end) in windows:
            # Create chunk-local training table with canonical names & types
            # Note: if chunk_by is None we still build one big 'this_train'
            if chunk_by is None:
                conn.execute(f"""
                    CREATE TEMP TABLE this_train AS
                    SELECT
                      CAST(row_id AS BIGINT) AS row_id,
                      CAST({entity_col} AS VARCHAR) AS entity,
                      CAST({time_col}  AS TIMESTAMP) AS t_row,
                      {label_col} AS current_label
                    FROM train_df
                    WHERE {time_col} IS NOT NULL;
                """)
            else:
                conn.execute(f"""
                    CREATE TEMP TABLE this_train AS
                    SELECT
                      CAST(row_id AS BIGINT) AS row_id,
                      CAST({entity_col} AS VARCHAR) AS entity,
                      CAST({time_col}  AS TIMESTAMP) AS t_row,
                      {label_col} AS current_label
                    FROM train_df
                    WHERE {time_col} IS NOT NULL
                      AND {time_col} >= ?
                      AND {time_col} <  ?;
                """, [start.to_pydatetime(), end.to_pydatetime()])

            # Skip empty chunk quickly
            n_rows = conn.execute("SELECT COUNT(*) FROM this_train;").fetchone()[0]
            if n_rows == 0:
                conn.execute("DROP TABLE this_train;")
                if chunk_by is None:
                    break
                else:
                    continue

            df_chunk = conn.execute(MAIN_QUERY).df()
            results.append(df_chunk)

            # Drop chunk-local temp to keep memory steady
            conn.execute("DROP TABLE this_train;")

            # If single-shot, don't loop further even if windows has 1 item
            if chunk_by is None:
                break

        # ---- Merge chunks
        if results:
            out = pd.concat(results, ignore_index=True)
        else:
            # Shouldn't happen, but be defensive
            out = pd.DataFrame(columns=["row_id", "degree", "homophily"])

        # Restore original order & index
        # row_id was the integer position from the *reset* index of the original train table
        out = out.set_index("row_id").reindex(range(len(orig_index)))
        out.index = orig_index
        out["degree"] = out["degree"].fillna(0).astype(int)

        return out

    finally:
        # Best-effort cleanup
        try:
            conn.execute("DROP TABLE IF EXISTS this_train;")
            conn.execute("DROP TABLE IF EXISTS train_df;")
            conn.execute("DROP TABLE IF EXISTS rel_df;")
            conn.execute("DROP TABLE IF EXISTS rel_first;")
            conn.execute("DROP TABLE IF EXISTS rel_first_use;")
            conn.execute("DROP TABLE IF EXISTS key_card;")
        except Exception:
            pass
        if close_when_done:
            try:
                conn.close()
            except Exception:
                pass


# ---------- helpers ----------

def _min_time(s: pd.Series) -> pd.Timestamp:
    t = pd.to_datetime(s, errors="coerce").dropna()
    if len(t) == 0:
        # arbitrary; caller checks emptiness earlier
        return pd.Timestamp("1970-01-01")
    return pd.Timestamp(t.min()).floor("S")


def _max_time(s: pd.Series) -> pd.Timestamp:
    t = pd.to_datetime(s, errors="coerce").dropna()
    if len(t) == 0:
        return pd.Timestamp("1970-01-01")
    return pd.Timestamp(t.max()).ceil("S")


def _next_tick(ts: pd.Timestamp) -> pd.Timestamp:
    # Exclusive-right bound helper
    return (pd.Timestamp(ts) + pd.Timedelta(seconds=1)).ceil("S")


def _time_windows(s: pd.Series, freq: str) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build half-open [start, end) windows covering all non-null times in s.
    `freq` is a pandas offset alias, e.g., 'MS'(month start), 'W', 'QS'.
    """
    tmin = _min_time(s).normalize()
    tmax = _max_time(s)
    # Ensure we include the last partial interval
    # Use date_range with closed='left' not available here; build edges and zip
    edges = pd.date_range(start=tmin, end=tmax + pd.Timedelta(days=7), freq=freq)  # generous tail
    if len(edges) == 0 or edges[0] > tmin:
        edges = pd.Index([tmin]).append(edges)
    # Ensure last edge beyond max
    if edges[-1] <= tmax:
        edges = edges.append(pd.Index([_next_tick(tmax)]))
    # Yield adjacent pairs
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        if start >= end:
            continue
        yield (start, end)



def aggregation_homophily_duckdb(
    train_task_table: pd.DataFrame,   # columns: entity_col, time_col, label_col
    relation_pairs_df: pd.DataFrame,  # columns: ['entity','key','time']
    *,
    entity_col: str,
    label_col: str,
    time_col: str,
    min_degree: int = 1,              # only score rows with >= this many labeled neighbors at t
    return_per_row: bool = True,
    conn: Optional[ddb.DuckDBPyConnection] = None,

    # ---- performance knobs (optional) ----
    threads: Optional[int] = None,            # e.g., os.cpu_count()
    memory_limit: Optional[str|int] = None,   # e.g., "16GB" or 16 (GB) or bytes
    temp_directory: Optional[str] = None,     # e.g., "/mnt/ssd/tmp"
    max_key_fanout: Optional[int] = None,     # drop keys with > this many participants
    chunk_by: Optional[str] = None,           # pandas offset alias: "MS","W","QS" → enables 2-pass chunking
) -> tuple[float, pd.DataFrame] | float:
    """
    Multi-class aggregation homophily with temporal masking (scalable):
      For each row (u, t), build neighbor label distribution d_u(t),
      compare ⟨d_u, μ_intra⟩ vs ⟨d_u, μ_out⟩, then average 1[⟨d_u, μ_intra⟩ ≥ ⟨d_u, μ_out⟩].

    Major speedups:
      - Compress relation to first appearance per (entity,key) → avoids join fanout.
      - ASOF JOIN to fetch neighbor labels at t_row (latest <= t_row) → no interval range join.
      - Early dedup of neighbors; avoid unnecessary ORDER BY/DISTINCT.
      - Optional fanout cap for pathological keys.
      - Optional two-pass time chunking to keep memory flat on very large data.

    Returns
    -------
    If return_per_row:
        (h_agg, scores_df) where h_agg is the mean win-rate,
        scores_df has columns ['row_id','s_intra','s_inter','win'].
    Else:
        h_agg only.
    """
    # ---- basic checks
    need = {entity_col, time_col, label_col}
    if not need.issubset(train_task_table.columns):
        raise ValueError(f"train_task_table must contain {need}")
    if not {"entity","key","time"}.issubset(relation_pairs_df.columns):
        raise ValueError("relation_pairs_df must have columns: ['entity','key','time']")

    # ---- materialize pandas → canonical types (tz-naive timestamps)
    train_df = train_task_table[[entity_col, time_col, label_col]].copy()
    train_df = train_df.reset_index(drop=False).rename(columns={"index": "row_id"})
    train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce").dt.tz_localize(None)
    train_df = train_df[train_df[time_col].notna()]  # drop empty time rows early

    rel_df = relation_pairs_df[["entity","key","time"]].copy()
    rel_df["time"] = pd.to_datetime(rel_df["time"], errors="coerce").dt.tz_localize(None)
    rel_df = rel_df[rel_df["time"].notna()]

    if len(train_df) == 0:
        # Nothing to score
        h_agg = float("nan")
        empty = pd.DataFrame(columns=["row_id","s_intra","s_inter","win"])
        return (h_agg, empty) if return_per_row else h_agg

    # ---- connect & PRAGMAs
    close_when_done = False
    if conn is None:
        conn = ddb.connect()
        close_when_done = True

    try:
        if threads is not None:
            if not isinstance(threads, int) or threads <= 0:
                raise ValueError("threads must be a positive integer")
            conn.execute(f"PRAGMA threads={threads};")
        if memory_limit is not None:
            if isinstance(memory_limit, int):
                mem_str = str(memory_limit) if memory_limit >= 1_000_000_000 else f"{memory_limit}GB"
            else:
                mem_str = str(memory_limit)
            conn.execute(f"PRAGMA memory_limit='{mem_str}';")
        if temp_directory:
            conn.execute(f"PRAGMA temp_directory='{temp_directory}';")
        conn.execute("PRAGMA enable_object_cache=true;")

        # ---- register & materialize as TEMP tables (lets DuckDB manage spill)
        conn.register("train_src", train_df)
        conn.register("rel_src", rel_df)
        conn.execute("CREATE TEMP TABLE train_df AS SELECT * FROM train_src;")
        conn.execute("CREATE TEMP TABLE rel_df   AS SELECT * FROM rel_src;")

        # ---- build first appearance per (entity,key)
        conn.execute("""
            CREATE TEMP TABLE rel_first AS
            SELECT CAST(entity AS VARCHAR) AS entity,
                   key,
                   CAST(MIN(time) AS TIMESTAMP) AS first_time
            FROM rel_df
            WHERE entity IS NOT NULL AND key IS NOT NULL
            GROUP BY 1,2;
        """)

        if max_key_fanout is not None:
            if not isinstance(max_key_fanout, int) or max_key_fanout <= 0:
                raise ValueError("max_key_fanout must be a positive integer")
            conn.execute("""
                CREATE TEMP TABLE key_card AS
                SELECT key, COUNT(*)::BIGINT AS n_entities
                FROM rel_first
                GROUP BY 1;
            """)
            conn.execute("""
                CREATE TEMP TABLE rel_first_use AS
                SELECT rf.*
                FROM rel_first rf
                JOIN key_card kc USING (key)
                WHERE kc.n_entities <= ?;
            """, [max_key_fanout])
        else:
            conn.execute("CREATE TEMP TABLE rel_first_use AS SELECT * FROM rel_first;")

        # ---- helper SQL fragment: from this_train → row_dist (row_id, y_u as c, k, d)
        ROW_DIST_SQL = f"""
        WITH
        tk AS (
          -- keys of u, first seen no later than t_row
          SELECT tr.row_id, tr.entity AS u, tr.t_row, tr.y AS y_u, rf.key
          FROM this_train tr
          JOIN rel_first_use rf
            ON rf.entity = tr.entity
           AND rf.first_time <= tr.t_row
        ),
        candidates AS (
          -- other entities who share those keys by t_row (exclude self)
          SELECT tk.row_id, CAST(r2.entity AS VARCHAR) AS neighbor_entity, tk.t_row, tk.y_u
          FROM tk
          JOIN rel_first_use r2
            ON r2.key = tk.key
           AND r2.first_time <= tk.t_row
          WHERE r2.entity <> tk.u
        ),
        neighbors_dedup AS (
          SELECT row_id, neighbor_entity,
                 any_value(t_row) AS t_row,
                 any_value(y_u)   AS y_u
          FROM candidates
          GROUP BY row_id, neighbor_entity
        ),
        train_dedup AS (
          SELECT entity, t_row, any_value(y) AS y
          FROM this_train
          GROUP BY entity, t_row
          ORDER BY entity, t_row
        ),
        neighbor_labels AS (
          -- latest neighbor label <= t_row (ASOF JOIN)
          SELECT n.row_id, n.y_u, t.y AS y_nbr
          FROM neighbors_dedup n
          ASOF JOIN (SELECT entity, t_row, y FROM train_dedup ORDER BY entity, t_row) t
            ON n.neighbor_entity = t.entity AND n.t_row >= t.t_row
        ),
        neighbor_counts AS (
          SELECT row_id, y_u, y_nbr AS k, COUNT(*)::BIGINT AS cnt
          FROM neighbor_labels
          WHERE y_nbr IS NOT NULL
          GROUP BY row_id, y_u, k
        ),
        row_degree AS (
          SELECT row_id, y_u, SUM(cnt)::DOUBLE AS deg
          FROM neighbor_counts
          GROUP BY row_id, y_u
        ),
        row_dist AS (
          SELECT nc.row_id, rd.y_u AS c, nc.k, nc.cnt / NULLIF(rd.deg,0) AS d
          FROM neighbor_counts nc
          JOIN row_degree rd USING (row_id, y_u)
          WHERE rd.deg >= {min_degree}
        )
        SELECT row_id, c, k, d
        FROM row_dist
        """

        # ---------- single-pass (no chunking) ----------
        if chunk_by is None:
            # Build a canonical this_train over all rows
            conn.execute(f"""
                CREATE TEMP TABLE this_train AS
                SELECT CAST(row_id AS BIGINT) AS row_id,
                       CAST({entity_col} AS VARCHAR) AS entity,
                       CAST({time_col}  AS TIMESTAMP) AS t_row,
                       CAST({label_col} AS VARCHAR)   AS y
                FROM train_df
                WHERE {time_col} IS NOT NULL;
            """)

            MAIN_SQL = f"""
            WITH
            row_dist AS (
              {ROW_DIST_SQL}
            ),
            class_mean AS (
              SELECT c, k, AVG(d) AS mu_ck, COUNT(DISTINCT row_id) AS n_c
              FROM row_dist
              GROUP BY c, k
            ),
            global_mean AS (
              SELECT k, AVG(d) AS g_k, COUNT(DISTINCT row_id) AS N
              FROM row_dist
              GROUP BY k
            ),
            class_out AS (
              SELECT cm.c, cm.k,
                     (gm.g_k * gm.N - cm.mu_ck * cm.n_c) / NULLIF(gm.N - cm.n_c, 0) AS mu_not_ck
              FROM class_mean cm
              JOIN global_mean gm USING (k)
            ),
            score_intra AS (
              SELECT rd.row_id, SUM(rd.d * cm.mu_ck) AS s_intra
              FROM row_dist rd
              JOIN class_mean cm ON cm.c = rd.c AND cm.k = rd.k
              GROUP BY rd.row_id
            ),
            score_inter AS (
              SELECT rd.row_id, SUM(rd.d * co.mu_not_ck) AS s_inter
              FROM row_dist rd
              JOIN class_out co ON co.c = rd.c AND co.k = rd.k
              GROUP BY rd.row_id
            )
            SELECT si.row_id, si.s_intra, so.s_inter,
                   CASE WHEN si.s_intra >= so.s_inter THEN 1.0 ELSE 0.0 END AS win
            FROM score_intra si
            JOIN score_inter so USING (row_id);
            """

            scores_df = conn.execute(MAIN_SQL).df()
            # deterministic small sort (cheap) for readability
            if len(scores_df):
                scores_df = scores_df.sort_values("row_id", kind="stable").reset_index(drop=True)

        # ---------- two-pass chunking ----------
        else:
            # 1st pass: accumulate class/global means (sum d and counts) across chunks
            class_sum: Dict[tuple, float] = {}   # (c,k) -> sum d
            class_count: Dict[str, int] = {}     # c -> n_c (#rows)
            global_sum: Dict[str, float] = {}    # k -> sum d
            N_total = 0

            for start, end in _time_windows(train_df[time_col], chunk_by):
                conn.execute(f"""
                    CREATE TEMP TABLE this_train AS
                    SELECT CAST(row_id AS BIGINT) AS row_id,
                           CAST({entity_col} AS VARCHAR) AS entity,
                           CAST({time_col}  AS TIMESTAMP) AS t_row,
                           CAST({label_col} AS VARCHAR)   AS y
                    FROM train_df
                    WHERE {time_col} IS NOT NULL
                      AND {time_col} >= ?
                      AND {time_col} <  ?;
                """, [start.to_pydatetime(), end.to_pydatetime()])

                # row_dist of this chunk
                rd = conn.execute(ROW_DIST_SQL).df()  # columns: row_id, c, k, d

                if not rd.empty:
                    # per-class row counts (distinct rows surviving min_degree)
                    n_c = rd[["row_id","c"]].drop_duplicates().groupby("c", as_index=False).size()
                    for _, row in n_c.iterrows():
                        c = row["c"]
                        class_count[c] = class_count.get(c, 0) + int(row["size"])

                    # class (c,k) sums
                    agg_ck = rd.groupby(["c","k"], as_index=False)["d"].sum()
                    for _, row in agg_ck.iterrows():
                        key = (row["c"], row["k"])
                        class_sum[key] = class_sum.get(key, 0.0) + float(row["d"])

                    # global sums
                    agg_k = rd.groupby("k", as_index=False)["d"].sum()
                    for _, row in agg_k.iterrows():
                        k = row["k"]
                        global_sum[k] = global_sum.get(k, 0.0) + float(row["d"])

                    # N_total (distinct row_id)
                    N_total += rd["row_id"].nunique()

                conn.execute("DROP TABLE this_train;")

            # compute μ_c(k) and μ_not_c(k)
            if N_total == 0:
                scores_df = pd.DataFrame(columns=["row_id","s_intra","s_inter","win"])
            else:
                mu_rows = []
                for (c,k), s in class_sum.items():
                    n_c = class_count.get(c, 0)
                    if n_c > 0:
                        mu_rows.append((str(c), str(k), s / n_c))
                mu_df = pd.DataFrame(mu_rows, columns=["c","k","mu_ck"])

                g_rows = []
                for k, s in global_sum.items():
                    g_rows.append((str(k), s / N_total))
                g_df = pd.DataFrame(g_rows, columns=["k","g_k"])

                n_df = pd.DataFrame([(str(c), int(n)) for c, n in class_count.items()], columns=["c","n_c"])
                N_df = pd.DataFrame({"N":[N_total]})

                # precompute μ_not outside DB to simplify joins
                mu_not = (
                    mu_df.merge(n_df, on="c", how="left")
                         .merge(g_df, on="k", how="left")
                )
                mu_not["mu_not_ck"] = (mu_not["g_k"] * N_total - mu_not["mu_ck"] * mu_not["n_c"]) / (
                    (N_total - mu_not["n_c"]).replace(0, pd.NA)
                )
                mu_not = mu_not[["c","k","mu_not_ck"]].dropna()

                # register μ tables
                conn.register("mu_src", mu_df)
                conn.register("mu_not_src", mu_not)
                conn.execute("CREATE TEMP TABLE mu_tbl AS SELECT * FROM mu_src;")
                conn.execute("CREATE TEMP TABLE mu_not_tbl AS SELECT * FROM mu_not_src;")

                # 2nd pass: per chunk compute dot products with fixed μ tables
                all_chunks = []
                for start, end in _time_windows(train_df[time_col], chunk_by):
                    conn.execute(f"""
                        CREATE TEMP TABLE this_train AS
                        SELECT CAST(row_id AS BIGINT) AS row_id,
                               CAST({entity_col} AS VARCHAR) AS entity,
                               CAST({time_col}  AS TIMESTAMP) AS t_row,
                               CAST({label_col} AS VARCHAR)   AS y
                        FROM train_df
                        WHERE {time_col} IS NOT NULL
                          AND {time_col} >= ?
                          AND {time_col} <  ?;
                    """, [start.to_pydatetime(), end.to_pydatetime()])

                    SCORES_SQL = f"""
                    WITH row_dist AS ({ROW_DIST_SQL}),
                    score_intra AS (
                      SELECT rd.row_id, SUM(rd.d * m.mu_ck) AS s_intra
                      FROM row_dist rd
                      JOIN mu_tbl m ON m.c = rd.c AND m.k = rd.k
                      GROUP BY rd.row_id
                    ),
                    score_inter AS (
                      SELECT rd.row_id, SUM(rd.d * mn.mu_not_ck) AS s_inter
                      FROM row_dist rd
                      JOIN mu_not_tbl mn ON mn.c = rd.c AND mn.k = rd.k
                      GROUP BY rd.row_id
                    )
                    SELECT si.row_id, si.s_intra, so.s_inter,
                           CASE WHEN si.s_intra >= so.s_inter THEN 1.0 ELSE 0.0 END AS win
                    FROM score_intra si
                    JOIN score_inter so USING (row_id);
                    """
                    df_chunk = conn.execute(SCORES_SQL).df()
                    if len(df_chunk):
                        all_chunks.append(df_chunk)
                    conn.execute("DROP TABLE this_train;")

                scores_df = pd.concat(all_chunks, ignore_index=True) if all_chunks else \
                            pd.DataFrame(columns=["row_id","s_intra","s_inter","win"])
                if len(scores_df):
                    scores_df = scores_df.sort_values("row_id", kind="stable").reset_index(drop=True)

        # ---- aggregate metric
        h_agg = float(scores_df["win"].mean()) if len(scores_df) else float("nan")

        return (h_agg, scores_df) if return_per_row else h_agg

    finally:
        # best-effort cleanup
        try:
            for tbl in [
                "this_train","train_df","rel_df","rel_first","rel_first_use",
                "key_card","mu_tbl","mu_not_tbl"
            ]:
                conn.execute(f"DROP TABLE IF EXISTS {tbl};")
        except Exception:
            pass
        if close_when_done:
            try:
                conn.close()
            except Exception:
                pass




# ---------------------------------------------------------------------
# 1) Fast degree & homophily (classification or regression similarity)
# ---------------------------------------------------------------------
def compute_degree_and_homophily_regression_duckdb(
    train_task_table: pd.DataFrame,
    relation_pairs_df: pd.DataFrame,
    entity_col: str,
    label_col: str,
    time_col: str,
    conn: Optional[ddb.DuckDBPyConnection] = None,
    *,
    task: Literal["classification", "regression"] = "classification",
    reg_metric: Literal["eps", "gaussian"] = "gaussian",
    eps: Optional[float] = None,
    sigma: Optional[float] = None,

    # --- optional performance knobs (safe defaults) ---
    threads: Optional[int] = None,            # e.g., os.cpu_count()
    memory_limit: Optional[str|int] = None,   # "16GB" or 16 (GB) or bytes
    temp_directory: Optional[str] = None,     # "/mnt/ssd/tmp"
    max_key_fanout: Optional[int] = None,     # drop keys with > this many participants
    chunk_by: Optional[str] = None,           # pandas offset alias: "MS","W","QS"
) -> pd.DataFrame:
    """
    Returns DataFrame aligned to train_task_table.index with columns ['degree','homophily'].

    task='classification':
        homophily = mean( 1[neighbor_label == current_label] ) over neighbors with known labels.
    task='regression':
        - reg_metric='eps': mean( 1[|neighbor_label - current_label| <= eps] )
        - reg_metric='gaussian': mean( exp(- (Δy)^2 / (2σ^2)) )
    """
    # --- validate
    need = {entity_col, time_col, label_col}
    if not need.issubset(train_task_table.columns):
        raise ValueError(f"train_task_table must contain {need}")
    if not {"entity", "key", "time"}.issubset(relation_pairs_df.columns):
        raise ValueError("relation_pairs_df must have columns ['entity','key','time']")

    orig_index = train_task_table.index

    # --- prep pandas
    train_df = train_task_table[[entity_col, time_col, label_col]].copy()
    train_df = train_df.reset_index(drop=False).rename(columns={"index": "row_id"})
    train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce").dt.tz_localize(None)
    train_df = train_df[train_df[time_col].notna()]

    rel_df = relation_pairs_df[["entity", "key", "time"]].copy()
    rel_df["time"] = pd.to_datetime(rel_df["time"], errors="coerce").dt.tz_localize(None)
    rel_df = rel_df[rel_df["time"].notna()]

    if len(train_df) == 0:
        out = pd.DataFrame({"degree": 0, "homophily": np.nan}, index=orig_index)
        out["degree"] = out["degree"].astype(int)
        return out

    # --- defaults for regression
    if task == "regression":
        y_num = pd.to_numeric(train_df[label_col], errors="coerce")
        if sigma is None or not (isinstance(sigma, (int, float)) and sigma > 0):
            s = float(np.nanstd(y_num.values)) if y_num.notna().sum() else None
            sigma = s if s and s > 0 else 1.0
        if reg_metric == "eps":
            if eps is None or not (isinstance(eps, (int, float)) and eps > 0):
                eps = 0.5 * float(sigma)
        # freeze to plain float for SQL literal
        sigma = float(sigma)
        eps = float(eps) if eps is not None else None

    # --- connect & PRAGMAs
    close_when_done = False
    if conn is None:
        conn = ddb.connect()
        close_when_done = True

    try:
        if threads is not None:
            if not isinstance(threads, int) or threads <= 0:
                raise ValueError("threads must be a positive integer")
            conn.execute(f"PRAGMA threads={threads};")
        if memory_limit is not None:
            if isinstance(memory_limit, int):
                mem_str = str(memory_limit) if memory_limit >= 1_000_000_000 else f"{memory_limit}GB"
            else:
                mem_str = str(memory_limit)
            conn.execute(f"PRAGMA memory_limit='{mem_str}';")
        if temp_directory:
            conn.execute(f"PRAGMA temp_directory='{temp_directory}';")
        conn.execute("PRAGMA enable_object_cache=true;")

        # materialize inputs
        conn.register("train_src", train_df)
        conn.register("rel_src", rel_df)
        conn.execute("CREATE TEMP TABLE train_df AS SELECT * FROM train_src;")
        conn.execute("CREATE TEMP TABLE rel_df   AS SELECT * FROM rel_src;")

        # (entity,key) first appearance to collapse fanout
        conn.execute("""
            CREATE TEMP TABLE rel_first AS
            SELECT CAST(entity AS VARCHAR) AS entity,
                   key,
                   CAST(MIN(time) AS TIMESTAMP) AS first_time
            FROM rel_df
            WHERE entity IS NOT NULL AND key IS NOT NULL
            GROUP BY 1,2;
        """)

        if max_key_fanout is not None:
            conn.execute("""
                CREATE TEMP TABLE key_card AS
                SELECT key, COUNT(*)::BIGINT AS n_entities
                FROM rel_first
                GROUP BY 1;
            """)
            conn.execute("""
                CREATE TEMP TABLE rel_first_use AS
                SELECT rf.*
                FROM rel_first rf
                JOIN key_card kc USING (key)
                WHERE kc.n_entities <= ?;
            """, [int(max_key_fanout)])
        else:
            conn.execute("CREATE TEMP TABLE rel_first_use AS SELECT * FROM rel_first;")

        # --- similarity expression
        if task == "classification":
            label_cast = f"CAST({label_col} AS VARCHAR) AS current_label"
            sim_expr = """
                CASE
                  WHEN nl.neighbor_label IS NULL THEN NULL
                  WHEN nl.neighbor_label = n.current_label THEN 1.0
                  ELSE 0.0
                END
            """
            train_dedup_cast = "any_value(y) AS y"
            this_label_cast = "CAST({label_col} AS VARCHAR)"
        else:
            label_cast = f"CAST({label_col} AS DOUBLE) AS current_label"
            if reg_metric == "eps":
                sim_expr = f"""
                    CASE
                      WHEN nl.neighbor_label IS NULL THEN NULL
                      WHEN ABS(nl.neighbor_label - n.current_label) <= {eps} THEN 1.0
                      ELSE 0.0
                    END
                """
            elif reg_metric == "gaussian":
                sim_expr = f"""
                    CASE
                      WHEN nl.neighbor_label IS NULL THEN NULL
                      ELSE EXP(-POWER(nl.neighbor_label - n.current_label, 2) / (2*{sigma}*{sigma}))
                    END
                """
            else:
                raise ValueError("reg_metric must be 'eps' or 'gaussian'")
            train_dedup_cast = "any_value(y) AS y"
            this_label_cast = "CAST({label_col} AS DOUBLE)"

        # --- core per-chunk query
        MAIN_SQL = f"""
        WITH
        tk AS (
          SELECT tr.row_id, tr.entity AS u, tr.t_row, tr.current_label, rf.key
          FROM this_train tr
          JOIN rel_first_use rf
            ON rf.entity = tr.entity
           AND rf.first_time <= tr.t_row
        ),
        candidates AS (
          SELECT tk.row_id, CAST(r2.entity AS VARCHAR) AS neighbor_entity, tk.t_row, tk.current_label
          FROM tk
          JOIN rel_first_use r2
            ON r2.key = tk.key
           AND r2.first_time <= tk.t_row
          WHERE r2.entity <> tk.u
        ),
        neighbors_dedup AS (
          SELECT row_id, neighbor_entity,
                 any_value(t_row) AS t_row,
                 any_value(current_label) AS current_label
          FROM candidates
          GROUP BY row_id, neighbor_entity
        ),
        train_dedup AS (
          SELECT entity, t_row, {train_dedup_cast}
          FROM this_train
          GROUP BY entity, t_row
          ORDER BY entity, t_row
        ),
        neighbor_labels AS (
          SELECT n.row_id, n.neighbor_entity, t.y AS neighbor_label
          FROM neighbors_dedup n
          ASOF JOIN (SELECT entity, t_row, y FROM train_dedup ORDER BY entity, t_row) t
            ON n.neighbor_entity = t.entity AND n.t_row >= t.t_row
        ),
        degree AS (
          SELECT row_id, COUNT(*)::BIGINT AS degree
          FROM neighbors_dedup
          GROUP BY row_id
        ),
        homophily AS (
          SELECT n.row_id, AVG({sim_expr}) AS homophily
          FROM neighbors_dedup n
          LEFT JOIN neighbor_labels nl
            ON nl.row_id = n.row_id
           AND nl.neighbor_entity = n.neighbor_entity
          GROUP BY n.row_id
        )
        SELECT tr.row_id,
               COALESCE(d.degree, 0) AS degree,
               h.homophily
        FROM this_train tr
        LEFT JOIN degree d USING (row_id)
        LEFT JOIN homophily h USING (row_id);
        """

        # chunk windows
        if chunk_by is None:
            conn.execute(f"""
                CREATE TEMP TABLE this_train AS
                SELECT CAST(row_id AS BIGINT) AS row_id,
                       CAST({entity_col} AS VARCHAR)  AS entity,
                       CAST({time_col}  AS TIMESTAMP) AS t_row,
                       {this_label_cast}              AS y,
                       {label_cast}
                FROM train_df
                WHERE {time_col} IS NOT NULL;
            """)
            out = conn.execute(MAIN_SQL).df()
            conn.execute("DROP TABLE this_train;")
        else:
            results = []
            for start, end in _time_windows(train_df[time_col], chunk_by):
                conn.execute(f"""
                    CREATE TEMP TABLE this_train AS
                    SELECT CAST(row_id AS BIGINT) AS row_id,
                           CAST({entity_col} AS VARCHAR)  AS entity,
                           CAST({time_col}  AS TIMESTAMP) AS t_row,
                           {this_label_cast}              AS y,
                           {label_cast}
                    FROM train_df
                    WHERE {time_col} IS NOT NULL
                      AND {time_col} >= ?
                      AND {time_col} <  ?;
                """, [start.to_pydatetime(), end.to_pydatetime()])
                df_chunk = conn.execute(MAIN_SQL).df()
                if len(df_chunk):
                    results.append(df_chunk)
                conn.execute("DROP TABLE this_train;")
            out = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["row_id","degree","homophily"])

        # align to original index
        out = out.set_index("row_id").reindex(range(len(orig_index)))
        out.index = orig_index
        out["degree"] = out["degree"].fillna(0).astype(int)
        return out

    finally:
        try:
            for tname in ["this_train","train_df","rel_df","rel_first","rel_first_use","key_card"]:
                conn.execute(f"DROP TABLE IF EXISTS {tname};")
        except Exception:
            pass
        if close_when_done:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------
# 2) Fast aggregation homophily (classification or regression via bins)
#     -> returns mean win-rate and (optionally) per-row scores
# ---------------------------------------------------------------------
def aggregation_homophily_regression_duckdb(
    train_task_table: pd.DataFrame,
    relation_pairs_df: pd.DataFrame,
    *,
    entity_col: str,
    label_col: str,
    time_col: str,
    min_degree: int = 1,
    return_per_row: bool = True,
    conn: Optional[ddb.DuckDBPyConnection] = None,

    # --- behavior knobs ---
    task: Literal["classification", "regression"] = "classification",
    num_bins: int = 20,               # regression only
    binning: Literal["uniform", "quantile"] = "uniform",
    sigma: Optional[float] = None,    # regression Gaussian bandwidth (for smoothing over bins)

    # --- performance knobs (optional) ---
    threads: Optional[int] = None,
    memory_limit: Optional[str|int] = None,
    temp_directory: Optional[str] = None,
    max_key_fanout: Optional[int] = None,
    chunk_by: Optional[str] = None,   # enables two-pass accumulation
) -> tuple[float, pd.DataFrame] | float:
    """
    Aggregation homophily with temporal masking.

    For each row, build neighbor label distribution d(row,k) (categories for classification; smoothed
    histogram over value bins for regression), then compare ⟨d, μ_intra⟩ vs ⟨d, μ_out⟩.
    Returns mean 1[⟨d, μ_intra⟩ ≥ ⟨d, μ_out⟩]; optionally per-row scores.
    """
    # --- validate
    need = {entity_col, time_col, label_col}
    if not need.issubset(train_task_table.columns):
        raise ValueError(f"train_task_table must contain {need}")
    if not {"entity","key","time"}.issubset(relation_pairs_df.columns):
        raise ValueError("relation_pairs_df must have columns: ['entity','key','time']")

    # --- prep pandas
    train_df = train_task_table[[entity_col, time_col, label_col]].copy()
    train_df = train_df.reset_index(drop=False).rename(columns={"index": "row_id"})
    train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce").dt.tz_localize(None)
    train_df = train_df[train_df[time_col].notna()]

    rel_df = relation_pairs_df[["entity","key","time"]].copy()
    rel_df["time"] = pd.to_datetime(rel_df["time"], errors="coerce").dt.tz_localize(None)
    rel_df = rel_df[rel_df["time"].notna()]

    if len(train_df) == 0:
        empty = pd.DataFrame(columns=["row_id","s_intra","s_inter","win"])
        return (float("nan"), empty) if return_per_row else float("nan")

    # --- build bins for regression
    bins_df = None
    if task == "regression":
        y_num = pd.to_numeric(train_df[label_col], errors="coerce")
        y_min = float(np.nanmin(y_num.values)) if y_num.notna().sum() else 0.0
        y_max = float(np.nanmax(y_num.values)) if y_num.notna().sum() else 1.0
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min, y_max = y_min - 0.5, y_max + 0.5

        if binning == "quantile" and y_num.notna().sum():
            qs = np.linspace(0, 1, num_bins + 1)
            edges = np.unique(np.quantile(y_num.dropna().values, qs))
            if len(edges) < 2:
                edges = np.linspace(y_min, y_max, num_bins + 1)
        else:
            edges = np.linspace(y_min, y_max, num_bins + 1)

        edges = np.asarray(edges, dtype=float)
        if len(edges) != num_bins + 1:
            edges = np.linspace(edges.min(), edges.max(), num_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bins_df = pd.DataFrame({
            "k": np.arange(num_bins, dtype=int),
            "left": edges[:-1],
            "right": edges[1:],
            "center": centers,
            "is_last": [False]*(num_bins-1) + [True],
        })

        if sigma is None or not (isinstance(sigma, (int, float)) and sigma > 0):
            s = float(np.nanstd(y_num.values)) if y_num.notna().sum() else None
            sigma = s if s and s > 0 else (y_max - y_min) / max(4.0, num_bins)
        sigma = float(sigma)

    # --- connect & PRAGMAs
    close_when_done = False
    if conn is None:
        conn = ddb.connect()
        close_when_done = True

    try:
        if threads is not None:
            if not isinstance(threads, int) or threads <= 0:
                raise ValueError("threads must be a positive integer")
            conn.execute(f"PRAGMA threads={threads};")
        if memory_limit is not None:
            if isinstance(memory_limit, int):
                mem_str = str(memory_limit) if memory_limit >= 1_000_000_000 else f"{memory_limit}GB"
            else:
                mem_str = str(memory_limit)
            conn.execute(f"PRAGMA memory_limit='{mem_str}';")
        if temp_directory:
            conn.execute(f"PRAGMA temp_directory='{temp_directory}';")
        conn.execute("PRAGMA enable_object_cache=true;")

        # materialize inputs
        conn.register("train_src", train_df)
        conn.register("rel_src", rel_df)
        conn.execute("CREATE TEMP TABLE train_df AS SELECT * FROM train_src;")
        conn.execute("CREATE TEMP TABLE rel_df   AS SELECT * FROM rel_src;")

        # bins table for regression
        if task == "regression":
            conn.register("bins_src", bins_df)
            conn.execute("CREATE TEMP TABLE bins AS SELECT * FROM bins_src;")

        # (entity,key) first appearance
        conn.execute("""
            CREATE TEMP TABLE rel_first AS
            SELECT CAST(entity AS VARCHAR) AS entity,
                   key,
                   CAST(MIN(time) AS TIMESTAMP) AS first_time
            FROM rel_df
            WHERE entity IS NOT NULL AND key IS NOT NULL
            GROUP BY 1,2;
        """)

        if max_key_fanout is not None:
            conn.execute("""
                CREATE TEMP TABLE key_card AS
                SELECT key, COUNT(*)::BIGINT AS n_entities
                FROM rel_first
                GROUP BY 1;
            """)
            conn.execute("""
                CREATE TEMP TABLE rel_first_use AS
                SELECT rf.*
                FROM rel_first rf
                JOIN key_card kc USING (key)
                WHERE kc.n_entities <= ?;
            """, [int(max_key_fanout)])
        else:
            conn.execute("CREATE TEMP TABLE rel_first_use AS SELECT * FROM rel_first;")

        # --- ROW_DIST builder (classification vs regression)
        if task == "classification":
            # row_dist(row_id, c, k, d) with c = y_u, k = neighbor label (string)
            ROW_DIST_SQL = f"""
            WITH
            tk AS (
              SELECT tr.row_id, tr.entity AS u, tr.t_row, tr.y AS y_u, rf.key
              FROM this_train tr
              JOIN rel_first_use rf
                ON rf.entity = tr.entity
               AND rf.first_time <= tr.t_row
            ),
            candidates AS (
              SELECT tk.row_id, CAST(r2.entity AS VARCHAR) AS neighbor_entity, tk.t_row, tk.y_u
              FROM tk
              JOIN rel_first_use r2
                ON r2.key = tk.key
               AND r2.first_time <= tk.t_row
              WHERE r2.entity <> tk.u
            ),
            neighbors_dedup AS (
              SELECT row_id, neighbor_entity,
                     any_value(t_row) AS t_row,
                     any_value(y_u)   AS y_u
              FROM candidates
              GROUP BY row_id, neighbor_entity
            ),
            train_dedup AS (
              SELECT entity, t_row, any_value(y) AS y
              FROM this_train
              GROUP BY entity, t_row
              ORDER BY entity, t_row
            ),
            neighbor_labels AS (
              SELECT n.row_id, n.y_u, t.y AS y_nbr
              FROM neighbors_dedup n
              ASOF JOIN (SELECT entity, t_row, y FROM train_dedup ORDER BY entity, t_row) t
                ON n.neighbor_entity = t.entity AND n.t_row >= t.t_row
            ),
            neighbor_counts AS (
              SELECT row_id, y_u AS c, y_nbr AS k, COUNT(*)::BIGINT AS cnt
              FROM neighbor_labels
              WHERE y_nbr IS NOT NULL
              GROUP BY row_id, c, k
            ),
            row_degree AS (
              SELECT row_id, c, SUM(cnt)::DOUBLE AS deg
              FROM neighbor_counts
              GROUP BY row_id, c
            )
            SELECT nc.row_id, nc.c, nc.k, nc.cnt / NULLIF(rd.deg,0) AS d
            FROM neighbor_counts nc
            JOIN row_degree rd USING (row_id, c)
            WHERE rd.deg >= {min_degree}
            """
            THIS_LABEL_CAST = "CAST({label_col} AS VARCHAR) AS y"

        else:
            # regression: smoothed histogram over bins (k = bin id), c = bin(y_u)
            ROW_DIST_SQL = f"""
            WITH
            tk AS (
              SELECT tr.row_id, tr.entity AS u, tr.t_row, tr.y AS y_u, rf.key
              FROM this_train tr
              JOIN rel_first_use rf
                ON rf.entity = tr.entity
               AND rf.first_time <= tr.t_row
            ),
            candidates AS (
              SELECT tk.row_id, CAST(r2.entity AS VARCHAR) AS neighbor_entity, tk.t_row, tk.y_u
              FROM tk
              JOIN rel_first_use r2
                ON r2.key = tk.key
               AND r2.first_time <= tk.t_row
              WHERE r2.entity <> tk.u
            ),
            neighbors_dedup AS (
              SELECT row_id, neighbor_entity,
                     any_value(t_row) AS t_row,
                     any_value(y_u)   AS y_u
              FROM candidates
              GROUP BY row_id, neighbor_entity
            ),
            train_dedup AS (
              SELECT entity, t_row, any_value(y) AS y
              FROM this_train
              GROUP BY entity, t_row
              ORDER BY entity, t_row
            ),
            neighbor_labels AS (
              SELECT n.row_id, n.y_u, t.y AS y_nbr
              FROM neighbors_dedup n
              ASOF JOIN (SELECT entity, t_row, y FROM train_dedup ORDER BY entity, t_row) t
                ON n.neighbor_entity = t.entity AND n.t_row >= t.t_row
            ),
            row_degree AS (
              SELECT row_id, y_u, COUNT(*)::BIGINT AS deg
              FROM neighbor_labels
              WHERE y_nbr IS NOT NULL
              GROUP BY row_id, y_u
            ),
            neighbor_weights AS (
              SELECT
                nl.row_id,
                nl.y_u,
                b.k,
                SUM(EXP(-POWER(nl.y_nbr - b.center, 2) / (2*{sigma}*{sigma}))) AS w
              FROM neighbor_labels nl
              JOIN bins b ON TRUE
              WHERE nl.y_nbr IS NOT NULL
              GROUP BY nl.row_id, nl.y_u, b.k
            ),
            row_weight_sum AS (
              SELECT row_id, y_u, SUM(w) AS wsum
              FROM neighbor_weights
              GROUP BY row_id, y_u
            ),
            row_dist_raw AS (
              SELECT nw.row_id, nw.y_u, nw.k, nw.w / NULLIF(rws.wsum, 0) AS d
              FROM neighbor_weights nw
              JOIN row_weight_sum rws USING (row_id, y_u)
              JOIN row_degree rd USING (row_id, y_u)
              WHERE rd.deg >= {min_degree}
            ),
            row_class AS (
              SELECT DISTINCT r.row_id, b.k AS c
              FROM row_dist_raw r
              JOIN bins b
                ON r.y_u >= b.left AND (r.y_u < b.right OR b.is_last)
            )
            SELECT r.row_id, rc.c, r.k, r.d
            FROM row_dist_raw r
            JOIN row_class rc ON rc.row_id = r.row_id
            """
            THIS_LABEL_CAST = "CAST({label_col} AS DOUBLE) AS y"

        # --- single-pass or two-pass
        if chunk_by is None:
            # build this_train for all rows
            conn.execute(f"""
                CREATE TEMP TABLE this_train AS
                SELECT CAST(row_id AS BIGINT) AS row_id,
                       CAST({entity_col} AS VARCHAR)  AS entity,
                       CAST({time_col}  AS TIMESTAMP) AS t_row,
                       {THIS_LABEL_CAST}
                FROM train_df
                WHERE {time_col} IS NOT NULL;
            """)
            MAIN_SQL = f"""
            WITH row_dist AS ({ROW_DIST_SQL}),
            class_mean AS (
              SELECT c, k, AVG(d) AS mu_ck, COUNT(DISTINCT row_id) AS n_c
              FROM row_dist
              GROUP BY c, k
            ),
            global_mean AS (
              SELECT k, AVG(d) AS g_k, COUNT(DISTINCT row_id) AS N
              FROM row_dist
              GROUP BY k
            ),
            class_out AS (
              SELECT cm.c, cm.k,
                     (gm.g_k * gm.N - cm.mu_ck * cm.n_c) / NULLIF(gm.N - cm.n_c, 0) AS mu_not_ck
              FROM class_mean cm
              JOIN global_mean gm USING (k)
            ),
            score_intra AS (
              SELECT rd.row_id, SUM(rd.d * cm.mu_ck) AS s_intra
              FROM row_dist rd
              JOIN class_mean cm ON cm.c = rd.c AND cm.k = rd.k
              GROUP BY rd.row_id
            ),
            score_inter AS (
              SELECT rd.row_id, SUM(rd.d * co.mu_not_ck) AS s_inter
              FROM row_dist rd
              JOIN class_out co ON co.c = rd.c AND co.k = rd.k
              GROUP BY rd.row_id
            )
            SELECT si.row_id, si.s_intra, so.s_inter,
                   CASE WHEN si.s_intra >= so.s_inter THEN 1.0 ELSE 0.0 END AS win
            FROM score_intra si
            JOIN score_inter so USING (row_id);
            """
            scores_df = conn.execute(MAIN_SQL).df()
            if len(scores_df):
                scores_df = scores_df.sort_values("row_id", kind="stable").reset_index(drop=True)

        else:
            # -------- two-pass accumulation across chunks --------
            class_sum: Dict[tuple, float] = {}  # (c,k) -> sum d
            class_count: Dict[str, int] = {}    # c -> #rows
            global_sum: Dict[str, float] = {}   # k -> sum d
            N_total = 0

            # pass 1: accumulate μ stats
            for start, end in _time_windows(train_df[time_col], chunk_by):
                conn.execute(f"""
                    CREATE TEMP TABLE this_train AS
                    SELECT CAST(row_id AS BIGINT) AS row_id,
                           CAST({entity_col} AS VARCHAR)  AS entity,
                           CAST({time_col}  AS TIMESTAMP) AS t_row,
                           {THIS_LABEL_CAST}
                    FROM train_df
                    WHERE {time_col} IS NOT NULL
                      AND {time_col} >= ?
                      AND {time_col} <  ?;
                """, [start.to_pydatetime(), end.to_pydatetime()])

                rd = conn.execute(f"WITH row_dist AS ({ROW_DIST_SQL}) SELECT * FROM row_dist;").df()
                if not rd.empty:
                    # per-class distinct rows
                    n_c = rd[["row_id","c"]].drop_duplicates().groupby("c", as_index=False).size()
                    for _, r in n_c.iterrows():
                        c = str(r["c"])
                        class_count[c] = class_count.get(c, 0) + int(r["size"])

                    # class (c,k) sums
                    agg_ck = rd.groupby(["c","k"], as_index=False)["d"].sum()
                    for _, r in agg_ck.iterrows():
                        key = (str(r["c"]), str(r["k"]))
                        class_sum[key] = class_sum.get(key, 0.0) + float(r["d"])

                    # global sums
                    agg_k = rd.groupby("k", as_index=False)["d"].sum()
                    for _, r in agg_k.iterrows():
                        k = str(r["k"])
                        global_sum[k] = global_sum.get(k, 0.0) + float(r["d"])

                    N_total += rd["row_id"].nunique()

                conn.execute("DROP TABLE this_train;")

            if N_total == 0:
                scores_df = pd.DataFrame(columns=["row_id","s_intra","s_inter","win"])
            else:
                mu_rows = [(c, k, s / class_count[c])
                           for (c, k), s in class_sum.items()
                           if class_count.get(c, 0) > 0]
                mu_df = pd.DataFrame(mu_rows, columns=["c","k","mu_ck"])

                g_rows = [(k, s / N_total) for k, s in global_sum.items()]
                g_df = pd.DataFrame(g_rows, columns=["k","g_k"])

                n_df = pd.DataFrame([(c, n) for c, n in class_count.items()], columns=["c","n_c"])

                mu_not = (mu_df.merge(n_df, on="c", how="left")
                               .merge(g_df, on="k", how="left"))
                denom = (N_total - mu_not["n_c"]).replace(0, np.nan)
                mu_not["mu_not_ck"] = (mu_not["g_k"] * N_total - mu_not["mu_ck"] * mu_not["n_c"]) / denom
                mu_not = mu_not.dropna(subset=["mu_not_ck"])[["c","k","mu_not_ck"]]

                # register fixed μ tables
                conn.register("mu_src", mu_df)
                conn.register("mu_not_src", mu_not)
                conn.execute("CREATE TEMP TABLE mu_tbl AS SELECT * FROM mu_src;")
                conn.execute("CREATE TEMP TABLE mu_not_tbl AS SELECT * FROM mu_not_src;")

                # pass 2: per-chunk row scores with fixed μ
                chunks = []
                for start, end in _time_windows(train_df[time_col], chunk_by):
                    conn.execute(f"""
                        CREATE TEMP TABLE this_train AS
                        SELECT CAST(row_id AS BIGINT) AS row_id,
                               CAST({entity_col} AS VARCHAR)  AS entity,
                               CAST({time_col}  AS TIMESTAMP) AS t_row,
                               {THIS_LABEL_CAST}
                        FROM train_df
                        WHERE {time_col} IS NOT NULL
                          AND {time_col} >= ?
                          AND {time_col} <  ?;
                    """, [start.to_pydatetime(), end.to_pydatetime()])

                    SCORES_SQL = f"""
                    WITH row_dist AS ({ROW_DIST_SQL}),
                    score_intra AS (
                      SELECT rd.row_id, SUM(rd.d * m.mu_ck) AS s_intra
                      FROM row_dist rd
                      JOIN mu_tbl m ON m.c = rd.c AND m.k = rd.k
                      GROUP BY rd.row_id
                    ),
                    score_inter AS (
                      SELECT rd.row_id, SUM(rd.d * mn.mu_not_ck) AS s_inter
                      FROM row_dist rd
                      JOIN mu_not_tbl mn ON mn.c = rd.c AND mn.k = rd.k
                      GROUP BY rd.row_id
                    )
                    SELECT si.row_id, si.s_intra, so.s_inter,
                           CASE WHEN si.s_intra >= so.s_inter THEN 1.0 ELSE 0.0 END AS win
                    FROM score_intra si
                    JOIN score_inter so USING (row_id);
                    """
                    df_chunk = conn.execute(SCORES_SQL).df()
                    if len(df_chunk):
                        chunks.append(df_chunk)
                    conn.execute("DROP TABLE this_train;")

                scores_df = pd.concat(chunks, ignore_index=True) if chunks else \
                            pd.DataFrame(columns=["row_id","s_intra","s_inter","win"])
                if len(scores_df):
                    scores_df = scores_df.sort_values("row_id", kind="stable").reset_index(drop=True)

        # final metric
        h_agg = float(scores_df["win"].mean()) if len(scores_df) else float("nan")
        return (h_agg, scores_df) if return_per_row else h_agg

    finally:
        try:
            for tname in [
                "this_train","train_df","rel_df","rel_first","rel_first_use","key_card",
                "bins","mu_tbl","mu_not_tbl"
            ]:
                conn.execute(f"DROP TABLE IF EXISTS {tname};")
        except Exception:
            pass
        if close_when_done:
            try:
                conn.close()
            except Exception:
                pass



def expand_gaussian_range(similarities, method='power', alpha=2.0, beta=1.0):
    """
    Transform Gaussian similarities to expand their range when they are concentrated near 0.
    
    Args:
        similarities: numpy array of Gaussian similarities (0-1 range)
        method: transformation method - 'power', 'sigmoid', 'log', 'sqrt', 'custom'
        alpha: parameter for power transformation (default=2.0)
        beta: parameter for sigmoid transformation (default=1.0)
    
    Returns:
        Transformed similarities with expanded range
    """
    if method == 'power':
        # Power transformation: x^alpha
        # alpha > 1: expands values near 0, compresses values near 1
        # alpha < 1: compresses values near 0, expands values near 1
        return np.power(similarities, alpha)
    
    elif method == 'sigmoid':
        # Sigmoid-like transformation: 1 / (1 + exp(-beta * (x - 0.5)))
        # Shifts the sigmoid center and adjusts steepness
        return 1 / (1 + np.exp(-beta * (similarities - 0.5)))
    
    elif method == 'log':
        # Logarithmic transformation: log(1 + x) / log(2)
        # Expands small values, compresses large values
        return np.log1p(similarities) / np.log(2)
    
    elif method == 'sqrt':
        # Square root transformation: sqrt(x)
        # Expands small values, compresses large values
        return np.sqrt(similarities)
    
    elif method == 'custom':
        # Custom transformation: x^2 * (2 - x)
        # Expands mid-range values while preserving 0 and 1
        return similarities ** 2 * (2 - similarities)
    
    elif method == 'tanh':
        # Hyperbolic tangent transformation: tanh(beta * x)
        # Expands small values with adjustable steepness
        return np.tanh(beta * similarities)
    
    elif method == 'exp':
        # Exponential transformation: (exp(alpha * x) - 1) / (exp(alpha) - 1)
        # Strongly expands small values
        return (np.exp(alpha * similarities) - 1) / (np.exp(alpha) - 1)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: power, sigmoid, log, sqrt, custom, tanh, exp")

def compute_entity_key_degree_duckdb(
    train_task_table: pd.DataFrame,
    relation_pairs_df: pd.DataFrame,
    *,
    entity_col: str,          # in train_task_table
    time_col: str,            # in train_task_table
    conn: Optional[ddb.DuckDBPyConnection] = None,
    threads: Optional[int] = None,
    memory_limit: Optional[str | int] = None,
    temp_directory: Optional[str] = None,
    chunk_by: Optional[str] = None,   # e.g., "MS", "W", "QS"
) -> pd.DataFrame:
    """
    Degree along the heterogeneous edge type entity->key:
        ek_degree(u, t) = # distinct keys K such that (u, K) first appears at time <= t.

    Inputs
    ------
    train_task_table: DataFrame with [entity_col, time_col] — evaluation rows (u, t_row)
    relation_pairs_df: DataFrame with ['entity','key','time'] — participation events

    Returns
    -------
    DataFrame aligned to train_task_table.index with one column:
        - ek_degree : int
    """
    # validate
    need_train = {entity_col, time_col}
    if not need_train.issubset(train_task_table.columns):
        raise ValueError(f"train_task_table must contain {need_train}.")
    if not {"entity", "key", "time"}.issubset(relation_pairs_df.columns):
        raise ValueError("relation_pairs_df must have columns ['entity','key','time'].")

    orig_index = train_task_table.index

    # normalize & stage
    train_df = train_task_table[[entity_col, time_col]].copy()
    train_df = train_df.reset_index(drop=False).rename(columns={"index": "row_id"})
    train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce").dt.tz_localize(None)

    rel_df = relation_pairs_df[["entity", "key", "time"]].copy()
    rel_df["time"] = pd.to_datetime(rel_df["time"], errors="coerce").dt.tz_localize(None)

    # drop null times
    train_df = train_df[train_df[time_col].notna()]
    rel_df   = rel_df[rel_df["time"].notna()]

    # early return if no valid rows
    if len(train_df) == 0:
        out = pd.DataFrame({"ek_degree": 0}, index=orig_index)
        out["ek_degree"] = out["ek_degree"].astype(int)
        return out

    close_when_done = False
    if conn is None:
        conn = ddb.connect()
        close_when_done = True

    try:
        # PRAGMAs
        if threads is not None:
            if not isinstance(threads, int) or threads <= 0:
                raise ValueError("threads must be a positive integer")
            conn.execute(f"PRAGMA threads={threads};")
        if memory_limit is not None:
            if isinstance(memory_limit, int):
                mem_str = str(memory_limit) if memory_limit >= 1_000_000_000 else f"{memory_limit}GB"
            else:
                mem_str = str(memory_limit)
            conn.execute(f"PRAGMA memory_limit='{mem_str}';")
        if temp_directory is not None:
            conn.execute(f"PRAGMA temp_directory='{temp_directory}';")
        conn.execute("PRAGMA enable_object_cache=true;")

        # register & materialize
        conn.register("train_src", train_df)
        conn.register("rel_src", rel_df)
        conn.execute("CREATE TEMP TABLE train_df AS SELECT * FROM train_src;")
        conn.execute("CREATE TEMP TABLE rel_df   AS SELECT * FROM rel_src;")

        # first appearance per (entity,key)
        conn.execute("""
            CREATE TEMP TABLE rel_first AS
            SELECT CAST(entity AS VARCHAR) AS entity,
                   key,
                   CAST(MIN(time) AS TIMESTAMP) AS first_time
            FROM rel_df
            WHERE entity IS NOT NULL AND key IS NOT NULL
            GROUP BY 1,2;
        """)

        # chunk windows
        windows = (
            [(_min_time(train_df[time_col]), _max_time(train_df[time_col]) + pd.Timedelta(microseconds=1))]
            if chunk_by is None else list(_time_windows(train_df[time_col], chunk_by))
        )

        results = []
        DEG_EK_QUERY = f"""
            SELECT
              tr.row_id,
              COALESCE(COUNT(DISTINCT rf.key), 0)::BIGINT AS ek_degree
            FROM this_train tr
            LEFT JOIN rel_first rf
              ON rf.entity = tr.entity
             AND rf.first_time <= tr.t_row
            GROUP BY tr.row_id
            ORDER BY tr.row_id;
        """

        for (start, end) in windows:
            if chunk_by is None:
                conn.execute(f"""
                    CREATE TEMP TABLE this_train AS
                    SELECT
                      CAST(row_id AS BIGINT) AS row_id,
                      CAST({entity_col} AS VARCHAR) AS entity,
                      CAST({time_col}  AS TIMESTAMP) AS t_row
                    FROM train_df
                    WHERE {time_col} IS NOT NULL;
                """)
            else:
                conn.execute(f"""
                    CREATE TEMP TABLE this_train AS
                    SELECT
                      CAST(row_id AS BIGINT) AS row_id,
                      CAST({entity_col} AS VARCHAR) AS entity,
                      CAST({time_col}  AS TIMESTAMP) AS t_row
                    FROM train_df
                    WHERE {time_col} IS NOT NULL
                      AND {time_col} >= ?
                      AND {time_col} <  ?;
                """, [start.to_pydatetime(), end.to_pydatetime()])

            n_rows = conn.execute("SELECT COUNT(*) FROM this_train;").fetchone()[0]
            if n_rows == 0:
                conn.execute("DROP TABLE this_train;")
                if chunk_by is None:
                    break
                else:
                    continue

            df_chunk = conn.execute(DEG_EK_QUERY).df()
            results.append(df_chunk)

            conn.execute("DROP TABLE this_train;")
            if chunk_by is None:
                break

        out = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["row_id","ek_degree"])
        out = out.set_index("row_id").reindex(range(len(orig_index)))
        out.index = orig_index
        out["ek_degree"] = out["ek_degree"].fillna(0).astype(int)
        return out

    finally:
        # cleanup
        try:
            conn.execute("DROP TABLE IF EXISTS this_train;")
            conn.execute("DROP TABLE IF EXISTS train_df;")
            conn.execute("DROP TABLE IF EXISTS rel_df;")
            conn.execute("DROP TABLE IF EXISTS rel_first;")
        except Exception:
            pass
        if close_when_done:
            try: conn.close()
            except Exception: pass


def compute_lag_autocorr(
    df: pd.DataFrame,
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    label_col: str = "y",
    lags: Sequence[int] = (1, 2, 3),
    task_type: Literal["classification", "regression"] = "classification",
    use_gaussian_kernel: bool = False,
    sigma: Optional[float] = None,
) -> Dict[int, float]:
    """
    Compute lag-specific label 'autocorrelation' across entities.

    For each lag k in {1, 2, 3}, this computes the similarity between labels at time t
    and labels at time t-k, for the same entity. This measures how "stable" or "predictable"
    labels are over specific time lags.

    Args:
        df: DataFrame with columns [entity_col, time_col, label_col]
        entity_col: Column name for entity ID (e.g., "driver_id", "article_id")
        time_col: Column name for timestamp (datetime or numeric)
        label_col: Column name for label/target
        lags: Sequence of lag values to compute (e.g., (1, 2, 3))
        task_type: "classification" or "regression"
        use_gaussian_kernel: For regression, whether to use Gaussian kernel similarity
        sigma: For Gaussian kernel, scale parameter; if None, uses median absolute difference

    Returns:
        Dictionary mapping lag -> autocorrelation value
        - For classification: equality rate (fraction of times y_t == y_{t-k})
        - For regression: Pearson correlation or Gaussian kernel similarity

    Example:
        >>> # Classification
        >>> lag_feats_cls = compute_lag_autocorr(
        ...     df_labels, entity_col="driver_id",
        ...     time_col="date", label_col="y",
        ...     lags=(1, 2, 3), task_type="classification"
        ... )
        >>> # -> {1: 0.82, 2: 0.74, 3: 0.70}

        >>> # Regression with Gaussian kernel
        >>> lag_feats_reg = compute_lag_autocorr(
        ...     df_labels, entity_col="article_id",
        ...     time_col="timestamp", label_col="sales",
        ...     lags=(1, 2, 3), task_type="regression",
        ...     use_gaussian_kernel=True
        ... )
    """
    # 1. sort by entity, then time
    df_sorted = df.sort_values([entity_col, time_col]).copy()

    results: Dict[int, float] = {}

    for k in lags:
        # 2. shift labels within each entity
        df_sorted[f"{label_col}_lag{k}"] = (
            df_sorted.groupby(entity_col)[label_col].shift(k)
        )

        # 3. drop rows where lagged value is missing
        mask = df_sorted[f"{label_col}_lag{k}"].notna()
        cur = df_sorted.loc[mask, label_col].to_numpy()
        prev = df_sorted.loc[mask, f"{label_col}_lag{k}"].to_numpy()

        if len(cur) == 0:
            results[k] = np.nan
            continue

        if task_type == "classification":
            # assume integer / categorical labels; equality rate is the cleanest
            eq_rate = (cur == prev).mean()
            results[k] = float(eq_rate)

        elif task_type == "regression":
            if use_gaussian_kernel:
                # if sigma not given, set a data-driven scale
                if sigma is None:
                    # median absolute difference as scale
                    diffs = np.abs(cur - prev)
                    med = np.median(diffs)
                    # avoid sigma=0
                    eff_sigma = med if med > 0 else (np.std(cur) + 1e-8)
                else:
                    eff_sigma = sigma
                sim = np.exp(-((cur - prev) ** 2) / (2.0 * eff_sigma**2))
                results[k] = float(sim.mean())
            else:
                # plain Pearson correlation
                if np.std(cur) == 0 or np.std(prev) == 0:
                    results[k] = np.nan
                else:
                    corr = np.corrcoef(cur, prev)[0, 1]
                    results[k] = float(corr)
        else:
            raise ValueError(f"Unknown task_type={task_type}")

    return results