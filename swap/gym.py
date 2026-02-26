import os
from utils.retrieve_search_data import WandBAnalyzer
from utils.information import rdl_cared_columns
import numpy as np
import pandas as pd
try:
    from scipy.stats import kendalltau
except Exception:
    kendalltau = None
from .regression import load_full_homophily_features, load_full_autotransfer_features, load_full_heuristics_features, load_model_task_embedding
import scipy.stats as stats
import matplotlib.pyplot as plt


def _freeze(v):
    """Make values hashable (lists/ndarrays -> tuple)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return tuple(np.asarray(v).tolist())
    return v


def _select_best_by_signature(df: pd.DataFrame, signature_cols, treat_na_equal: bool) -> pd.DataFrame:
    """
    Keep the best row for each (dataset, task, signature) combination.

    Selection is based on VALIDATION performance (to simulate real model selection),
    but the returned row includes the TEST metric for evaluation.
    """
    if df.empty:
        return df.copy()

    missing = [col for col in signature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} for signature computation.")

    scored = df.copy()
    if 'task_type' not in scored.columns:
        raise ValueError("Expected 'task_type' column to determine metric direction.")

    # Check if val_metric exists, fallback to test_metric if not available
    if 'val_metric' in scored.columns:
        selection_metric = 'val_metric'
    else:
        selection_metric = 'test_metric'
        import warnings
        warnings.warn(
            "Column 'val_metric' not found. Falling back to 'test_metric' for model selection. "
            "This may lead to overfitting in evaluation.",
            UserWarning
        )

    is_regression = scored['task_type'].eq('regression')
    # Score based on VALIDATION metric (higher is better for classification, lower for regression)
    scored['_score'] = np.where(is_regression, -scored[selection_metric], scored[selection_metric])

    # Group by (dataset, task, signature) and select the best checkpoint based on validation score
    group_keys = ['dataset_name', 'task_name'] + list(signature_cols)
    best_idx = scored.groupby(group_keys, dropna=False)['_score'].idxmax()
    best = scored.loc[best_idx].copy()
    best.drop(columns=['_score'], inplace=True)

    def build_signature(row):
        parts = []
        for col in signature_cols:
            value = row[col]
            if treat_na_equal and pd.isna(value):
                value = '__NA__'
            parts.append(_freeze(value))
        return tuple(parts)

    best['anchor_signature'] = best.apply(build_signature, axis=1)
    return best


def _filter_tasks_with_required_models(stacked: pd.DataFrame, required_models, min_required: int = None) -> pd.DataFrame:
    """
    Keep only (dataset, task) pairs that contain required model families.

    Args:
        stacked: DataFrame with anchor models
        required_models: Collection of model family names
        min_required: Minimum number of required models that must be present.
                     If None, ALL required_models must be present (strict mode).
                     If set to a number, at least that many models must be present.

    Examples:
        - required_models=('rdl', 'tabpfn', 'ft_transformer'), min_required=None
          → Keep only tasks with all 3 models (strictest)
        - required_models=('rdl', 'tabpfn', 'ft_transformer'), min_required=2
          → Keep tasks with at least 2 of the 3 models (more flexible)
    """
    if stacked.empty:
        return stacked

    pair_models = (
        stacked[['dataset_name', 'task_name', '__model_name__']]
        .drop_duplicates()
        .groupby(['dataset_name', 'task_name'])['__model_name__']
        .agg(set)
        .reset_index(name='model_set')
    )
    required = set(required_models)

    if min_required is None:
        # Strict mode: ALL required models must be present
        good_pairs = pair_models[pair_models['model_set'].apply(lambda s: required.issubset(s))]
    else:
        # Flexible mode: At least min_required models must be present
        good_pairs = pair_models[
            pair_models['model_set'].apply(lambda s: len(required.intersection(s)) >= min_required)
        ]

    if good_pairs.empty:
        return stacked.iloc[0:0]

    keep_index = pd.MultiIndex.from_frame(good_pairs[['dataset_name', 'task_name']])
    mask = stacked.set_index(['dataset_name', 'task_name']).index.isin(keep_index)
    return stacked.loc[mask].copy()


def _keep_signatures_common_to_all_tasks(stacked: pd.DataFrame) -> pd.DataFrame:
    """Ensure every remaining anchor signature appears in every task."""
    if stacked.empty:
        return stacked

    total_pairs = stacked[['dataset_name', 'task_name']].drop_duplicates().shape[0]
    if total_pairs == 0:
        return stacked.iloc[0:0]

    combos = (
        stacked[['__model_name__', 'anchor_signature', 'dataset_name', 'task_name']]
        .drop_duplicates()
    )
    counts = (
        combos.groupby(['__model_name__', 'anchor_signature'])
        .size()
        .reset_index(name='pair_count')
    )
    common = counts[counts['pair_count'].eq(total_pairs)][['__model_name__', 'anchor_signature']]
    return stacked.merge(common, on=['__model_name__', 'anchor_signature'], how='inner')


def _assign_anchor_ids_and_rank(stacked: pd.DataFrame, category_order) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign per-family and global anchor IDs and compute per-task ranks."""
    if stacked.empty:
        return stacked, stacked.iloc[0:0]

    out = stacked.copy()
    out['anchor_id_in_space'] = (
        out.groupby('__model_name__')['anchor_signature']
        .transform(lambda s: pd.factorize(s, sort=False)[0])
    )

    present = out['__model_name__'].unique().tolist()
    ordered = [c for c in category_order if c in present] + [c for c in present if c not in category_order]

    offsets = {}
    current = 0
    for fam in ordered:
        max_id = out.loc[out['__model_name__'].eq(fam), 'anchor_id_in_space'].max()
        max_id = int(max_id) if pd.notna(max_id) else -1
        offsets[fam] = current
        current += max_id + 1

    out['anchor_global_id'] = out['anchor_id_in_space'] + out['__model_name__'].map(offsets)

    out['rank_score'] = np.where(out['task_type'].eq('regression'),
                                 -out['test_metric'], out['test_metric'])
    out['rank'] = out.groupby(['dataset_name', 'task_name'])['rank_score'] \
        .rank(ascending=False, method='average')
    out['task_key'] = out['dataset_name'].astype(str) + '::' + out['task_name'].astype(str)

    rank_mat = (
        out.pivot_table(index='task_key', columns='anchor_global_id', values='rank', aggfunc='mean')
        .sort_index()
        .sort_index(axis=1)
    )

    return out, rank_mat


def _save_kendall_heatmap(sim: pd.DataFrame, prefix: str = 'kendall_similarity') -> None:
    """Best-effort heatmap export for debugging. Silently ignores plotting errors."""
    if sim.empty:
        return
    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt  # noqa
    except Exception:
        return

    data = sim.copy()
    np.fill_diagonal(data.values, np.nan)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(data, annot=False, fmt='.2f', cmap=cmap,
                xticklabels=False, yticklabels=False, cbar=False, ax=ax)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    try:
        plt.savefig(f'{prefix}.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{prefix}.pdf', bbox_inches='tight', pad_inches=0)
    finally:
        plt.close(fig)

def _kendall_pairwise_by_order(stacked: pd.DataFrame):
    """
    Task-wise similarity based on the ORDER of anchors:
    - For each task, sort anchors by (rank asc, rank_score desc, anchor_global_id asc)
    - Build a permutation position vector over the common anchor set
    - Compute Kendall tau between tasks using those position vectors
    Assumes: stacked has columns ['task_key','anchor_global_id','rank','rank_score'] and
    that all tasks share the same anchor set (enforced earlier).
    """
    # Anchor universe (common across tasks due to intersection step)
    anchors = np.sort(stacked['anchor_global_id'].unique())
    A = len(anchors)
    anchor_index = {a: i for i, a in enumerate(anchors)}

    # Build task → positions vector (length A), where positions are 0..A-1
    task_keys = stacked['task_key'].unique().tolist()
    pos_mat = np.empty((len(task_keys), A), dtype=np.int32)

    for ti, tk in enumerate(task_keys):
        df_t = stacked.loc[stacked['task_key'].eq(tk),
                           ['anchor_global_id','rank','rank_score']]
        # Stable, deterministic order: best first
        df_t = df_t.sort_values(
            by=['rank', 'rank_score', 'anchor_global_id'],
            ascending=[True, False, True]
        )
        # Now df_t['anchor_global_id'] is the ordered sequence for this task
        # Turn it into a position vector aligned to 'anchors'
        order = df_t['anchor_global_id'].to_numpy()
        # position of each anchor in this task's sequence
        pos_of = {a: i for i, a in enumerate(order)}
        # fill the vector in canonical anchor order
        pos_vec = np.fromiter((pos_of[a] for a in anchors), dtype=np.int32, count=A)
        pos_mat[ti, :] = pos_vec

    # Pairwise Kendall on position vectors
    sim = pd.DataFrame(np.eye(len(task_keys)), index=task_keys, columns=task_keys, dtype=float)
    for i in range(len(task_keys)):
        for j in range(i+1, len(task_keys)):
            if kendalltau is not None:
                tau, _ = kendalltau(pos_mat[i], pos_mat[j])
            else:
                # fallback via pandas corr on a temporary frame (same result for permutations)
                tmp = pd.DataFrame({'a': pos_mat[i], 'b': pos_mat[j]})
                tau = tmp['a'].corr(tmp['b'], method='kendall')
            sim.iat[i, j] = sim.iat[j, i] = tau
    return sim



# ---------------------------
# Main API
# ---------------------------

def get_anchor_models_and_task_embedding(
    df: pd.DataFrame,
    *,
    required_models = ('rdl', 'tabpfn', 'ft_transformer'),
    category_order = ('rdl', 'tabpfn', 'ft_transformer'),
    treat_na_equal: bool = True,
    anchor_mode_type: str = 'common',
    min_required_models: int = None
):
    """Build anchor tables per model family and derive task similarities.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment table containing at least dataset/task identifiers, model
        configuration columns, `test_metric`, and `task_type`.
    required_models : Iterable[str]
        Model families that must be present for every retained task.
    category_order : Iterable[str]
        Ordering used when assigning global anchor ids.
    treat_na_equal : bool
        When forming signatures, treat NaNs as a stable placeholder so they
        compare equal.
    anchor_mode_type : {'common', 'architecture'}
        The signature granularity. ``'common'`` keeps full configuration
        signatures (original behaviour). ``'architecture'`` collapses RDL to
        ``(pre_sf, mpnn_type)`` and DFS models to ``(dfs_layer,
        torch_frame_model_cls)``.
    min_required_models : int, optional
        Minimum number of required models that must be present per task.
        If None (default), ALL required_models must be present (strict mode).
        If set to a number, at least that many models from required_models must be present.
        Example: required_models=('rdl', 'tabpfn', 'ft_transformer'), min_required_models=2
        would keep tasks with at least 2 of the 3 models.

    Returns
    -------
    kendall_sim : pd.DataFrame
        Kendall τ similarity matrix between tasks based on anchor rankings.
    stacked : pd.DataFrame
        Anchor rows with per-space/global ids and per-task ranks.
    rank_mat : pd.DataFrame
        Task × anchor matrix of ranks (lower is better).
    rdl_anchors : pd.DataFrame
        RDL anchor table aligned with the filtered result set.
    """

    if anchor_mode_type not in {'common', 'architecture'}:
        raise ValueError("anchor_mode_type must be 'common' or 'architecture'")

    base_cols = {'dataset_name', 'task_name', 'test_metric', 'task_type'}
    missing_base = base_cols - set(df.columns)
    if missing_base:
        raise ValueError(f"Missing required columns: {sorted(missing_base)}")

    df = df.copy()
    df = df.dropna(subset=['dataset_name', 'task_name', 'test_metric', 'task_type'])
    if df.empty:
        raise ValueError("Input dataframe is empty after dropping NaNs.")

    # ------------------------------------------------------------------
    # Model specifications per anchor mode
    # -----------------------------------------------------------------
    # Normalize column values to lowercase for case-insensitive matching
    if 'model_type' in df.columns:
        df['model_type'] = df['model_type'].astype(str).str.lower()
    if 'torch_frame_model_cls' in df.columns:
        df['torch_frame_model_cls'] = df['torch_frame_model_cls'].astype(str).str.lower()

    model_specs = {
        'rdl': {
            'selector': lambda frame: frame[frame['model_type'].eq('rdl')],
            'signature': ['num_layers', 'pre_sf', 'mpnn_type'],
        },
        'tabpfn': {
            'selector': lambda frame: frame[frame['torch_frame_model_cls'].eq('tabpfn')],
            'signature': ['dfs_layer'],
        },
        'ft_transformer': {
            'selector': lambda frame: frame[frame['torch_frame_model_cls'].eq('ft_transformer')],
            'signature': ['num_layers'],
        },
    }

    anchors_by_family: dict[str, pd.DataFrame] = {}
    rdl_anchor_table = pd.DataFrame()

    for family, spec in model_specs.items():
        try:
            fam_df = spec['selector'](df)
        except KeyError as exc:
            raise ValueError(f"Column '{exc.args[0]}' is required to build anchors for '{family}'.") from exc
        if fam_df.empty:
            anchors_by_family[family] = fam_df.copy()
            continue

        best = _select_best_by_signature(fam_df, spec['signature'], treat_na_equal)
        best['__model_name__'] = family
        anchors_by_family[family] = best
        if family == 'rdl':
            rdl_anchor_table = best.copy()

    missing_families = [fam for fam in required_models if anchors_by_family.get(fam, pd.DataFrame()).empty]
    if missing_families:
        raise ValueError(f"Missing anchors for required families: {missing_families}")

    stacked = pd.concat(
        [anchors_by_family[fam] for fam in anchors_by_family if not anchors_by_family[fam].empty],
        ignore_index=True,
    )
    if stacked.empty:
        raise ValueError("No anchor rows available after initial selection.")

    stacked = _filter_tasks_with_required_models(stacked, required_models, min_required=min_required_models)
    if stacked.empty:
        raise ValueError("No tasks satisfy the required model coverage.")

    stacked = _keep_signatures_common_to_all_tasks(stacked)
    if stacked.empty:
        raise ValueError("No common anchors remain after intersection across tasks.")

    stacked, rank_mat = _assign_anchor_ids_and_rank(stacked, category_order)
    if stacked.empty or rank_mat.empty:
        raise ValueError("Failed to assign anchor ids or build rank matrix.")

    kendall_sim = _kendall_pairwise_by_order(stacked)
    _save_kendall_heatmap(kendall_sim)

    # Align the returned RDL table with the final set of signatures/tasks
    if not rdl_anchor_table.empty:
        keep_signatures = set(stacked.loc[stacked['__model_name__'].eq('rdl'), 'anchor_signature'])
        keep_pairs = set(zip(stacked.loc[stacked['__model_name__'].eq('rdl'), 'dataset_name'],
                             stacked.loc[stacked['__model_name__'].eq('rdl'), 'task_name']))
        if keep_signatures and keep_pairs:
            current_pairs = pd.Series(
                list(zip(rdl_anchor_table['dataset_name'], rdl_anchor_table['task_name'])),
                index=rdl_anchor_table.index,
            )
            mask = (
                rdl_anchor_table['anchor_signature'].isin(keep_signatures)
                & current_pairs.isin(keep_pairs)
            )
            rdl_anchor_table = rdl_anchor_table.loc[mask].copy()
        rdl_anchor_table.drop(columns=['anchor_signature', '__model_name__'], errors='ignore', inplace=True)

    return kendall_sim, stacked, rank_mat, rdl_anchor_table



# --- helpers ---

def _task_sim_from_anchor_ranks(rank_mat: pd.DataFrame) -> pd.DataFrame:
    """
    GraphGym-style similarity between tasks:
    Kendall τ between their ORDERINGS of anchors (best rank first).
    rank_mat: index=task_key, columns=anchor_global_id, values=ranks (smaller=better)
              (assumed same anchor set for all tasks; no NaNs)
    """
    tasks = rank_mat.index.tolist()
    anchors = rank_mat.columns.to_numpy()
    A = len(anchors)

    # build per-task permutation positions of anchors (0..A-1)
    pos = np.empty((len(tasks), A), dtype=np.int32)
    for i, t in enumerate(tasks):
        r = rank_mat.loc[t]
        # stable tie-break by anchor id
        order = r.sort_values(ascending=True, kind="mergesort").index.to_numpy()
        pos_of = {a: j for j, a in enumerate(order)}
        pos[i] = np.fromiter((pos_of[a] for a in anchors), dtype=np.int32, count=A)

    # pairwise Kendall on permutations
    S = np.eye(len(tasks), dtype=float)
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if kendalltau:
                tau, _ = kendalltau(pos[i], pos[j])
            else:
                tmp = pd.DataFrame({'a': pos[i], 'b': pos[j]})
                tau = tmp['a'].corr(tmp['b'], method='kendall')
            S[i, j] = S[j, i] = tau
    return pd.DataFrame(S, index=tasks, columns=tasks)

def _task_sim_from_embedding(emb: pd.DataFrame, metric: str = "cosine") -> pd.DataFrame:
    """
    task embedding similarity (cosine or dot).
    emb: index=task_key, columns=embedding dims
    """
    X = emb.to_numpy(dtype=float)
    if metric == "cosine":
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        X = X / n
        S = X @ X.T
    elif metric == "dot":
        S = X @ X.T
    else:
        raise ValueError("metric must be 'cosine' or 'dot'")
    return pd.DataFrame(S, index=emb.index, columns=emb.index)

def _kendall_between_similarity_rankings_orders(S1: pd.DataFrame, S2: pd.DataFrame) -> pd.Series:
    """
    For each center task i, compute Kendall τ between the **ranking orders of other tasks**
    induced by S1[i, -i] and S2[i, -i] (descending similarity). Uses stable tie-break by name.
    """
    # align tasks
    common = S1.index.intersection(S2.index)
    S1 = S1.loc[common, common].copy()
    S2 = S2.loc[common, common].copy()

    # drop self by setting diagonal to NaN (we will drop NaNs when ranking)
    np.fill_diagonal(S1.values, np.nan)
    np.fill_diagonal(S2.values, np.nan)

    taus = []
    for t in S1.index:
        a = S1.loc[t].dropna()
        b = S2.loc[t].dropna()

        # keep intersection of others
        others = a.index.intersection(b.index)
        if len(others) < 2:
            taus.append(np.nan)
            continue

        a = a.loc[others]
        b = b.loc[others]

        # ranking orders (desc), stable tie-break via mergesort + name
        a_ord = a.sort_values(ascending=False, kind="mergesort").index
        b_ord = b.sort_values(ascending=False, kind="mergesort").index

        # convert two orders into position vectors over the same label order
        base = sorted(others)  # deterministic base alignment
        pos_a = {k: i for i, k in enumerate(a_ord)}
        pos_b = {k: i for i, k in enumerate(b_ord)}
        va = np.fromiter((pos_a[k] for k in base), dtype=np.int32, count=len(base))
        vb = np.fromiter((pos_b[k] for k in base), dtype=np.int32, count=len(base))

        if kendalltau is not None:
            tau, _ = kendalltau(va, vb)
        else:
            tau = pd.Series(va).corr(pd.Series(vb), method="kendall")
        taus.append(tau)

    return pd.Series(taus, index=S1.index, name="kendall_corr")


def plot_combined_similarity_ranking_correlation(
    S_rank: pd.DataFrame,
    embedding_matrices: list[pd.DataFrame],
    embedding_names: list[str] = None,
    *,
    emb_metric: str = "cosine",
    figsize=(12, 8),
    fill_value: float | None = None,
    title: str = "Combined Similarity Ranking Correlation",
):
    """
    Create a single figure with all embedding-based similarities overlaid on the same plot.
    
    Args:
        S_rank: DataFrame with anchor ranking similarities
        embedding_matrices: List of DataFrames with embedding-based similarities
        embedding_names: List of names for each embedding matrix (optional)
        emb_metric: Metric for embedding similarity calculation
        figsize: Figure size
        fill_value: Value to fill NaN values with
        title: Title for the combined plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Default names if not provided
    if embedding_names is None:
        embedding_names = [f"Embedding_{i+1}" for i in range(len(embedding_matrices))]
    
    # Create single figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors for different embeddings
    colors = plt.cm.tab10(np.linspace(0, 1, len(embedding_matrices)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    results = {}
    
    # Plot each embedding matrix
    for i, (embedding_matrix, name) in enumerate(zip(embedding_matrices, embedding_names)):
        # Calculate Kendall correlation
        tau_by_task = _kendall_between_similarity_rankings_orders(S_rank, embedding_matrix)
        results[name] = tau_by_task
        
        # Calculate average correlation
        avg_corr = tau_by_task.mean()
        
        # Plot scatter points
        xs = np.arange(len(tau_by_task))
        ax.scatter(xs, tau_by_task.values, 
                  color=colors[i], 
                  marker=markers[i % len(markers)],
                  label=f'{name} (avg: {avg_corr:.3f})', 
                  alpha=0.7, 
                  s=60)
        
        # Add horizontal line for average
        ax.axhline(y=avg_corr, color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
    
    # Set up the plot
    ax.set_xticks(xs)
    ax.set_xticklabels(tau_by_task.index, rotation=90)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Kendall Corr (order vs order)")
    ax.set_xlabel("Tasks")
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add summary text box with average correlations
    avg_correlations = [results[name].mean() for name in embedding_names]
    summary_text = "Average Correlations:\n" + "\n".join([f"{name}: {avg:.3f}" for name, avg in zip(embedding_names, avg_correlations)])
    
    # Position text box in upper right corner
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10, fontfamily='monospace')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('combined_similarity_ranking_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def plot_similarity_ranking_correlation_right(
    S_rank: pd.DataFrame,
    task_embedding: pd.DataFrame,
    *,
    emb_metric: str = "cosine",
    ax: plt.Axes | None = None,
    figsize=(9, 4),
    title: str = "homophily_loo_Kendall",
    fill_value: float | None = None,   # e.g. 1.0 if you want to prefill NaNs off-diagonal
):
    """
    S_rank: task×task similarity (from your anchor-ranking GraphGym-style method).
    task_embedding: task×dim numeric features (index must be task keys).
    Computes Kendall τ between **task-wise ranking orders** from S_rank and from embedding similarity,
    and plots the per-task τ.
    """
    # optional NaN handling (usually not needed; self will be dropped anyway)
    if fill_value is not None:
        S_rank = S_rank.fillna(fill_value)
        task_embedding = task_embedding.fillna(fill_value)

    # per-task Kendall τ over ranking orders
    tau_by_task = _kendall_between_similarity_rankings_orders(S_rank, task_embedding)

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        save_individual = True
    else:
        save_individual = False
        
    xs = np.arange(len(tau_by_task))
    ax.scatter(xs, tau_by_task.values)
    ax.set_xticks(xs)
    ax.set_xticklabels(tau_by_task.index, rotation=90)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Kendall Corr (order vs order)")
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    
    # Only save individual files if not used as subplot
    if save_individual:
        plt.tight_layout()
        plt.savefig(title + ".png")
        plt.show()
    
    return tau_by_task

def build_task_embedding_from_heuristics(
    heuristics_df: pd.DataFrame,
    *,
    make_task_key=lambda r: f"{r['dataset_name']}::{r['task_name']}",
    standardize: str = "zscore",   # "zscore" | "robust" | None
    row_normalize: bool = True,    # good if you use cosine similarity
    rank_mat: pd.DataFrame | None = None,  # if given, align to intersection
):
    df = heuristics_df.copy()

    # 1) task index
    df["task_key"] = df.apply(make_task_key, axis=1)
    df = df.set_index("task_key")

    # 2) pick numeric feature columns (drop identifiers)
    drop_cols = {"dataset_name","task_name"}
    num_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    X = df[num_cols].copy()

    # 3) impute and drop zero-variance cols
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    var = X.var(axis=0)
    keep = var[var > 0].index
    X = X[keep]

    # 4) column scaling (so features contribute comparably)
    if standardize == "zscore":
        X = (X - X.mean(0)) / X.std(0).replace(0, 1)
    elif standardize == "robust":
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = (q3 - q1).replace(0, 1)
        X = (X - X.median(0)) / iqr

    # 5) optional row-wise normalization (cosine is then just dot)
    if row_normalize:
        norms = np.linalg.norm(X.values, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = pd.DataFrame(X.values / norms, index=X.index, columns=X.columns)

    # 6) align to rank_mat if provided
    if rank_mat is not None:
        common = X.index.intersection(rank_mat.index)
        X = X.loc[common]
    return X

def seaborn_plot_similarity_matrix(similarity_matrix: pd.DataFrame, name: str):
    ## make the left to right diagonal white and ignore it
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 5})
    np.fill_diagonal(similarity_matrix.values, np.nan)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad(color="white")
    sns.heatmap(similarity_matrix, annot=False, fmt='.2f', cmap=cmap)
    plt.savefig(name + '_similarity_matrix.png')
    plt.close()

if __name__ == "__main__":
    analyzer = WandBAnalyzer(entity=os.getenv("WANDB_ENTITY", "msuczk"))
    analyzer.check_and_download_datasets()
    dataframe = analyzer.combine_and_generate_a_large_dataframe()
    task_embeddings = load_model_task_embedding('./taskemb')
    homophily_df = load_full_homophily_features(task_embeddings)
    homophily_df = homophily_df[['dataset_name', 'task_name'] + [col for col in homophily_df.columns]]
    homophily_df.to_csv('homophily_df.csv', index=True, float_format="%.2f")