import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]

LABEL_ALIAS = {
    'kumorfm (fine-tuned)': 'kumorfm (ft)',
}

IGNORED_ROWS = ['RDL (val-selected)', 'DFS (val-selected)', 'Best Search']


def load_and_process_data(csv_path):
    """Load a results CSV whose first data row holds task types."""
    df = pd.read_csv(csv_path)
    task_types = df.iloc[0].to_dict()
    task_types.pop('Model', None)

    data_df = df.iloc[1:].copy()
    data_df.set_index('Model', inplace=True)
    data_df = data_df[~data_df.index.isin(IGNORED_ROWS)]
    data_df = data_df.apply(pd.to_numeric, errors='coerce')
    return data_df, task_types


def _rank_scores_with_ratio(scores, ascending, ratio):
    """
    Rank a series of scores, optionally merging nearby values when ratio > 0.
    The merge window is anchored at the 2nd best and 2nd worst scores to reduce
    the influence of extreme outliers.
    """
    valid_scores = scores.dropna()
    if valid_scores.empty:
        return pd.Series(index=scores.index, dtype=float)

    if ratio is None or ratio <= 0:
        return valid_scores.rank(ascending=ascending, method='average').reindex(scores.index)

    sorted_scores = valid_scores.sort_values(ascending=ascending)
    if len(sorted_scores) < 2:
        return valid_scores.rank(ascending=ascending, method='average').reindex(scores.index)

    if len(sorted_scores) >= 4:
        top2_value = sorted_scores.iloc[1]
        last2_value = sorted_scores.iloc[-2]
    else:
        top2_value = sorted_scores.iloc[0]
        last2_value = sorted_scores.iloc[-1]

    gap = abs(last2_value - top2_value)
    if gap == 0:
        return valid_scores.rank(ascending=ascending, method='min').reindex(scores.index)

    if ascending:
        band_min = top2_value
        band_max = top2_value + ratio * gap
    else:
        band_max = top2_value
        band_min = top2_value - ratio * gap

    if band_min > band_max:
        band_min, band_max = band_max, band_min

    in_band_mask = (sorted_scores >= band_min) & (sorted_scores <= band_max)
    if not in_band_mask.any():
        return valid_scores.rank(ascending=ascending, method='average').reindex(scores.index)

    values = sorted_scores.values
    indices = sorted_scores.index.tolist()
    n = len(sorted_scores)
    epsilon = 1e-12

    groups = []
    i = 0
    while i < n:
        val = values[i]
        idxs = [indices[i]]
        i += 1
        while i < n and abs(values[i] - val) <= epsilon:
            idxs.append(indices[i])
            i += 1
        groups.append((val, idxs))

    ranks_dict = {}
    current_rank = 1
    band_active = False
    band_indices = []

    def _flush_band():
        nonlocal current_rank, band_active, band_indices
        if band_active and band_indices:
            for idx in band_indices:
                ranks_dict[idx] = current_rank
            current_rank += len(band_indices)
            band_indices = []
            band_active = False

    for value, idxs in groups:
        in_band = (band_min - epsilon) <= value <= (band_max + epsilon)
        if in_band:
            band_active = True
            band_indices.extend(idxs)
        else:
            _flush_band()
            for idx in idxs:
                ranks_dict[idx] = current_rank
            current_rank += len(idxs)

    _flush_band()

    rank_series = pd.Series(ranks_dict, dtype=float)
    return rank_series.reindex(scores.index)


def calculate_average_rank(data_df, task_types, task_filter=None, ratio=None):
    """
    Calculate the average rank for a subset of tasks.
    task_filter: None (all), 'classification', or 'regression'.
    """
    tasks_to_rank = list(task_types.keys())
    if task_filter:
        tasks_to_rank = [t for t, tt in task_types.items() if tt == task_filter]

    if not tasks_to_rank:
        return pd.Series(dtype=float)

    ranks = pd.DataFrame(index=data_df.index)
    for task in tasks_to_rank:
        task_type = task_types[task]
        if task in data_df.columns:
            ascending = task_type != 'classification'
            ranks[task] = _rank_scores_with_ratio(data_df[task], ascending, ratio)

    return ranks.mean(axis=1)


def create_plot(csv_path, output_dir, ratio=None):
    """Create a 2-panel plot showing classification and regression rank comparisons."""
    data_df, task_types = load_and_process_data(csv_path)

    class_rank = calculate_average_rank(
        data_df, task_types, task_filter='classification', ratio=ratio
    ).sort_values(ascending=True)

    reg_rank = calculate_average_rank(
        data_df, task_types, task_filter='regression', ratio=ratio
    ).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    axes[0].bar(range(len(class_rank)), class_rank.values, color=COLORS[:len(class_rank)])
    axes[0].set_xticks(range(len(class_rank)))
    axes[0].set_xticklabels([LABEL_ALIAS.get(l, l) for l in class_rank.index])
    axes[0].set_title('Classification Rank ↓', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Average Rank (Lower is Better)', fontsize=16)
    axes[0].set_ylim(1, len(class_rank) + 1)

    axes[1].bar(range(len(reg_rank)), reg_rank.values, color=COLORS[:len(reg_rank)])
    axes[1].set_xticks(range(len(reg_rank)))
    axes[1].set_xticklabels([LABEL_ALIAS.get(l, l) for l in reg_rank.index])
    axes[1].set_title('Regression Rank ↓', fontsize=16, fontweight='bold')
    axes[1].set_ylim(1, len(reg_rank) + 1)

    for ax in axes:
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.invert_yaxis()
        ax.tick_params(axis='x', rotation=45, labelsize=20)
        ax.tick_params(axis='y', labelsize=18)
        for label in ax.get_xticklabels():
            label.set_ha('right')

    plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15, wspace=0.3)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'result31.png'), bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(output_dir, 'result31.pdf'), bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved plots to {output_dir}/result31.png and {output_dir}/result31.pdf")


if __name__ == "__main__":
    csv_path = "results/result31.csv"
    output_dir = "plot"
    create_plot(csv_path, output_dir)
