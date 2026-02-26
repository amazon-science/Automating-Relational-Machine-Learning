"""Analysis and ranking utilities for experiment trial DataFrames.

Pure DataFrame operations — no WandB/MongoDB dependencies.
"""
import pandas as pd
from typing import Optional, Literal, Iterable


BetterIs = Literal["val", "test"]
TaskType = Literal[
    "auto_binary_classification", "binary_classification", "regression", "recommendation"
]

_DEFAULT_ALLOWED: tuple[TaskType, ...] = (
    "auto_binary_classification",
    "binary_classification",
    "regression",
)


def _metric_col(metric: BetterIs) -> str:
    if metric in ("val", "val_metric"):
        return "val_metric"
    if metric in ("test", "test_metric"):
        return "test_metric"
    raise ValueError("metric must be 'val' or 'test'")


def _categorize_method(model_type: str) -> str:
    mt = (model_type or "").strip().lower()
    if mt == "dfs":
        return "dfs"
    if mt == "rdl":
        return "rdl"
    return "other"


def _is_regression_group(g: pd.DataFrame) -> bool:
    """True only if ALL rows in the group have task_type='regression'."""
    ttypes = set(map(str, g["task_type"].astype(str)))
    return ttypes == {"regression"}


def result_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate statistics for each task showing run counts by model type.

    Returns a DataFrame with columns: dataset_name, task_name, rdl_runs,
    dfs_runs, total_runs, plus one column per torch_frame_model_cls.
    """
    basic_counts = (
        df.groupby(["dataset_name", "task_name", "model_type"])
        .size()
        .reset_index(name="count")
    )

    pivot_basic = basic_counts.pivot_table(
        index=["dataset_name", "task_name"],
        columns="model_type",
        values="count",
        fill_value=0,
    ).reset_index()

    if "rdl" in pivot_basic.columns:
        pivot_basic.rename(columns={"rdl": "rdl_runs"}, inplace=True)
    else:
        pivot_basic["rdl_runs"] = 0

    if "dfs" in pivot_basic.columns:
        pivot_basic.rename(columns={"dfs": "dfs_runs"}, inplace=True)
    else:
        pivot_basic["dfs_runs"] = 0

    pivot_basic["total_runs"] = pivot_basic["rdl_runs"] + pivot_basic["dfs_runs"]

    if "torch_frame_model_cls" in df.columns:
        dfs_df = df[df["model_type"] == "dfs"].copy()
        dfs_df = dfs_df[dfs_df["torch_frame_model_cls"].notna()]

        if not dfs_df.empty:
            model_cls_counts = (
                dfs_df.groupby(["dataset_name", "task_name", "torch_frame_model_cls"])
                .size()
                .reset_index(name="count")
            )

            model_cls_dict = (
                model_cls_counts.groupby(["dataset_name", "task_name"])
                .apply(lambda x: dict(zip(x["torch_frame_model_cls"], x["count"])))
                .reset_index(name="torch_frame_model_cls_counts")
            )
            pivot_basic = pivot_basic.merge(
                model_cls_dict, on=["dataset_name", "task_name"], how="left"
            )

            model_cls_pivot = model_cls_counts.pivot_table(
                index=["dataset_name", "task_name"],
                columns="torch_frame_model_cls",
                values="count",
                fill_value=0,
            ).reset_index()

            model_cls_columns = {}
            for col in model_cls_pivot.columns:
                if col not in ["dataset_name", "task_name"]:
                    clean_name = str(col).lower().replace(" ", "_").replace("-", "_")
                    model_cls_columns[col] = f"{clean_name}_runs"
            model_cls_pivot.rename(columns=model_cls_columns, inplace=True)

            pivot_basic = pivot_basic.merge(
                model_cls_pivot, on=["dataset_name", "task_name"], how="left"
            )
            for col in model_cls_columns.values():
                if col in pivot_basic.columns:
                    pivot_basic[col] = pivot_basic[col].fillna(0).astype(int)
        else:
            pivot_basic["torch_frame_model_cls_counts"] = None
    else:
        pivot_basic["torch_frame_model_cls_counts"] = None

    pivot_basic["rdl_runs"] = pivot_basic["rdl_runs"].astype(int)
    pivot_basic["dfs_runs"] = pivot_basic["dfs_runs"].astype(int)
    pivot_basic["total_runs"] = pivot_basic["total_runs"].astype(int)

    return pivot_basic.sort_values(["dataset_name", "task_name"]).reset_index(drop=True)


def add_taskwise_rank(
    df: pd.DataFrame,
    *,
    metric: BetterIs = "val",
    allowed_task_types: Optional[Iterable[TaskType]] = _DEFAULT_ALLOWED,
    by_category: bool = False,
    rank_col: str = "rank",
    rank_method: Literal["dense", "min", "max", "first"] = "dense",
) -> pd.DataFrame:
    """
    Return a copy of *df* with a ``method_category`` column and an integer-like
    *rank_col* giving the within-task rank based on the chosen metric.

    For regression tasks ranks are ascending (smaller is better); for all
    other task types ranks are descending (larger is better).
    """
    metric_col = _metric_col(metric)
    if metric_col not in df.columns:
        raise KeyError(f"Expected column '{metric_col}' not in dataframe")

    need_cols = {"dataset_name", "task_name", "task_type", metric_col, "model_type"}
    missing = need_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    if allowed_task_types is not None:
        out = out[out["task_type"].isin(allowed_task_types)].copy()

    out["method_category"] = out["model_type"].astype(str).map(_categorize_method)

    group_cols = ["dataset_name", "task_name"] + (["method_category"] if by_category else [])

    def _rank_one(g: pd.DataFrame) -> pd.DataFrame:
        has_metric = g[metric_col].notna()
        g_ok = g[has_metric].copy()
        ascending = _is_regression_group(g)
        g_ok[rank_col] = g_ok[metric_col].rank(
            ascending=ascending, method=rank_method, na_option="bottom"
        )
        g_bad = g[~has_metric].copy()
        if not g_bad.empty:
            g_bad[rank_col] = float("inf")
        return pd.concat([g_ok, g_bad], axis=0)

    ranked = (
        out.groupby(group_cols, group_keys=False, sort=False)
        .apply(_rank_one)
        .sort_values(group_cols + [rank_col], kind="mergesort")
        .reset_index(drop=True)
    )
    return ranked


def summarize_best_trials(
    df: pd.DataFrame,
    *,
    metric: BetterIs = "val",
    allowed_task_types: Optional[Iterable[TaskType]] = _DEFAULT_ALLOWED,
    by_category: bool = False,
    tie_strategy: Literal["all", "first", "top_k"] = "all",
    top_k: int = 1,
    include_rank: bool = True,
    filter_mode_type: Optional[Literal["rdl", "dfs", "other"]] = None,
) -> pd.DataFrame:
    """Return the best trial(s) per task, chosen by *metric* with correct direction."""
    metric_col = _metric_col(metric)
    if filter_mode_type:
        df = df[df["model_type"] == filter_mode_type]
    ranked = add_taskwise_rank(
        df,
        metric=metric,
        allowed_task_types=allowed_task_types,
        by_category=by_category,
        rank_col="rank",
        rank_method="dense",
    )

    group_cols = ["dataset_name", "task_name"] + (["method_category"] if by_category else [])

    def _best(g: pd.DataFrame) -> pd.DataFrame:
        g = g[g[metric_col].notna()].copy()
        if g.empty:
            return g
        ascending = _is_regression_group(g)
        g_sorted = g.sort_values(
            by=[metric_col, "model_type"],
            ascending=[ascending, True],
            kind="mergesort",
        )
        if tie_strategy == "all":
            best_val = g_sorted[metric_col].iloc[0]
            return g_sorted[g_sorted[metric_col] == best_val]
        elif tie_strategy == "first":
            return g_sorted.head(1)
        elif tie_strategy == "top_k":
            return g_sorted.head(top_k)
        else:
            raise ValueError("tie_strategy must be 'all', 'first', or 'top_k'")

    best = (
        ranked.groupby(group_cols, group_keys=True, sort=False)
        .apply(_best)
        .reset_index(drop=True)
    )
    if not include_rank:
        best = best.drop(columns=["rank"], errors="ignore")
    return best


def get_best_k_trials(
    df: pd.DataFrame,
    k: int,
    *,
    metric: BetterIs = "val",
    allowed_task_types: Optional[Iterable[TaskType]] = _DEFAULT_ALLOWED,
    by_category: bool = False,
    include_rank: bool = True,
    filter_mode_type: Optional[Literal["rdl", "dfs", "other"]] = None,
) -> pd.DataFrame:
    """Return the top *k* trials per task, chosen by *metric* with correct direction."""
    metric_col = _metric_col(metric)
    if filter_mode_type:
        df = df[df["model_type"] == filter_mode_type]

    ranked = add_taskwise_rank(
        df,
        metric=metric,
        allowed_task_types=allowed_task_types,
        by_category=by_category,
        rank_col="rank",
        rank_method="dense",
    )

    group_cols = ["dataset_name", "task_name"] + (["method_category"] if by_category else [])

    def _best_k(g: pd.DataFrame) -> pd.DataFrame:
        g = g[g[metric_col].notna()].copy()
        if g.empty:
            return g
        ascending = _is_regression_group(g)
        g_sorted = g.sort_values(
            by=[metric_col, "model_type"],
            ascending=[ascending, True],
            kind="mergesort",
        )
        return g_sorted.head(k)

    best_k = (
        ranked.groupby(group_cols, group_keys=True, sort=False)
        .apply(_best_k)
        .reset_index(drop=True)
    )
    if not include_rank:
        best_k = best_k.drop(columns=["rank"], errors="ignore")
    return best_k


def summarize_successful_runs(
    df: pd.DataFrame,
    *,
    metric: BetterIs = "val",
    allowed_task_types: Optional[Iterable[TaskType]] = _DEFAULT_ALLOWED,
    include_other: bool = False,
) -> pd.DataFrame:
    """Count successful runs (non-NaN metric) per task, split by method category."""
    metric_col = _metric_col(metric)
    need_cols = {"dataset_name", "task_name", "model_type", "task_type", metric_col}
    missing = need_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    working = df.copy()
    if allowed_task_types is not None:
        working = working[working["task_type"].isin(allowed_task_types)].copy()

    working = working[working[metric_col].notna()].copy()
    if working.empty:
        base_cols = ["dataset_name", "task_name", "rdl_runs", "dfs_runs"]
        if include_other:
            base_cols.append("other_runs")
        base_cols.append("total_runs")
        return pd.DataFrame(columns=base_cols)

    working["method_category"] = working["model_type"].astype(str).map(_categorize_method)
    counts = (
        working.groupby(["dataset_name", "task_name", "method_category"], dropna=False)
        .size()
        .reset_index(name="num_runs")
    )

    pivot = counts.pivot_table(
        index=["dataset_name", "task_name"],
        columns="method_category",
        values="num_runs",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.rename(columns={"rdl": "rdl_runs", "dfs": "dfs_runs", "other": "other_runs"})

    for col in ("rdl_runs", "dfs_runs", "other_runs"):
        if col not in pivot.columns:
            pivot[col] = 0
        pivot[col] = pivot[col].astype(int)

    keep_cols = ["dataset_name", "task_name", "rdl_runs", "dfs_runs"]
    if include_other:
        keep_cols.append("other_runs")

    pivot = pivot.reset_index()
    pivot["total_runs"] = pivot[
        [c for c in ("rdl_runs", "dfs_runs", "other_runs") if c in keep_cols]
    ].sum(axis=1)
    keep_cols.append("total_runs")

    return pivot[keep_cols].sort_values(["dataset_name", "task_name"]).reset_index(drop=True)
