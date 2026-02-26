"""Experiment 3: HPO comparison (Random / TPE / Hyperband vs knowledge-transfer)."""

import argparse
import os

import numpy as np
import torch

from swap.hpo import TaskEmbeddingHPO
from utils.hpo import (
    build_experiment3_catalog,
    create_objective_function,
    aggregate_seed_runs,
    aggregate_seed_runs_landscape,
    aggregate_seed_runs_llm,
    print_trial_details,
)
from utils.information import EVAL_TASK_LIST, CLASSIFICATION_TASK_LIST
from utils.type import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_task_type(task_name):
    """Return task type string based on classification task list."""
    if task_name in CLASSIFICATION_TASK_LIST:
        return 'binary_classification'
    return 'regression'


def _process_seed_runs(seed_runs, task_type, debug_trials, db_name, task_name,
                       algo_key, budget, landscape_topk, use_landscape,
                       landscape_details, row_label,
                       use_llm_select=False, llm_backend='anthropic',
                       llm_model='claude-sonnet-4-5-20250929'):
    """Process seed runs and return formatted score string.

    Shared logic for both normal and transfer method loops.
    Also populates *landscape_details* in-place when landscape scoring is on.
    """
    if debug_trials:
        print_trial_details(seed_runs, db_name, task_name, algo_key, budget)

    # Disable post-hoc landscape selection for small budgets — too few
    # candidates make landscape voting unreliable.
    if budget < 10:
        use_landscape = False
        use_llm_select = False

    top_k = 1 if budget < landscape_topk else min(landscape_topk, budget)

    if use_llm_select:
        agg_mean, agg_std, voted_scores, extra = aggregate_seed_runs_llm(
            seed_runs, task_type, top_k=top_k,
            llm_backend=llm_backend, llm_model=llm_model,
        )
        val_scores = extra.get('validation_scores', []) if isinstance(extra, dict) else []
        best_test_scores = extra.get('best_test_scores', []) if isinstance(extra, dict) else []
        selections = extra.get('selections', []) if isinstance(extra, dict) else []
        landscape_details[(row_label, db_name, task_name)] = {
            'voted_scores': voted_scores,
            'validation_scores': val_scores,
            'best_test_scores': best_test_scores,
            'selections': selections,
        }
        mean_score, std_score = agg_mean, agg_std
    elif use_landscape:
        agg_mean, agg_std, voted_scores, extra = aggregate_seed_runs_landscape(
            seed_runs, task_type, top_k=top_k, meta_for_landscape=use_landscape,
        )
        val_scores = extra.get('validation_scores', []) if isinstance(extra, dict) else []
        best_test_scores = extra.get('best_test_scores', []) if isinstance(extra, dict) else []
        selections = extra.get('selections', []) if isinstance(extra, dict) else []
        landscape_details[(row_label, db_name, task_name)] = {
            'voted_scores': voted_scores,
            'validation_scores': val_scores,
            'best_test_scores': best_test_scores,
            'selections': selections,
        }
        mean_score, std_score = agg_mean, agg_std
    else:
        mean_score, std_score, _ = aggregate_seed_runs(seed_runs, task_type)

    if mean_score is None or std_score is None:
        return None

    if use_landscape or use_llm_select:
        label = "llm" if use_llm_select else "vote"
        parts = [f"{label} {mean_score:.4f} ± {std_score:.4f}"]
        if val_scores:
            val_arr = np.array(val_scores, dtype=float)
            parts.append(f"val {val_arr.mean():.4f} ± {val_arr.std(ddof=0):.4f}")
        if best_test_scores:
            test_arr = np.array(best_test_scores, dtype=float)
            parts.append(f"test {test_arr.mean():.4f} ± {test_arr.std(ddof=0):.4f}")
        return ' | '.join(parts)

    return f"{mean_score:.4f} ± {std_score:.4f}"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment_3(output_csv_path='exp3_results.csv',
                     seeds_list=[42, 0, 2],
                     budgets_normal=(3, 30, 180),
                     budgets_transfer=(3, 30),
                     prior_strength=0.7,
                     version="baseline",
                     skip_unknown_params=False,
                     debug_trials=True,
                     meta_threshold=0.8,
                     landscape_topk=3,
                     use_meta_predictor=True,
                     meta_for_landscape=True,
                     include_tabpfn=False,
                     downstream_id=-1,
                     meta_classifier='lr',
                     parallel_trials=None,
                     parallel_devices=None,
                     skip_autotransfer=True,
                     only_show_prediction=False,
                     use_llm_select=False,
                     llm_backend='anthropic',
                     llm_model='claude-sonnet-4-5-20250929'):
    """Compare Random/TPE/Hyperband vs knowledge-transfer baselines.

    Seeds control *hyperparameter choice* only (training seed is fixed at 42
    inside ``swap.execution``).

    Args:
        debug_trials: Print detailed trial info per task.
        parallel_trials: Max concurrent trials per task.
        parallel_devices: Device pool (e.g. ``"cuda:0,cuda:1"``).
        landscape_topk: Top-k trials for landscape selection.
        meta_classifier: Branch predictor (``'lr'``, ``'rf'``, or ``'tabpfn'``).
        include_tabpfn: Add TabPFN to DFS family.
        skip_autotransfer: Skip loading autotransfer embeddings.

    Returns:
        ``(results_rows, trained_g_model, landscape_details_dict)``
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    landscape_topk = max(1, int(landscape_topk))

    hpo = TaskEmbeddingHPO(
        parallel_trials=parallel_trials,
        parallel_devices=parallel_devices,
        meta_classifier=meta_classifier,
        skip_autotransfer=skip_autotransfer,
    )
    allowed_dfs_models = ['ft_transformer']
    hpo.allowed_dfs_models = allowed_dfs_models

    def _space_for_task(task_type):
        return build_experiment3_catalog(allowed_dfs_models=allowed_dfs_models)

    def _objective_for(db, task, ttype):
        return create_objective_function(
            database_name=db, task_name=task, task_type=ttype,
            hpo_instance=hpo, skip_unknown_params=skip_unknown_params,
            use_meta_predictor=use_meta_predictor,
        )

    # ---- methods & budgets ------------------------------------------------
    if version == "baseline":
        transfer_methods = []
        normal_methods = [
            ('Random', 'random', budgets_normal),
            ('TPE', 'tpe', budgets_normal),
            ('Hyperband', 'hyperband', budgets_normal),
        ]
    elif version == "ours":
        transfer_methods = []
        normal_methods = [
            ('TPE', 'tpe', budgets_normal),
        ]
    else:
        normal_methods = []
        transfer_methods = [
            ('Autotransfer', 'autotransfer', budgets_transfer),
        ]

    col_name_map = {
        ('rel-f1', 'driver-position'): 'driver-position',
        ('rel-f1', 'driver-top3'): 'driver-top3',
        ('rel-hm', 'user-churn'): 'user-churn',
    }
    table_cols = [col_name_map.get(pair, pair[1]) for pair in EVAL_TASK_LIST]

    row_labels = []
    results_rows = []
    landscape_details = {}
    use_landscape = meta_for_landscape

    eval_task_list = EVAL_TASK_LIST if downstream_id == -1 else [EVAL_TASK_LIST[downstream_id]]

    # ---- iterate over method groups ---------------------------------------
    for methods in (normal_methods, transfer_methods):
        for display_name, algo_key, budgets in methods:
            for B in budgets:
                row_label = f"{display_name} ({B} trials)"
                row_vals = []
                for db_name, task_name in eval_task_list:
                    task_type = _infer_task_type(task_name)
                    space = _space_for_task(task_type)
                    objective = _objective_for(db_name, task_name, task_type)

                    seed_runs = hpo.optimize_for_task(
                        objective_fn=objective,
                        config_space=space,
                        database_name=db_name,
                        task_name=task_name,
                        task_type=task_type,
                        max_evals=B,
                        prior_strength=prior_strength,
                        task_embedding_type='homophily',
                        seeds_list=seeds_list,
                        algorithm=algo_key,
                        meta_threshold=meta_threshold,
                        landscape_topk=landscape_topk,
                        landscape_enable=use_landscape,
                        enable_meta_predictor=use_meta_predictor,
                        enable_meta_for_landscape=use_landscape,
                        skip_unknown_params=skip_unknown_params,
                        parallel_trials=parallel_trials,
                        parallel_devices=parallel_devices,
                    )

                    score_str = _process_seed_runs(
                        seed_runs, task_type, debug_trials, db_name, task_name,
                        algo_key, B, landscape_topk, use_landscape,
                        landscape_details, row_label,
                        use_llm_select=use_llm_select,
                        llm_backend=llm_backend, llm_model=llm_model,
                    )
                    row_vals.append(score_str)

                row_labels.append(row_label)
                results_rows.append(row_vals)

    print(results_rows)
    print(row_labels)

    return results_rows, hpo.g_model, landscape_details


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run HPO experiments")
    parser.add_argument("--task", type=str, default="experiment3",
                        choices=["experiment1", "experiment3"])
    parser.add_argument("--data_id", type=int, default=0)
    parser.add_argument("--parallel-trials", type=int, default=None)
    parser.add_argument("--parallel-devices", type=str, default=None,
                        help="Comma separated device list (e.g., 'cuda:0,cuda:1').")
    parser.add_argument("--version", type=str, default="baseline",
                        choices=["baseline", "ours", "autotransfer"])
    parser.add_argument("--landscape", action="store_true")
    parser.add_argument("--landscape-ratio", type=int, default=3,
                        help="Top-k trials for landscape selection.")
    parser.add_argument("--metapredictor", action="store_true")
    parser.add_argument("--skip-autotransfer", action="store_true",
                        help="Skip loading autotransfer embeddings.")
    parser.add_argument("--llm-select", action="store_true",
                        help="Use LLM post-selection instead of voting.")
    parser.add_argument("--llm-backend", type=str, default="anthropic",
                        choices=["anthropic", "bedrock"],
                        help="LLM backend for post-selection.")
    parser.add_argument("--llm-model", type=str, default="claude-sonnet-4-5-20250929",
                        help="LLM model ID for post-selection.")
    parser.add_argument("--needed", type=int, default=-1,
                        help="Number of tasks (experiment 2). -1 = all.")
    args = parser.parse_args()

    seed_everything(42)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    cli_parallel_devices = args.parallel_devices.strip() if args.parallel_devices else None

    if args.task == "experiment1":
        hpo = TaskEmbeddingHPO(
            parallel_trials=args.parallel_trials,
            parallel_devices=cli_parallel_devices,
            meta_classifier=args.metapredictor,
        )
        feature_combinations = [
            ['AT_All'],
            ['All_Model_Based'],
            ['Real_Entity'],
            ['Guess'],
        ]
        hpo.graphgym_similarity_all_combinations(feature_combinations, anchor_mode_type='common')

    elif args.task == "experiment3":
        skip_at = args.skip_autotransfer
        if args.version == "autotransfer":
            skip_at = False
        run_experiment_3(
            output_csv_path="exp3_results.csv",
            seeds_list=[42, 43, 44],
            budgets_normal=(30,),
            budgets_transfer=(3, 10),
            prior_strength=1.0,
            version=args.version,
            skip_unknown_params=False,
            debug_trials=True,
            use_meta_predictor=args.metapredictor,
            meta_for_landscape=args.landscape,
            downstream_id=args.data_id,
            parallel_trials=args.parallel_trials,
            parallel_devices=cli_parallel_devices,
            landscape_topk=max(1, args.landscape_ratio),
            skip_autotransfer=skip_at,
            only_show_prediction=True,
            use_llm_select=args.llm_select,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
        )


if __name__ == "__main__":
    main()
