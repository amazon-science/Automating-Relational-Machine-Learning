# swap/ — Architecture Search and HPO Engine

This package implements the core architecture search, hyperparameter optimization (HPO),
and task signal analysis pipeline for **Relatron**.

## File Overview

| File | Description |
|------|-------------|
| `hpo.py` | **Core HPO engine.** `TaskEmbeddingHPO` class that orchestrates task-embedding-based meta-selection, prior-guided sampling, and landscape-based post-hoc selection. Entry point for `scripts/run_hpo.py`. |
| `hpo_exec.py` | HPO execution utilities. Bridges between the HPO sampler and the training loop, handling DFS/RDL dispatch and result collection. |
| `execution.py` | Training execution wrapper. Runs a single (model config, dataset, task) trial and returns metrics + model weights. Used by both the online search agents and the HPO engine. |
| `generate.py` | Search space generation. Defines the RDL (GNN) and DFS (FT-Transformer) configuration spaces and samples random configurations. |
| `search_and_plot.py` | Loss landscape analysis. Provides `plot_loss_landscape_and_calculate_metrics()` for computing Lipschitz/sharpness/barrier metrics, and `load_torch_frame_loader()` for DFS data loading. |
| `gym.py` | Anchor model analysis and task similarity. Builds anchor tables per model family, computes Kendall-τ task similarity from anchor rankings, and provides embedding-based similarity utilities. |
| `regression.py` | Task embedding extraction. Loads and transforms homophily, heuristic, AutoTransfer, and loss-barrier features into DataFrames for downstream meta-regression. |
| `same_class_analysis.py` | Homophily computation. `SameClassAnalyzer` samples sequences from relational databases and computes same-class ratios (classification) or label correlations (regression) as task-level features. |
| `heuristics.py` | Task heuristic signal computation. Aggregates homophily features from `same_class_analysis` with other task-level signals. |
| `heuristics_at.py` | AutoTransfer task embedding generation. Standalone script that runs anchor model probing to produce per-task AutoTransfer embeddings. |

## Dependency Graph

```
scripts/run_hpo.py
  └─ swap/hpo.py
       ├─ swap/gym.py
       │    └─ swap/regression.py
       └─ swap/hpo_exec.py
            └─ swap/execution.py

swap_agent_online.py / swap_agent_tabnn.py
  ├─ swap/execution.py
  └─ swap/generate.py

utils/hpo.py
  ├─ swap/search_and_plot.py
  └─ utils/llm_selector.py      (LLM post-selection)

swap/heuristics.py
  └─ swap/same_class_analysis.py
```

## Post-Selection Modes

After HPO completes, the top-k trials can be re-ranked using loss landscape analysis.
Two post-selection modes are available via `scripts/run_hpo.py`:

1. **Voting** (`--landscape`): Weighted scoring over validation metric (0.4) + Lipschitz (0.2) + sharpness (0.2) + barrier (0.2). Deterministic, no external dependencies.

2. **LLM** (`--landscape --llm-select`): Sends candidate validation metrics and landscape properties to a Claude LLM, which reasons about the trade-offs and returns the best index as JSON. Falls back to voting on API failure. Requires `ANTHROPIC_API_KEY` in `.env` (or AWS credentials for `--llm-backend bedrock`).

Example:
```bash
python scripts/run_hpo.py --task experiment3 --version baseline \
  --landscape --llm-select --llm-backend anthropic --data_id 0
```
