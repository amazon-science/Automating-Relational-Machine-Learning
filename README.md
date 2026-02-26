# (ICLR 2026) Relatron: Automating Relational Machine Learning over Relational Databases

**Zhikai Chen, Han Xie, Jian Zhang, Jiliang Tang, Xiang Song, Huzefa Rangwala**

## Abstract

Predictive modeling over relational databases (RDBs) powers applications in various domains, yet remains challenging due to the need to capture both cross-table dependencies and complex feature interactions. Recent Relational Deep Learning (RDL) methods automate feature engineering via message passing, while classical approaches like Deep Feature Synthesis (DFS) rely on predefined non-parametric aggregators. Despite promising performance gains, the comparative advantages of RDL over DFS and the design principles for selecting effective architectures remain poorly understood.

We present a comprehensive study that unifies RDL and DFS in a shared design space and conducts large-scale architecture-centric searches across diverse RDB tasks. Our analysis yields three key findings: (1) RDL does not consistently outperform DFS, with performance being highly task-dependent; (2) no single architecture dominates across tasks, underscoring the need for task-aware model selection; and (3) validation accuracy is an unreliable guide for architecture choice. This search yields a curated model performance bank that links model architecture configurations to their performance; leveraging this bank, we analyze the drivers of the RDL-DFS performance gap and introduce two task signals -- RDB task homophily and an affinity embedding that captures size, path, feature, and temporal structure -- whose correlation with the gap enables principled routing. Guided by these signals, we propose **Relatron**, a task embedding-based meta-selector that first chooses between RDL and DFS and then prunes the within-family search to deliver strong performance. Lightweight loss-landscape metrics further guard against brittle checkpoints by preferring flatter optima. In experiments, Relatron resolves the "more tuning, worse performance" effect and, in joint hyperparameter-architecture optimization, achieves up to **18.5% improvement** over strong baselines with **10x lower computational cost** than Fisher information-based alternatives.

## Environment Setup

This project uses [pixi](https://pixi.sh) for environment and dependency management. All Python and system dependencies (including CUDA-enabled PyTorch, PyG, and RAPIDS) are specified in `pixi.toml`.

### Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Install dependencies

```bash
pixi install -e iclr
```

This creates an isolated environment with Python 3.12, PyTorch 2.6.0 (CUDA 12.4), PyTorch Geometric, RAPIDS, and all required packages.

### Activate the environment

```bash
pixi shell -e iclr
```

All commands below should be run inside the pixi shell (or prefixed with `pixi run -e iclr`).

## Repository Structure

```
Relatron/
├── configs/              # Hydra/YAML configuration files
│   └── default/          # Default task and model configs
├── data/                 # Data loading and dataset classes
│   ├── relbench.py       # RelBench data loader
│   ├── mtaskdataset.py   # Multi-task tabular dataset
│   └── gboltdataset.py   # Graph-based online learning dataset
├── dbinfer/              # Database inference and preprocessing
│   ├── preprocess/       # Data transforms (e.g., float compression)
│   └── solutions/        # Solution backends (TabPFN, etc.)
├── models/               # Model implementations
│   ├── nn/               # Neural network architectures (AutoGNN, encoder)
│   ├── dfs/              # Deep Feature Synthesis pipeline
│   ├── heads/            # Prediction heads
│   ├── automl/           # AutoTransfer and loss-landscape utilities
│   ├── embedder.py       # Feature embedder
│   └── regressor.py      # Regression model wrapper
├── swap/                 # Architecture search and HPO engine
│   ├── hpo.py            # Task-embedding HPO (Relatron core)
│   ├── execution.py      # Training execution wrapper
│   ├── generate.py       # Search space generation
│   ├── heuristics.py     # Task heuristic signal computation
│   ├── heuristics_at.py  # AutoTransfer heuristics
│   ├── regression.py     # Meta-regression models
│   └── search_and_plot.py# Search result analysis and plotting
├── utils/                # Shared utilities
│   ├── information.py    # Task lists and dataset metadata
│   ├── heuristics.py     # Homophily and task signal computation
│   ├── hpo.py            # HPO catalog and objective helpers
│   ├── graph.py          # Graph construction utilities
│   └── database.py       # Database helpers
├── scripts/              # Experiment and data scripts
│   ├── download_and_generate_dfs.py  # Download datasets + generate DFS features
│   ├── exp31.py          # Figure 3-1: rank comparison plot
│   ├── exp42.py          # Table 4 & 5: task signal evaluation
│   └── run_hpo.py        # HPO experiments (Tables 4 & 5)
├── plot/                 # Plotting and analysis scripts
├── results/              # Output CSVs and figures
├── relbench/             # Local fork of RelBench (editable install)
├── pytorch-frame/        # Local fork of PyTorch Frame (editable install)
├── rustler/              # Rust acceleration module (editable install)
└── pixi.toml             # Environment and dependency specification
```

## Downloading Datasets and Generating DFS Features

Use the provided script to download all RelBench datasets and generate DFS features:

```bash
# Download all datasets and generate DFS features
pixi run -e iclr python scripts/download_and_generate_dfs.py

# Download a single dataset
pixi run -e iclr python scripts/download_and_generate_dfs.py --dataset rel-f1

# Download a single dataset-task pair
pixi run -e iclr python scripts/download_and_generate_dfs.py --dataset rel-f1 --task driver-dnf

# Download only (skip DFS generation)
pixi run -e iclr python scripts/download_and_generate_dfs.py --download-only

# Generate DFS only (assume datasets are already cached)
pixi run -e iclr python scripts/download_and_generate_dfs.py --dfs-only
```

Downloaded data is cached under `cache_data/`, and DFS features are saved to `cache_data/old_dfs/`.

## Reproducing Experimental Results

### Figure 3-1: Rank Comparison (Classification vs. Regression)

This generates the rank comparison bar charts from the model performance bank results in `results/result31.csv`:

```bash
pixi run -e iclr python scripts/exp31.py
```

Output is saved to `plot/result31.png` and `plot/result31.pdf`.

### Table 4 & Table 5: Task Signal Evaluation and Meta-Selector

Run the task signal feature evaluation (LOO analysis of homophily, affinity embeddings, and model-based features):

```bash
pixi run -e iclr python scripts/exp42.py
```

Results are saved to `results/exp42_feature_evaluation.csv` and `results/exp42_task_features.csv`.

### Generating the Model Performance Bank

The performance bank is built by running large-scale architecture searches for both RDL (GNN-based) and DFS (tabular) model families across all tasks. The data config at `configs/pyg/hpo/all-nodes.yaml` defines the full set of dataset-task pairs used in the paper.

**Step 1: Download datasets and generate DFS features** (if not already done):

```bash
pixi run -e iclr python scripts/download_and_generate_dfs.py
```

**Step 2: Run RDL (GNN) architecture search:**

This searches over the GNN design space (`pre_sf`, `mpnn_type`, `post_sf`, `loader_type`) with sampled hyperparameters, running multiple trials per configuration in parallel across GPUs.

```bash
# Run on all tasks (uses all available GPUs)
pixi run -e iclr python swap_agent_online.py \
    --data_config_path configs/pyg/hpo/all-nodes.yaml

# Customize parallelism and GPU memory threshold
pixi run -e iclr python swap_agent_online.py \
    --data_config_path configs/pyg/hpo/all-nodes.yaml \
    --num_parallel_task 4 \
    --gpu_threshold 20000 \
    --max_search_time_for_prediction 15 \
    --max_search_time_for_ranking 10

# Debug mode (single process, no subprocess spawning)
pixi run -e iclr python swap_agent_online.py \
    --data_config_path configs/pyg/hpo/all-nodes.yaml \
    --debug_mode
```

**Step 3: Run DFS (FT-Transformer) architecture search:**

This first generates DFS features and runs TabPFN/LightGBM baselines per DFS layer, then performs parallel FT-Transformer HPO over `(dfs_layer, text_to_pca)` combinations.

```bash
# Run on all tasks
pixi run -e iclr python swap_agent_tabnn.py \
    --data_config_path configs/pyg/hpo/all-nodes.yaml

# Customize settings
pixi run -e iclr python swap_agent_tabnn.py \
    --data_config_path configs/pyg/hpo/all-nodes.yaml \
    --num_parallel_task 4 \
    --gpu_threshold 2000 \
    --max_search_time_for_prediction 20

# Debug mode
pixi run -e iclr python swap_agent_tabnn.py \
    --data_config_path configs/pyg/hpo/all-nodes.yaml \
    --debug_mode
```

Results are stored in a local SQLite database (`./swap_db/`). We use SQLite here for demo purposes; for large-scale runs you can switch to MongoDB by setting `db_backend: "mongodb"` in the config files.

**Running on a single dataset:**

Create a minimal data config YAML, e.g.:

```yaml
# configs/pyg/hpo/rel-f1.yaml
dbs:
  - db_name: "rel-f1"
    task_name:
      - "driver-dnf"
      - "driver-top3"
      - "driver-position"
```

Then pass it to either agent:

```bash
pixi run -e iclr python swap_agent_online.py --data_config_path configs/pyg/hpo/rel-f1.yaml
pixi run -e iclr python swap_agent_tabnn.py --data_config_path configs/pyg/hpo/rel-f1.yaml
```

### Tasks requiring performance bank

The following experiments require the model performance bank generated above.
If you prefer to skip the search and use our pre-computed bank, please contact chenzh85@msu.edu.

For the first column of Table 4, use:

```bash
pixi run -e iclr python scripts/run_hpo.py --task experiment1
```

For Table 6 (HPO comparison), run experiment 3 with the baseline, ours, and autotransfer versions:

```bash
# Baseline methods (Random, TPE, Hyperband) on a single task (--data_id selects task index)
pixi run -e iclr python scripts/run_hpo.py --task experiment3 --version baseline --data_id 0

# Ours (TPE + landscape post-hoc selection)
pixi run -e iclr python scripts/run_hpo.py --task experiment3 --version ours --data_id 0

# Autotransfer baseline
pixi run -e iclr python scripts/run_hpo.py --task experiment3 --version autotransfer --data_id 0
```

Instead of the weighted-voting post-selection, you can use an LLM to reason
about the trade-off between validation performance and loss landscape smoothness.
Add `--llm-select` alongside `--landscape`:

```bash
pixi run -e iclr python scripts/run_hpo.py --task experiment3 --version baseline \
  --landscape --llm-select --llm-backend anthropic --data_id 0
```

This requires `ANTHROPIC_API_KEY` in your `.env` file. AWS Bedrock is also supported
via `--llm-backend bedrock`. On API failure, the system falls back to voting.

## Security

See [CONTRIBUTING](CONTRIBUTING.md) for more information.

## License

This library is licensed under the CC BY-NC 4.0 License.

## Citation

```bibtex
@inproceedings{
chen2026relatron,
title={Relatron: Automating Relational Machine Learning over Relational Databases},
author={Zhikai Chen and Han Xie and Jian Zhang and Jiliang Tang and Xiang song and Huzefa Rangwala},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=59avbH4HnU}
}
```
