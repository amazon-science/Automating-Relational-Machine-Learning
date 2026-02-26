"""
Download all RelBench datasets used in task_feature_summary.csv and generate DFS features.

DFS hop count is determined by:
  - 3 hops if task is in ALL_CAN_DO_THREE (REGRESSION_CAN_DO_THREE + CLASSIFICATION_CAN_DO_THREE)
  - 2 hops otherwise

Usage:
    python scripts/download_and_generate_dfs.py
    python scripts/download_and_generate_dfs.py --cache-dir /path/to/cache
    python scripts/download_and_generate_dfs.py --dataset rel-f1          # single dataset
    python scripts/download_and_generate_dfs.py --dataset rel-f1 --task driver-dnf  # single task
    python scripts/download_and_generate_dfs.py --download-only           # skip DFS generation
    python scripts/download_and_generate_dfs.py --dfs-only                # skip download (assume cached)
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CACHE_DIR = "cache_data"
DEFAULT_DFS_SAVE_DIR = "cache_data/old_dfs"

# Set XDG_CACHE_HOME so relbench finds cached datasets in cache_data/
# Must be absolute path and set BEFORE any relbench imports
os.environ["XDG_CACHE_HOME"] = os.path.join(PROJECT_ROOT, DEFAULT_CACHE_DIR)

from loguru import logger
from utils.information import SUMMARY_TASKS, ALL_CAN_DO_THREE
from data.relbench import load_relbench_data, load_relbench_task
from data.mtaskdataset import MultiTabularDataset
from models.dfs import generate_dfs_data_pre_post

BASIC_CONFIG = {
    "cache_dir": DEFAULT_CACHE_DIR,
    "seed": 42,
    "text_encoder": "glove",
    "feature_transform": "none",
    "train_sample_size": -1,
}


def get_dfs_hops(task_name: str) -> int:
    """Return 3 if the task can do 3-hop DFS, else 2."""
    return 3 if task_name in ALL_CAN_DO_THREE else 2


def download_dataset(dataset_name: str):
    """Download and cache a RelBench dataset."""
    logger.info(f"Downloading dataset: {dataset_name}")
    db = load_relbench_data(dataset_name)
    logger.info(f"Dataset {dataset_name} loaded successfully ({len(db.table_dict)} tables)")
    return db


def prepare_and_generate_dfs(dataset_name: str, task_name: str, cache_dir: str,
                              dfs_save_dir: str = DEFAULT_DFS_SAVE_DIR,
                              text_to_pca: bool = True, text_to_pca_dim: int = 3):
    """Prepare dataset and generate DFS features for a single task."""
    max_depth = get_dfs_hops(task_name)
    logger.info(f"Processing {dataset_name}/{task_name} with {max_depth}-hop DFS")

    db = load_relbench_data(dataset_name)
    task = load_relbench_task(dataset_name, task_name)
    os.makedirs(cache_dir, exist_ok=True)

    dataset = MultiTabularDataset(
        db,
        [task],
        cache_dir,
        text_encoder=BASIC_CONFIG["text_encoder"],
    )
    dataset.train_sample_size = BASIC_CONFIG["train_sample_size"]
    dataset.train_sample_seed = BASIC_CONFIG["seed"]

    scaler_path_name = f"scaler_{dataset_name}_{task_name}_{max_depth}"
    generate_dfs_data_pre_post(
        dfs_save_dir,
        dataset,
        max_depth=max_depth,
        load_scalers_from=scaler_path_name,
        mode="easy",
        lte=False,
        text_to_pca=text_to_pca,
        text_to_pca_dim=text_to_pca_dim,
    )
    logger.info(f"DFS generation complete for {dataset_name}/{task_name}")


def main():
    parser = argparse.ArgumentParser(description="Download RelBench datasets and generate DFS features")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
                        help="Cache directory for downloaded data")
    parser.add_argument("--dfs-save-dir", type=str, default=DEFAULT_DFS_SAVE_DIR,
                        help="Directory to save generated DFS features")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Process only this dataset (e.g. rel-f1)")
    parser.add_argument("--task", type=str, default=None,
                        help="Process only this task (requires --dataset)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download datasets, skip DFS generation")
    parser.add_argument("--dfs-only", action="store_true",
                        help="Only generate DFS, skip download step")
    parser.add_argument("--text-to-pca", action="store_true", default=True,
                        help="Apply PCA to text embeddings (default: True)")
    parser.add_argument("--text-to-pca-dim", type=int, default=3,
                        help="PCA dimension for text embeddings")
    args = parser.parse_args()

    if args.task and not args.dataset:
        parser.error("--task requires --dataset")

    BASIC_CONFIG["cache_dir"] = args.cache_dir
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.dfs_save_dir, exist_ok=True)

    # Determine which dataset-task pairs to process
    if args.dataset:
        if args.dataset not in SUMMARY_TASKS:
            parser.error(f"Unknown dataset: {args.dataset}. Available: {list(SUMMARY_TASKS.keys())}")
        if args.task:
            if args.task not in SUMMARY_TASKS[args.dataset]:
                parser.error(f"Task {args.task} not in {args.dataset}. Available: {SUMMARY_TASKS[args.dataset]}")
            pairs = [(args.dataset, args.task)]
        else:
            pairs = [(args.dataset, t) for t in SUMMARY_TASKS[args.dataset]]
    else:
        pairs = [(ds, t) for ds, tasks in SUMMARY_TASKS.items() for t in tasks]

    logger.info(f"Will process {len(pairs)} dataset-task pairs")
    for ds, t in pairs:
        logger.info(f"  {ds}/{t} -> {get_dfs_hops(t)}-hop DFS")

    # Step 1: Download datasets
    if not args.dfs_only:
        datasets_to_download = sorted(set(ds for ds, _ in pairs))
        for ds_name in datasets_to_download:
            try:
                download_dataset(ds_name)
            except Exception as e:
                logger.error(f"Failed to download {ds_name}: {e}")
                continue

    # Step 2: Generate DFS features
    if not args.download_only:
        for ds_name, task_name in pairs:
            try:
                prepare_and_generate_dfs(
                    ds_name, task_name, args.cache_dir,
                    dfs_save_dir=args.dfs_save_dir,
                    text_to_pca=args.text_to_pca,
                    text_to_pca_dim=args.text_to_pca_dim,
                )
            except Exception as e:
                logger.error(f"Failed to generate DFS for {ds_name}/{task_name}: {e}")
                continue

    logger.info("All done!")


if __name__ == "__main__":
    main()
