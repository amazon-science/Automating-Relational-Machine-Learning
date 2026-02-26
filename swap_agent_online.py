"""Parallel GPU agent for GNN/RDL hyperparameter search.

Orchestrates exhaustive search over fixed GNN architecture space
(pre_sf, mpnn_type, post_sf, loader_type) with sampled hyperparameters,
running multiple trials in parallel across available GPUs.
"""

import argparse
import gc
import itertools
import json
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import GPUtil
import torch
from loguru import logger
from tqdm import tqdm

from data.mtaskdataset import prepare_transductive_rdb_datasets_save_memory
from swap.execution import execution
from utils.database import SwapDB, create_swap_db
from utils.misc import load_yaml
from utils.space import RDL_SEARCH_SPACE, search

# Config paths
NODE_CONFIG_PATH = "configs/default/basic.yaml"
LINK_CONFIG_PATH = "configs/default/link.yaml"
TEST_MODEL_CONFIG_PATH = "configs/model/nbfnet.yaml"

TASK_TYPE_MAP = {
    "TaskType.LINK_PREDICTION": "recommendation",
    "TaskType.BINARY_CLASSIFICATION": "binary_classification",
    "TaskType.MULTICLASS_CLASSIFICATION": "multiclass_classification",
    "TaskType.REGRESSION": "regression",
    "link_prediction": "recommendation",
    "binary_classification": "binary_classification",
    "multiclass_classification": "multiclass_classification",
    "regression": "regression",
}


def get_available_gpus(
    gpu_threshold: int, occupied_gpus: List[int] = None
) -> List[int]:
    """Get GPU indices with free memory above threshold, excluding occupied ones."""
    occupied_gpus = occupied_gpus or []
    return [
        gpu.id
        for gpu in GPUtil.getGPUs()
        if gpu.id not in occupied_gpus and gpu.memoryFree >= gpu_threshold
    ]


def get_remaining_trials(
    result_db: SwapDB,
    task_name: str,
    db_name: str,
    fixed_space: List[Tuple[str, str, str, str]],
    max_search_time: int,
) -> Tuple[int, Dict[str, int]]:
    """Count how many more trials each fixed config needs."""
    remaining = {}
    for config in fixed_space:
        pre_sf, mpnn_type, post_sf, loader_type = config
        finished = result_db.get_num_of_finished_rows(
            database_name=db_name,
            task_name=task_name,
            pre_sf=pre_sf,
            mpnn_type=mpnn_type,
            post_sf=post_sf,
        )
        if finished < max_search_time:
            remaining[json.dumps(config)] = max_search_time - finished
    return len(remaining), remaining


def sync_wandb():
    """Sync offline wandb runs and clean up."""
    subprocess.run(["wandb", "sync", "--sync-all", "--mark-synced"])
    subprocess.run("rm -rf wandb/*", shell=True)
    subprocess.run(["wandb", "offline"])


def kill_running_processes(running_processes: dict):
    """Kill any still-running subprocesses."""
    for pid, info in running_processes.items():
        proc = info["process"]
        if proc.poll() is None:
            logger.info(f"Killing process {pid}")
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {pid} did not terminate cleanly")
                proc.kill()
        else:
            logger.info(
                f"Process {pid} already finished (code={proc.returncode})"
            )


def warmup_data_cache(
    db_name: str,
    task_name: str,
    basic_config: dict,
    result_db: SwapDB,
    device: str,
):
    """Run a single dummy execution to pre-cache dataset loading."""
    test_db = create_swap_db(
        backend=basic_config.get("db_backend", "sqlite"),
        path_to_db=basic_config.get("db_location", "./swap_db"),
        collection_name=basic_config.get("result_db_name", "test"),
        db_name=basic_config.get("db_name", "swap"),
    )
    test_db.test = True
    test_model_config = load_yaml(TEST_MODEL_CONFIG_PATH)
    execution(
        search_params=test_model_config,
        basic_config=basic_config,
        data_config={
            "dbs": [{"db_name": db_name, "task_name": [task_name], "path": ""}]
        },
        save_model=False,
        recovery_file="",
        recovery_directory="",
        result_db=test_db,
        device=device,
        noexit=True,
    )
    torch.cuda.empty_cache()
    gc.collect()


def run_search(
    data_config_path: str,
    sleep_time: int = 30,
    num_parallel_task: int = 3,
    gpu_threshold: int = 20000,
    max_continuous_failure: int = 10,
    device: str = "cuda",
    debug_mode: bool = False,
    max_search_time_for_prediction: int = 15,
    max_search_time_for_ranking: int = 10,
):
    """Run parallel GNN/RDL hyperparameter search across datasets and tasks."""
    node_basic_config = load_yaml(NODE_CONFIG_PATH)
    link_basic_config = load_yaml(LINK_CONFIG_PATH)
    data_config = load_yaml(data_config_path)

    iter_dbs = prepare_transductive_rdb_datasets_save_memory(
        data_config,
        node_basic_config,
        stype_proposal="heuristic",
        graph_construct="r2n",
        sample_with_edge_ft=False,
    )

    result_db = create_swap_db(
        backend=node_basic_config.get("db_backend", "sqlite"),
        path_to_db=node_basic_config.get("db_location", "./swap_db"),
        collection_name=node_basic_config["result_db_name"],
        db_name=node_basic_config.get("db_name", "swap"),
    )

    for db in tqdm(iter_dbs, desc="Running datasets"):
        logger.info(f"Processing dataset: {db.database.name}")

        for task in db.tasks:
            sync_wandb()

            level = (
                "link"
                if str(task.task_type)
                in ("link_prediction", "TaskType.LINK_PREDICTION")
                else "node"
            )
            loader_type_options = (
                ["link", "node"] if level == "link" else ["node"]
            )
            max_search_time = (
                max_search_time_for_prediction
                if level == "node"
                else max_search_time_for_ranking
            )

            # Build the fixed architecture search space
            fixed_params = RDL_SEARCH_SPACE["full_entities"]
            fixed_space = [
                config
                for config in itertools.product(
                    fixed_params["pre_sf"],
                    fixed_params["mpnn_type"],
                    fixed_params["post_sf"][level],
                    loader_type_options,
                )
                if config[3] == "node"
                or (config[3] == "link" and config[2] == "none")
            ]
            target = len(fixed_space) * max_search_time

            # Skip if already completed
            num_finished = result_db.get_num_of_finished_rows(
                database_name=db.database.name,
                task_name=task.name,
                pre_sf="",
                mpnn_type="",
                post_sf="",
            )
            if num_finished >= target:
                logger.info(
                    f"Skipping {db.database.name}/{task.name}: already complete"
                )
                continue
            logger.info(
                f"Running {db.database.name}/{task.name}: {num_finished}/{target}"
            )

            # Warmup: pre-cache data with a dummy execution
            basic_config = (
                node_basic_config if level == "node" else link_basic_config
            )
            warmup_data_cache(
                db.database.name, task.name, node_basic_config, result_db, device
            )

            task_type_str = TASK_TYPE_MAP[str(task.task_type)]
            config_path = (
                NODE_CONFIG_PATH if level == "node" else LINK_CONFIG_PATH
            )

            logger.info(
                f"Starting {task.name}: {len(fixed_space)} fixed configurations"
            )

            running_processes = {}
            continuous_failure = 0
            task_pbar = tqdm(
                total=len(fixed_space) * max_search_time,
                desc=f"Progress for {db.database.name}/{task.name}",
                file=sys.stdout,
                leave=True,
                dynamic_ncols=True,
            )

            # Main search loop
            while True:
                total_remaining, detailed_remaining = get_remaining_trials(
                    result_db,
                    task.name,
                    db.database.name,
                    fixed_space,
                    max_search_time,
                )
                if total_remaining == 0:
                    logger.info(f"All trials complete for {task.name}")
                    break

                logger.info(
                    f"Remaining trials for {task.name}: {total_remaining}"
                )

                if continuous_failure >= max_continuous_failure:
                    logger.error(
                        f"Exceeded {max_continuous_failure} continuous failures. "
                        f"Stopping {task.name}."
                    )
                    break

                # 1. Reap finished processes
                for pid in [
                    p
                    for p, info in running_processes.items()
                    if info["process"].poll() is not None
                ]:
                    info = running_processes.pop(pid)
                    rc = info["process"].returncode
                    if rc == 0:
                        logger.info(
                            f"Success: PID {pid} on GPU {info['gpu_id']}"
                        )
                        task_pbar.update(1)
                        continuous_failure = 0
                    else:
                        info["process"].communicate()
                        logger.error(
                            f"Failure: PID {pid} on GPU {info['gpu_id']} "
                            f"(code={rc})"
                        )
                        continuous_failure += 1
                        if continuous_failure >= max_continuous_failure:
                            logger.error(
                                f"Exceeded {max_continuous_failure} failures. "
                                f"Stopping {task.name}."
                            )
                            sys.exit(1)

                # 2. Launch new processes
                while len(running_processes) < num_parallel_task:
                    unfinished = list(detailed_remaining.keys())
                    if not unfinished:
                        break

                    config_key = random.choice(unfinished)
                    pre_sf, mpnn_type, post_sf, loader_type = json.loads(
                        config_key
                    )

                    occupied = [
                        info["gpu_id"] for info in running_processes.values()
                    ]
                    available_gpus = get_available_gpus(gpu_threshold, occupied)
                    if not available_gpus:
                        break

                    gpu_id = available_gpus[0]

                    # Determine minimum GNN layers for specific tasks
                    minimum_layers = (
                        4
                        if db.database.name == "rel-trial"
                        and str(task.task_type) == "link_prediction"
                        else 0
                    )

                    hpo_params = search(
                        "rdl", task_type_str, minimum_layers=minimum_layers
                    )
                    hpo_params["gnn_config"].update(
                        {
                            "pre_sf": pre_sf,
                            "mpnn_type": mpnn_type,
                            "post_sf": post_sf,
                        }
                    )
                    hpo_params["sampler_config"]["loader_type"] = loader_type

                    command_args = [
                        "python3", "-u", "-m", "swap.execution",
                        "--db-name", db.database.name,
                        "--task-name", task.name,
                        "--rdl-params", json.dumps(hpo_params),
                        "--params_type", "json",
                        "--device", device,
                        "--basic-config", config_path,
                        "--deterministic",
                        "--save-model",
                    ]

                    logger.info(
                        f"Launching on GPU {gpu_id}: "
                        f"pre_sf={pre_sf}, mpnn={mpnn_type}, post_sf={post_sf}"
                    )

                    if debug_mode:
                        logger.info("Debug mode: running in-process")
                        execution(
                            search_params=hpo_params,
                            basic_config=basic_config,
                            data_config={
                                "dbs": [
                                    {
                                        "db_name": db.database.name,
                                        "task_name": [task.name],
                                        "path": "",
                                    }
                                ]
                            },
                            save_model=False,
                            recovery_file="",
                            recovery_directory="",
                            result_db=result_db,
                            device=device,
                        )
                        break
                    else:
                        try:
                            env = os.environ.copy()
                            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                            process = subprocess.Popen(command_args, env=env)
                            running_processes[process.pid] = {
                                "process": process,
                                "gpu_id": gpu_id,
                                "hpo_params": hpo_params,
                                "config_key": config_key,
                            }
                        except Exception as e:
                            logger.opt(exception=True).error(
                                f"Error launching process: {e}"
                            )
                            time.sleep(sleep_time)

                # 3. Wait before next cycle
                time.sleep(sleep_time)

            kill_running_processes(running_processes)
            task_pbar.close()
            logger.info(
                f"Task '{task.name}' for '{db.database.name}' finished."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel GNN/RDL hyperparameter search on available GPUs."
    )
    parser.add_argument(
        "--data_config_path", type=str, required=True,
        help="Path to the data configuration YAML file.",
    )
    parser.add_argument(
        "--sleep_time", type=float, default=30.0,
        help="Seconds to wait between process checks.",
    )
    parser.add_argument(
        "--num_parallel_task", type=int, default=3,
        help="Maximum number of parallel tasks.",
    )
    parser.add_argument(
        "--gpu_threshold", type=int, default=20000,
        help="Minimum free GPU memory (MB) to use a GPU.",
    )
    parser.add_argument(
        "--max_continuous_failure", type=int, default=10,
        help="Max continuous failures before stopping a task.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for training.",
    )
    parser.add_argument(
        "--debug_mode", action="store_true",
        help="Run in-process for debugging.",
    )
    parser.add_argument(
        "--max_search_time_for_prediction", type=int, default=15,
        help="Max trials per config for prediction tasks.",
    )
    parser.add_argument(
        "--max_search_time_for_ranking", type=int, default=10,
        help="Max trials per config for ranking tasks.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("parallel_execution.log", rotation="10 MB", level="DEBUG")

    run_search(
        data_config_path=args.data_config_path,
        sleep_time=args.sleep_time,
        num_parallel_task=args.num_parallel_task,
        gpu_threshold=args.gpu_threshold,
        max_continuous_failure=args.max_continuous_failure,
        device=args.device,
        debug_mode=args.debug_mode,
        max_search_time_for_prediction=args.max_search_time_for_prediction,
        max_search_time_for_ranking=args.max_search_time_for_ranking,
    )
