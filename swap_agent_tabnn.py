"""Parallel GPU agent for DFS/TabNN (FT-Transformer) hyperparameter search.

Orchestrates:
1. DFS feature generation for each depth layer
2. Baseline runs (TabPFN + LightGBM) per layer
3. Parallel FT-Transformer HPO across (dfs_layer, text_to_pca) combinations
"""

import argparse
import itertools
import json
import os
import random
import string
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import GPUtil
from loguru import logger
from tqdm import tqdm

from data.mtaskdataset import prepare_transductive_rdb_datasets_save_memory
from swap.execution import execution as swap_execution
from swap.generate import execution as generate_execution
from utils.database import SwapDB, create_swap_db
from utils.information import ALL_CAN_DO_THREE
from utils.misc import load_yaml
from utils.space import DFS_SEARCH_SPACE, search

# Config paths
TABNN_CONFIG_PATH = "configs/default/tabnn.yaml"
LIGHTGBM_MODEL_CONFIG_PATH = "configs/model/lightgbm.yaml"
TABPFN_MODEL_CONFIG_PATH = "configs/model/tabpfn-2.yaml"

# Return codes from swap.execution
RC_SUCCESS = 0
RC_SAME_ARCH = 3
RC_DUPLICATE_PARAMS = 4
RC_MEMORY_ERROR = 5
RC_CUDA_OOM = 6
RC_PREPROCESS_TIMEOUT = 7


def generate_random_filename() -> str:
    """Generate a random string for unique file naming."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=10))


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
    fixed_space: List[Tuple[int, bool]],
    data_ok: Dict[str, Dict[int, bool]],
    max_search_time: int,
) -> Tuple[int, Dict[str, int]]:
    """Count how many more FT-Transformer trials each (dfs_layer, text_to_pca) needs."""
    remaining = {}
    for dfs_layer, text_to_pca in fixed_space:
        if not data_ok[task_name].get(dfs_layer, False):
            continue
        finished = result_db.get_num_of_finished_rows_with_tree(
            database_name=db_name,
            task_name=task_name,
            dfs_layer=dfs_layer,
            model_name="ft_transformer",
            text_to_pca=text_to_pca,
        )
        if finished < max_search_time:
            remaining[json.dumps((dfs_layer, text_to_pca))] = (
                max_search_time - finished
            )
    return len(remaining), remaining


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


def make_data_config(db_name: str, task_name: str) -> dict:
    """Build a minimal data config dict for execution calls."""
    return {"dbs": [{"db_name": db_name, "task_name": [task_name], "path": ""}]}


def run_dfs_generation(
    db_name: str,
    task_name: str,
    task_auto: bool,
    allow_dfs_layers: List[int],
    generation_config: dict,
    result_db: SwapDB,
    device: str,
) -> Dict[int, bool]:
    """Generate DFS features for each layer. Returns {layer: data_available}."""
    data_ok = {}
    for dfs_layer in allow_dfs_layers:
        model_config = load_yaml(LIGHTGBM_MODEL_CONFIG_PATH)
        model_config["dfs_layer"] = dfs_layer
        generate_execution(
            search_params=model_config,
            basic_config=generation_config,
            data_config=make_data_config(db_name, task_name),
            save_model=False,
            recovery_file="",
            recovery_directory="",
            result_db=result_db,
            device=device,
            easy=False,
            text_to_pca=True,
            text_to_pca_dim=3,
        )
        lte_postfix = "_lte" if task_auto else ""
        data_path = (
            f"datasets/dfs/{db_name}/"
            f"post_dfs_{dfs_layer}_{task_name}_text_pca_3{lte_postfix}_4dbinfer"
        )
        data_ok[dfs_layer] = os.path.exists(data_path)
    return data_ok


def run_baselines(
    db_name: str,
    task_name: str,
    allow_dfs_layers: List[int],
    generation_config: dict,
    result_db: SwapDB,
    device: str,
):
    """Run TabPFN and LightGBM baselines for each DFS layer."""
    for dfs_layer in allow_dfs_layers:
        # TabPFN baseline
        if (
            result_db.get_num_of_finished_rows_with_tree(
                database_name=db_name,
                task_name=task_name,
                dfs_layer=dfs_layer,
                model_name="tabpfn",
            )
            > 0
        ):
            logger.debug(f"Skipping tabpfn (layer={dfs_layer}): already done")
        else:
            tabpfn_config = load_yaml(TABPFN_MODEL_CONFIG_PATH)
            tabpfn_config["model_type"] = "fs"
            tabpfn_config["dfs_layer"] = dfs_layer
            tabpfn_config["text_to_pca"] = True
            swap_execution(
                search_params=tabpfn_config,
                basic_config=generation_config,
                data_config=make_data_config(db_name, task_name),
                save_model=False,
                recovery_file="",
                recovery_directory="",
                result_db=result_db,
                device=device,
                text_to_pca=True,
                noexit=True,
            )

        # LightGBM baseline
        if (
            result_db.get_num_of_finished_rows_with_tree(
                database_name=db_name,
                task_name=task_name,
                dfs_layer=dfs_layer,
                model_name="lightgbm",
            )
            > 0
        ):
            logger.debug(f"Skipping lgbm (layer={dfs_layer}): already done")
        else:
            lgbm_config = load_yaml(LIGHTGBM_MODEL_CONFIG_PATH)
            lgbm_config["model_type"] = "fs"
            lgbm_config["dfs_layer"] = dfs_layer
            lgbm_config["text_to_pca"] = True
            swap_execution(
                search_params=lgbm_config,
                basic_config=generation_config,
                data_config=make_data_config(db_name, task_name),
                save_model=False,
                recovery_file="",
                recovery_directory="",
                result_db=result_db,
                device=device,
                text_to_pca=True,
                noexit=True,
            )


def run_search(
    data_config_path: str,
    sleep_time: int = 30,
    num_parallel_task: int = 3,
    gpu_threshold: int = 2000,
    max_continuous_failure: int = 10,
    device: str = "cuda",
    debug_mode: bool = False,
    max_search_time_for_prediction: int = 20,
):
    """Run parallel DFS/FT-Transformer HPO search across datasets and tasks."""
    tabnn_config = load_yaml(TABNN_CONFIG_PATH)
    data_config = load_yaml(data_config_path)

    iter_dbs = prepare_transductive_rdb_datasets_save_memory(
        data_config,
        tabnn_config,
        stype_proposal="heuristic",
        graph_construct="r2n",
        sample_with_edge_ft=False,
    )

    result_db = create_swap_db(
        backend=tabnn_config.get("db_backend", "sqlite"),
        path_to_db=tabnn_config.get("db_location", "./swap_db"),
        collection_name=tabnn_config["result_db_name"],
        db_name=tabnn_config.get("db_name", "swap"),
    )

    if debug_mode:
        num_parallel_task = 1
        result_db.test = True

    for db in tqdm(iter_dbs, desc="Running datasets"):
        logger.info(f"Running dataset: {db.database.name}")

        for task in db.tasks:
            allow_dfs_layers = (
                [1, 2, 3] if task.name in ALL_CAN_DO_THREE else [1, 2]
            )
            target = len(allow_dfs_layers) * max_search_time_for_prediction

            # Skip if already completed
            finished = result_db.get_num_of_finished_rows_with_tree(
                database_name=db.database.name,
                task_name=task.name,
                dfs_layer="",
                model_name="ft_transformer",
                text_to_pca="",
            )
            if finished >= target:
                logger.info(
                    f"Skipping {db.database.name}/{task.name}: already complete"
                )
                continue
            logger.info(
                f"Running {db.database.name}/{task.name}: {finished}/{target}"
            )

            # Phase 1: Generate DFS features
            data_ok_for_task = run_dfs_generation(
                db.database.name,
                task.name,
                task.auto,
                allow_dfs_layers,
                tabnn_config,
                result_db,
                device,
            )

            # Phase 2: Run baselines (TabPFN + LightGBM)
            run_baselines(
                db.database.name,
                task.name,
                allow_dfs_layers,
                tabnn_config,
                result_db,
                device,
            )

            # Phase 3: Parallel FT-Transformer HPO
            fixed_params = DFS_SEARCH_SPACE["full_entities"]
            fixed_params["dfs_layer"] = allow_dfs_layers
            fixed_space = list(
                itertools.product(
                    fixed_params["dfs_layer"], fixed_params["text_to_pca"]
                )
            )
            data_ok = defaultdict(dict, {task.name: data_ok_for_task})

            logger.info(
                f"Starting FT-Transformer HPO for {task.name}: "
                f"{len(fixed_space)} fixed configurations"
            )

            running_processes = {}
            available_gpus_pool = get_available_gpus(gpu_threshold, [])
            exit_code_record = {}
            recovery_file_record = {}
            continuous_failure = 0
            debug_break = False

            task_pbar = tqdm(
                total=len(fixed_space) * max_search_time_for_prediction,
                desc=f"Progress for {db.database.name}/{task.name}",
                file=sys.stdout,
                leave=True,
                dynamic_ncols=True,
            )

            task_type_str = (
                task.task_type.value
                if not isinstance(task.task_type, str)
                else task.task_type
            )

            # Main HPO loop
            while True:
                total_remaining, detailed_remaining = get_remaining_trials(
                    result_db,
                    task.name,
                    db.database.name,
                    fixed_space,
                    data_ok,
                    max_search_time_for_prediction,
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

                if debug_break:
                    break

                # 1. Reap finished processes
                processes_to_remove = []
                for pid, info in list(running_processes.items()):
                    proc = info["process"]
                    if proc.poll() is None:
                        continue

                    processes_to_remove.append(pid)
                    gpu_id = info["gpu_id"]
                    available_gpus_pool.append(gpu_id)
                    available_gpus_pool = sorted(set(available_gpus_pool))

                    rc = proc.returncode
                    if rc == RC_SUCCESS:
                        logger.info(f"Success: PID {pid} on GPU {gpu_id}")
                        recovery_file = info.get("recovery_file", "")
                        if recovery_file and os.path.exists(recovery_file):
                            os.remove(recovery_file)
                        task_pbar.update(1)
                        continuous_failure = 0
                    elif rc in (RC_SAME_ARCH, RC_DUPLICATE_PARAMS, RC_PREPROCESS_TIMEOUT):
                        reason = {
                            RC_SAME_ARCH: "same arch found",
                            RC_DUPLICATE_PARAMS: "duplicate params",
                            RC_PREPROCESS_TIMEOUT: "preprocessing timeout",
                        }[rc]
                        logger.warning(
                            f"PID {pid}: {reason} (code={rc}). Skipping."
                        )
                        continuous_failure += 1
                    else:
                        proc.communicate()
                        logger.error(
                            f"Failure: PID {pid} on GPU {gpu_id} (code={rc})"
                        )
                        exit_code_record[json.dumps(info["hpo_params"])] = rc
                        continuous_failure += 1
                        if continuous_failure >= max_continuous_failure:
                            logger.error(
                                f"Exceeded {max_continuous_failure} failures. "
                                f"Stopping {task.name}."
                            )
                            sys.exit(1)

                for pid in processes_to_remove:
                    del running_processes[pid]

                # 2. Launch new processes
                while (
                    len(running_processes) < num_parallel_task
                    and available_gpus_pool
                ):
                    unfinished = list(detailed_remaining.keys())
                    if not unfinished:
                        break

                    dfs_key = random.choice(unfinished)
                    dfs_layer, text_to_pca = json.loads(dfs_key)

                    if not data_ok[task.name].get(dfs_layer, False):
                        logger.error(
                            f"Data unavailable for {task.name} "
                            f"layer={dfs_layer}"
                        )
                        continue

                    occupied = [
                        info["gpu_id"] for info in running_processes.values()
                    ]
                    system_gpus = get_available_gpus(gpu_threshold, occupied)
                    launchable = [
                        g for g in system_gpus if g in available_gpus_pool
                    ]
                    if not launchable:
                        break

                    gpu_id = launchable[0]
                    available_gpus_pool.remove(gpu_id)

                    hpo_params = search("fs", task_type_str)
                    hpo_params.update(
                        {"dfs_layer": dfs_layer, "text_to_pca": text_to_pca}
                    )

                    # Skip known-bad configs
                    params_key = json.dumps(hpo_params)
                    if params_key in exit_code_record:
                        prev_rc = exit_code_record[params_key]
                        if prev_rc == RC_MEMORY_ERROR:
                            logger.info("Recovering from memory error")
                            recovery_file_name = recovery_file_record[params_key]
                        elif prev_rc in (RC_CUDA_OOM, RC_PREPROCESS_TIMEOUT):
                            logger.info(f"Skipping (prev code={prev_rc})")
                            continue
                        else:
                            logger.info(f"Skipping unknown error (code={prev_rc})")
                            continue
                    else:
                        recovery_file_name = f"{generate_random_filename()}.pt"
                        recovery_file_record[params_key] = recovery_file_name

                    command_args = [
                        "python3", "-u", "-m", "swap.execution",
                        "--db-name", db.database.name,
                        "--task-name", task.name,
                        "--rdl-params", json.dumps(hpo_params),
                        "--params_type", "json",
                        "--device", device,
                        "--basic-config", TABNN_CONFIG_PATH,
                        "--deterministic",
                        "--text-to-pca",
                        "--save-model",
                    ]

                    logger.info(
                        f"Launching on GPU {gpu_id}: "
                        f"dfs_layer={dfs_layer}, text_to_pca={text_to_pca}"
                    )

                    if debug_mode:
                        logger.info("Debug mode: running in-process")
                        result_db.test = True
                        swap_execution(
                            search_params=hpo_params,
                            basic_config=tabnn_config,
                            data_config=make_data_config(
                                db.database.name, task.name
                            ),
                            save_model=False,
                            recovery_file="",
                            recovery_directory="",
                            result_db=result_db,
                            device=device,
                            text_to_pca=text_to_pca,
                        )
                        debug_break = True
                    else:
                        try:
                            env = os.environ.copy()
                            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                            process = subprocess.Popen(command_args, env=env)
                            running_processes[process.pid] = {
                                "process": process,
                                "gpu_id": gpu_id,
                                "recovery_file": recovery_file_name,
                                "hpo_params": hpo_params,
                                "config_key": dfs_key,
                            }
                        except Exception as e:
                            logger.error(f"Error launching process: {e}")
                            if gpu_id not in available_gpus_pool:
                                available_gpus_pool.append(gpu_id)
                                available_gpus_pool = sorted(
                                    set(available_gpus_pool)
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
        description="Parallel DFS/FT-Transformer HPO search on available GPUs."
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
        "--gpu_threshold", type=int, default=2000,
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
        "--max_search_time_for_prediction", type=int, default=20,
        help="Max trials per config for prediction tasks.",
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
    )
