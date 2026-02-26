import os

import torch
import argparse
import warnings

warnings.filterwarnings("ignore")

from loguru import logger
from data.mtaskdataset import prepare_transductive_rdb_datasets_save_memory
from utils.misc import load_yaml
from models.automl.autotransfer import generate_model_task_embedding, SIX_ANCHOR_CONFIG, DFS_ANCHOR_CONFIG
from models.dfs import generate_dfs_data_pre_post


def clear_task_memory():
    """
    Clear memory after processing each task to prevent memory accumulation.
    """
    import gc
    import torch
    
    # Synchronize CUDA to release orphaned tensors without forcing a cache flush
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Clear Python garbage collection
    gc.collect()
    
    # Force garbage collection again
    gc.collect()
    
    # Log memory usage if available
    if torch.cuda.is_available() and hasattr(torch.cuda, 'memory_allocated'):
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logger.debug(f"GPU Memory after task cleanup - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")


def _taskemb_path(task_identifier: str, suffix: str) -> str:
    """
    Keep file naming consistent and avoid rebuilding strings all over the place.
    """
    return os.path.join('./taskemb/taskemb/atemb/', f"{task_identifier}-{suffix}.pt")


def task_embedding_execution(basic_config, data_config, args):
    """
        Called by swap agent
        Args:
            search_params: a dictionary of search parameters
            output_prefix: the prefix of the output file
            save_model: whether to save the model
            recovery_file: the file name of the recovery file
        Returns:
    """
    os.makedirs(args.root_dir, exist_ok=True)
    databases = prepare_transductive_rdb_datasets_save_memory(data_config, basic_config)
    for database in databases:
        for task_id, task in enumerate(database.tasks):
            task_identifier = f"{database.database.name}-{task.name}"
            atrdl_path = _taskemb_path(task_identifier, "model-atrdl")
            dfs_path = _taskemb_path(task_identifier, "model-atdfs")

            if os.path.exists(atrdl_path):
                logger.info(f"Task {database.database.name}-{task.name} already exists")
            else:
                logger.info(f"Processing task {database.database.name}-{task.name}")
                ta_th_at = generate_model_task_embedding(
                    database,
                    task_id,
                    basic_config,
                    method_type='training-head',
                    model_name='auto-transfer',
                    anchor_config=SIX_ANCHOR_CONFIG,
                )
                task_embedding_for_this_task = {
                    "ta_th_at": ta_th_at,
                }
                torch.save(task_embedding_for_this_task, atrdl_path)
                
                # Make sure intermediate tensors are reclaimed before the next heavy run
                clear_task_memory()
            
            if os.path.exists(dfs_path):
                logger.info(f"Task {database.database.name}-{task.name} already exists")
                continue
            else:
                ## generate dfs data if not exists
                one_layer_scaler_path_name = f"scaler_{database.database.name}_{task.name}_1"
                two_layer_scaler_path_name = f"scaler_{database.database.name}_{task.name}_2"
                ## control to one task and one layer
                two_layer_rdb_dataset = generate_dfs_data_pre_post("datasets/dfs", database, max_depth=2, 
                load_scalers_from=two_layer_scaler_path_name, lte = database.tasks[task_id].auto, text_to_pca=True, text_to_pca_dim=3, task_idx=task_id) 
                one_layer_rdb_dataset = generate_dfs_data_pre_post("datasets/dfs", database, max_depth=1, 
                load_scalers_from=one_layer_scaler_path_name, lte = database.tasks[task_id].auto, text_to_pca=True, text_to_pca_dim=3, task_idx=task_id) 
                ## configure model and loaders
                ta_th_at_dfs_2 = generate_model_task_embedding(two_layer_rdb_dataset, task_id, basic_config, method_type = 'training-head', model_name = 'auto-transfer', anchor_config = DFS_ANCHOR_CONFIG, model_type = 'dfs')
                ta_th_at_dfs_1 = generate_model_task_embedding(one_layer_rdb_dataset, task_id, basic_config, method_type = 'training-head', model_name = 'auto-transfer', anchor_config = DFS_ANCHOR_CONFIG, model_type = 'dfs')

                task_embedding_for_this_task = {
                    "ta_th_at_dfs_2": ta_th_at_dfs_2,
                    "ta_th_at_dfs_1": ta_th_at_dfs_1,
                }
                torch.save(task_embedding_for_this_task, dfs_path)
                
                # Make sure intermediate tensors are reclaimed before the next heavy run
                clear_task_memory()
    
    # Clear memory after processing all tasks in a database
    clear_task_memory()
    logger.info("Completed processing all tasks in database")
                        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=str, default="")
    parser.add_argument("--db-name", type=str, default="rel-f1")
    parser.add_argument("--task-name", type=str, default="driver-dnf")
    parser.add_argument("--path", type=str, default="", help="If not null string, \
                        this can load custom dataset")
    parser.add_argument("--root-dir", type=str, default="taskemb")
    parser.add_argument("--basic-config", type=str, default="configs/default/task.yaml")
    parser.add_argument("--time", default=5, help="The time to run the heuristics and calculate average")
    parser.add_argument("--max_sample", default=15000, help="The maximum number of samples to run the heuristics")
    args = parser.parse_args()
    logger.add("heuristics.log", rotation="1 MB") 
    basic_config = load_yaml(args.basic_config)
    if args.dataset_config:
        data_config = load_yaml(args.dataset_config)
    else:
        data_config = {
            "dbs": [
                {
                    "db_name": args.db_name,
                    "task_name": [args.task_name],
                    "path": args.path
                }
            ]
        }
    logger.info("Starting execution")
    import wandb
    wandb.init(mode="disabled")
    task_embedding_execution(basic_config, data_config, args)
