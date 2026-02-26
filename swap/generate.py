import torch
import os
## this is some local machine specific settings
os.environ.setdefault('XDG_CACHE_HOME', 'cache_data')
import argparse
import warnings
warnings.filterwarnings("ignore")
from loguru import logger
from data.mtaskdataset import MultiTabularDataset, TaskType, prepare_transductive_rdb_datasets
import sys
from utils.type import seed_everything
import sys
import json
from utils.database import SwapDB, create_swap_db
from utils.misc import load_yaml, save_dict_as_yaml
from models.dfs import generate_dfs_data_pre_post
from relbench.base import AutoCompleteTask
import torch_geometric 
import time
import signal
torch_geometric.backend.use_segment_matmul = False

# Global variable to track start time
START_TIME = None
MAX_PREPROCESSING_TIME = 2 * 60 * 60  # 2 hours in seconds
EXIT_CODE_PREPROCESSING_TIMEOUT = 7  # Custom exit code for preprocessing timeout

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    logger.error(f"Preprocessing timeout: exceeded {MAX_PREPROCESSING_TIME/3600:.1f} hours")
    sys.exit(EXIT_CODE_PREPROCESSING_TIMEOUT)

def check_timeout():
    """Check if preprocessing has exceeded the time limit"""
    global START_TIME
    if START_TIME is None:
        return
    
    elapsed_time = time.time() - START_TIME
    if elapsed_time > MAX_PREPROCESSING_TIME:
        logger.error(f"Preprocessing timeout: {elapsed_time/3600:.2f} hours elapsed, exceeding {MAX_PREPROCESSING_TIME/3600:.1f} hour limit")
        sys.exit(EXIT_CODE_PREPROCESSING_TIMEOUT)
    
    # Log progress every 30 minutes
    if int(elapsed_time) % 1800 == 0 and elapsed_time > 0:
        logger.info(f"Preprocessing in progress: {elapsed_time/3600:.2f} hours elapsed")

def sync_train_process_fs(database: MultiTabularDataset, model_config: dict, full_name, 
                       save_model: bool, recovery_file: str, result_db: SwapDB, easy: bool = False, text_to_pca: bool = False, text_to_pca_dim: int = 3):
    global START_TIME
    START_TIME = time.time()
    
    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_PREPROCESSING_TIME)
    
    try:
        autog_in_name = ""
        scaler_path_name = f"scaler_{database.database.name}_{database.tasks[0].name}_{model_config['model_params']['dfs_layer']}"
        lte = isinstance(database.tasks[0], AutoCompleteTask)
        
        logger.info("Starting DFS data generation...")
        check_timeout()  # Check before starting
        
        logger.info("Using LightGBM model, generating DFS data with pre/post processing...")
        generate_dfs_data_pre_post("datasets/dfs", database, max_depth=model_config['model_params']['dfs_layer'],
                                                    load_scalers_from=scaler_path_name,
                                                    mode='easy' if easy else 'normal', lte=lte, text_to_pca=text_to_pca, text_to_pca_dim=text_to_pca_dim,
                                                    agg_primitives=model_config['model_params'].get('agg_primitives', None))
    
        # Cancel the alarm since preprocessing completed successfully
        signal.alarm(0)
        elapsed_time = time.time() - START_TIME
        logger.info(f"Generation completed successfully in {elapsed_time/60:.2f} minutes")
        
    except SystemExit as e:
        # Re-raise SystemExit to propagate the exit code
        raise e
    except Exception as e:
        # Cancel the alarm on any other exception
        signal.alarm(0)
        logger.error(f"Error during preprocessing: {e}")
        raise e
    finally:
        # Ensure alarm is cancelled
        signal.alarm(0)



def execution(search_params, basic_config, data_config, save_model, recovery_file, recovery_directory, result_db: SwapDB, device: str = 'cuda', easy: bool = False, text_to_pca: bool = False, text_to_pca_dim: int = 3):
    """
        Called by swap agent
        Args:
            search_params: a dictionary of search parameters
            output_prefix: the prefix of the output file
            save_model: whether to save the model
            recovery_file: the file name of the recovery file
        Returns:
    """
    try:
        database = prepare_transductive_rdb_datasets(data_config, basic_config)[0]
        model_config = {
            **basic_config,
            **data_config
        }
        data_config = data_config["dbs"][0]
        model_config['model_params'] = search_params
        full_name = f"{data_config['db_name']}_{data_config['task_name'][0]}"
        sync_train_process_fs(database, model_config, full_name, 
                              save_model, recovery_file, result_db, easy, text_to_pca, text_to_pca_dim)
    except SystemExit as e:
        # Propagate the exit code
        sys.exit(e.code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-name", type=str, default="rel-f1")
    parser.add_argument("--task-name", type=str, default="driver-dnf")
    parser.add_argument("--path", type=str, default="", help="If not null string, \
                        this can load custom dataset")
    parser.add_argument("--rdl-params", type=str, default="{}")
    parser.add_argument("--params_type", type=str, default="json", choices=["json", "file"])
    parser.add_argument("--recovery-file", type=str, default="")
    parser.add_argument("--recovery-directory", type=str, default="modelcache")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--basic-config", type=str, default="configs/default/basic.yaml")
    parser.add_argument("--debug-save", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--easy", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--text-to-pca", action="store_true")
    parser.add_argument("--text-to-pca-dim", type=int, default=3)
    args = parser.parse_args()
    basic_config = load_yaml(args.basic_config)
    seed_everything(basic_config['seed'])
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if args.params_type == "json":
        rdl_args = json.loads(args.rdl_params)
        if len(rdl_args) == 0:
            ## if no rdl, sampled one random 
            from utils.space import search
            rdl_args = search(model_type="rdl", task_type="binary_classification")
        if args.debug_save != "":
            save_dict_as_yaml(rdl_args, args.debug_save)
    else:
        rdl_args = load_yaml(args.rdl_params)
    data_config = {
        "dbs": [
            {
                "db_name": args.db_name,
                "task_name": [args.task_name],
                "path": args.path
            }
        ]
    }
    result_db = create_swap_db(
        backend=basic_config.get("db_backend", "sqlite"),
        path_to_db=basic_config["db_location"],
        collection_name=basic_config["result_db_name"],
        db_name=basic_config["db_name"],
    )
    if not rdl_args.get("model_type", None):
        ## default to rdl
        rdl_args["model_type"] = "rdl"
    logger.info("Starting execution")
    execution(rdl_args, basic_config, data_config, 
              args.save_model, args.recovery_file, 
              args.recovery_directory, result_db, args.device, args.easy, args.text_to_pca, args.text_to_pca_dim)