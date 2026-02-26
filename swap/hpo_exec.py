from swap.execution import SwapDB, seed_everything, sync_train_process, sync_train_process_fs, SameParams, SameParamsMoreGPU, load_yaml, save_dict_as_yaml
from data.mtaskdataset import prepare_transductive_rdb_datasets_save_memory as prepare_transductive_rdb_datasets
from utils.database import create_swap_db
from loguru import logger
import argparse
import os
import json
import torch
import sys


def execution(search_params, basic_config, data_config, save_model, recovery_file, recovery_directory,
              result_db: SwapDB, device: str = 'cuda', noexit: bool = False, text_to_pca: bool = False,
              text_to_pca_dim: int = 3, eval_val: bool = False, use_lightning: bool = False,
              lightning_devices: int = 1, trainer_precision: str = "32", max_epochs: int | None = None,
              resume_from: str | None = None):
    """
        Called by swap agent
        Args:
            search_params: a dictionary of search parameters
            output_prefix: the prefix of the output file
            save_model: whether to save the model
            recovery_file: the file name of the recovery file
        Returns:
    """
    seed_everything(basic_config['seed'])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    database = next(prepare_transductive_rdb_datasets(data_config, basic_config))
    ## exit code:
    ## 0: success
    ## 5: unexpected memory error
    ## 6: cuda out of memory
    result_db.test = False
    result = None
    try: 
        model_config = {
            **basic_config,
            **data_config
        }
        data_config = data_config["dbs"][0]
        model_config['model_params'] = search_params
        full_name = f"{data_config['db_name']}_{data_config['task_name'][0]}"        
        if search_params['model_type'] == 'rdl':
            logger.debug(f"Running RDL with search params: {search_params}")
            result = sync_train_process(database, model_config, full_name, save_model, recovery_file, recovery_directory, result_db, device, eval_val)
        elif search_params['model_type'] == 'fs' or search_params['model_type'] == 'dfs':
            logger.debug(f"Running DFS with search params: {search_params}")
            result = sync_train_process_fs(database, model_config, full_name, save_model, recovery_file, result_db, text_to_pca, text_to_pca_dim)
        elif search_params['model_type'] == 'ng':
            result = {}
    except MemoryError as e:
        logger.error(f"Memory error. Exiting.")
        result = {}
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            result_db.add_a_row(model_config, {}, "memory_out", -1)
            logger.error(f"CUDA out of memory. Exiting.")
            result = {}
        else:
            logger.error(f"Error in execution: {e}")
            raise e
    finally:
        # Proactively release GPU/CPU memory between trials
        try:
            del database
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
    return result

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
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--noexit", action="store_true")
    parser.add_argument("--text-to-pca", action="store_true")
    parser.add_argument("--text-to-pca-dim", type=int, default=3)
    parser.add_argument("--lightning", action="store_true")
    parser.add_argument("--trainer-precision", type=str, default="32")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
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
    result_db.test = args.test
    if not rdl_args.get("model_type", None):
        ## default to rdl
        rdl_args["model_type"] = "rdl"
    rdl_args["time_free"] = rdl_args.get("time_free", False)
    logger.info("Starting execution")
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    execution(
        rdl_args,
        basic_config,
        data_config,
        args.save_model,
        args.recovery_file,
        args.recovery_directory,
        result_db,
        args.device,
        args.noexit,
        args.text_to_pca,
        args.text_to_pca_dim,
        eval_val=False,
        use_lightning=args.lightning,
        lightning_devices=args.devices,
        trainer_precision=args.trainer_precision,
        max_epochs=args.max_epochs,
        resume_from=args.resume_from,
    )
