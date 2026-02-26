import queue
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from utils.space import RDL_SEARCH_SPACE, DFS_SEARCH_SPACE
from typing import Optional, Sequence, Union, List, Any, Dict
from hyperopt import STATUS_OK
from loguru import logger

class DevicePool:
    """Thread-safe pool of device identifiers for parallel trial execution."""

    def __init__(self, devices):
        self._devices = list(devices)
        if not self._devices:
            raise ValueError("Device pool requires at least one entry")
        self._queue = queue.Queue()
        for dev in self._devices:
            self._queue.put(dev)

    def __len__(self):
        return len(self._devices)

    @contextmanager
    def acquire(self, timeout=None):
        device = self._queue.get(timeout=timeout)
        try:
            yield device
        finally:
            self._queue.put(device)

    
def rng_to_seed(rng):
    """Convert a numpy/random generator or int into an Optuna-compatible seed."""
    if rng is None:
        return None
    if isinstance(rng, np.random.RandomState):
        return int(rng.randint(0, 2**31 - 1))
    if hasattr(np.random, 'Generator') and isinstance(rng, np.random.Generator):
        return int(rng.integers(0, 2**31 - 1))
    if isinstance(rng, (int, np.integer)):
        return int(rng)
    try:
        return int(rng)
    except (TypeError, ValueError):
        return int(np.random.randint(0, 2**31 - 1))


def weights_to_repeated_choices(values, prior_map=None, prior_strength=0.7):
    """Simulate weighted categorical sampling by repeating values according to weights."""
    unique_values = list(dict.fromkeys(values))
    if not unique_values:
        return []

    base = np.ones(len(unique_values), dtype=float)
    base /= base.sum()

    weights = base
    if prior_map:
        prior = np.asarray([float(prior_map.get(val, 0.0)) for val in unique_values], dtype=float)
        if prior.sum() > 0:
            prior /= prior.sum()
            weights = (1.0 - prior_strength) * base + prior_strength * prior

    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0 or not np.isfinite(weights).all():
        weights = base
    else:
        weights = weights / weights.sum()

    counts = np.maximum(1, np.round(weights * 100).astype(int))
    repeated = []
    for value, count in zip(unique_values, counts):
        repeated.extend([value] * int(count))
    return repeated or unique_values


def config_to_key(config):
    return tuple(sorted(config.items()))


def build_experiment3_catalog(allowed_dfs_models=None):
    """Return branch-wise discrete choices for Experiment 3 sampling."""
    allowed_dfs_models = allowed_dfs_models or ['ft_transformer']

    rdl_space = deepcopy(RDL_SEARCH_SPACE)
    dfs_space = deepcopy(DFS_SEARCH_SPACE)

    rdl_full = rdl_space["full_entities"]
    rdl_model = rdl_space["model_config"]
    gnn_cfg = {k: list(v) for k, v in rdl_model["gnn_config"].items()}
    sampler_cfg = {k: list(v) for k, v in rdl_model["sampler_config"].items()}
    opt_cfg = {k: list(v) for k, v in rdl_model["optimizer_config"].items()}

    gnn_cfg['skip_connection'] = [False]
    gnn_cfg['post_sf'] = ['none']
    sampler_cfg['loader_type'] = ['node']

    batch_choices = list(rdl_model.get('batch_size', [128]))
    batch_choices = [128] if 128 in batch_choices else batch_choices
    torch_frame_choices = list(rdl_model.get('torch_frame_model_cls', ['resnet']))

    rdl_branch = {
        'pre_sf': list(rdl_full['pre_sf']),
        'mpnn_type': list(rdl_full['mpnn_type']),
        'post_sf': list(rdl_full['post_sf'].get('node', ['none'])),
        'hidden_channels': gnn_cfg['hidden_channels'],
        'num_heads': gnn_cfg['num_heads'],
        'dropout': gnn_cfg['dropout'],
        'norm': gnn_cfg['norm'],
        'aggregation': gnn_cfg['aggregation'],
        'skip_connection': gnn_cfg['skip_connection'],
        'temporal': sampler_cfg['temporal'],
        'num_neighbors': sampler_cfg['num_neighbors'],
        'num_layers': sampler_cfg['num_layers'],
        'loader_type': sampler_cfg['loader_type'],
        'lr': opt_cfg['lr'],
        'weight_decay': opt_cfg['weight_decay'],
        'gamma': opt_cfg['gamma'],
        'batch_size': batch_choices,
        'torch_frame_model_cls': torch_frame_choices,
    }

    dfs_full = dfs_space["full_entities"]
    dfs_model = dfs_space["model_config"]
    dfs_auto = dfs_space["auto_config"]
    dfs_opt = dfs_space["optimizer_config"]

    dfs_branch = {
        'torch_frame_model_cls': list(allowed_dfs_models),
        'dfs_layer': list(dfs_full['dfs_layer']),
        'hidden_size': list(dfs_model['hidden_size']),
        'dropout': list(dfs_model['dropout']),
        'num_layers': list(dfs_model['num_layers']),
        'attn_dropout': list(dfs_model['attn_dropout']),
        'num_heads': list(dfs_model['num_heads']),
        'normalization': list(dfs_model['normalization']),
        'max_train_samples': list(dfs_auto['max_train_samples']),
        'training_sample_selection_strategy': list(dfs_auto['training_sample_selection_strategy']),
        'num_trials': list(dfs_auto['num_trials']),
        'temporal': list(dfs_auto['temporal']),
        'lr': list(dfs_opt['lr']),
        'weight_decay': list(dfs_opt['weight_decay']),
        'gamma': list(dfs_opt['gamma']),
        'batch_size': list(dfs_space['batch_size']),
        'negative_sampling_ratio': list(dfs_space['negative_sampling_ratio']),
    }

    return {'rdl': rdl_branch, 'dfs': dfs_branch}


def load_heuristics_features(csv_path='results/task_feature_summary.csv', db_name=None, task_name=None):
    """
    Load heuristics related features from a pre-computed CSV file.

    Args:
        csv_path: Path to the task feature summary CSV file.
        db_name: Optional database filter.
        task_name: Optional task filter.

    Returns:
        Tuple of three pandas.DataFrames:
            1) heuristic_df  – table stats + model-free corr features + model-based heuristics
            2) autotransfer_df – atdfs / atrdl auto-transfer features
            3) simple_df – simple baseline metrics pulled from model-free payloads
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    full_df = pd.read_csv(csv_path)

    # CSV uses 'dataset_name'; downstream expects 'db_name'
    if 'dataset_name' in full_df.columns and 'db_name' not in full_df.columns:
        full_df.rename(columns={'dataset_name': 'db_name'}, inplace=True)

    # Apply filters
    if db_name:
        full_df = full_df[full_df['db_name'] == db_name]
    if task_name:
        full_df = full_df[full_df['task_name'] == task_name]

    if full_df.empty:
        empty = pd.DataFrame(columns=["db_name", "task_name"])
        return empty.copy(), empty.copy(), empty.copy()

    id_cols = ['db_name', 'task_name']

    # Metadata/performance columns (not features)
    skip_cols = {
        'task_type', 'rdl_test_metric_selected_by_val',
        'dfs_test_metric_selected_by_val',
        'rdl_test_metric_selected_by_test',
        'dfs_test_metric_selected_by_test',
        'rdl_advantage_selected_by_val',
        'rdl_advantage_selected_by_test',
    }

    # AutoTransfer columns → autotransfer_df
    at_cols = [c for c in full_df.columns if c.startswith('at_')]

    # Simple baseline columns → simple_df
    simple_prefixes = ("global_zero", "global_mean", "global_median",
                       "entity_mean", "entity_median", "majority")
    simple_cols = [c for c in full_df.columns
                   if any(c.startswith(p) for p in simple_prefixes)]

    # Everything else (excluding id, skip, at, simple) → heuristic_df
    excluded = set(id_cols) | skip_cols | set(at_cols) | set(simple_cols)
    heur_cols = [c for c in full_df.columns if c not in excluded]

    heuristic_df = full_df[id_cols + heur_cols].copy().reset_index(drop=True)
    autotransfer_df = full_df[id_cols + at_cols].copy().reset_index(drop=True)
    simple_df = full_df[id_cols + simple_cols].copy().reset_index(drop=True)

    return heuristic_df, autotransfer_df, simple_df

# 2) Weighted two-step sampling builders (use these for priors!)
def build_execution_config_from_search_config(search_config, model_family_type, task_type):
    """Convert a sampled search configuration into the execution format."""
    from utils.space import search_rdl, search_dfs
    kind = model_family_type[0] if isinstance(model_family_type, (list, tuple)) else model_family_type

    def convert_to_normal_type(x):
        """
        Convert numpy types to normal Python types recursively.
        Handles nested dictionaries and lists.
        """
        import numpy as np
        
        if isinstance(x, dict):
            return {key: convert_to_normal_type(value) for key, value in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(convert_to_normal_type(item) for item in x)
        elif isinstance(x, np.integer):
            return int(x)
        elif isinstance(x, np.floating):
            return float(x)
        elif isinstance(x, np.bool_):
            return bool(x)
        elif isinstance(x, np.str_):
            return str(x)
        elif isinstance(x, np.ndarray):
            return x.tolist()
        else:
            return x 
    
    # Convert numpy types to normal Python types
    search_config = convert_to_normal_type(search_config)
    
    if kind == 'rdl':
        # Build baseline RDL config for the task
        execution_config = search_rdl(
            task_type=task_type,
            pre_sf=search_config['pre_sf'],
            mpnn_type=search_config['mpnn_type'], 
            post_sf=search_config['post_sf']
        )
        
        # Override with sampled choices
        execution_config['gnn_config']['hidden_channels'] = search_config['hidden_channels']
        execution_config['gnn_config']['num_heads'] = search_config['num_heads']
        execution_config['gnn_config']['dropout'] = search_config['dropout']
        execution_config['gnn_config']['norm'] = search_config['norm']
        execution_config['gnn_config']['aggregation'] = search_config['aggregation']
        execution_config['gnn_config']['skip_connection'] = search_config['skip_connection']
        
        execution_config['sampler_config']['temporal'] = search_config['temporal']
        execution_config['sampler_config']['num_neighbors'] = search_config['num_neighbors']
        execution_config['sampler_config']['num_layers'] = search_config['num_layers']
        execution_config['sampler_config']['loader_type'] = search_config['loader_type']
        
        execution_config['optimizer_config']['lr'] = search_config['lr']
        execution_config['optimizer_config']['weight_decay'] = search_config['weight_decay']
        execution_config['optimizer_config']['gamma'] = search_config['gamma']
        
        execution_config['batch_size'] = search_config['batch_size']
        execution_config['torch_frame_model_cls'] = search_config['torch_frame_model_cls']
        
    elif kind == 'dfs':
        # Build baseline DFS config for the task
        execution_config = search_dfs(task_type, selected_model='ft_transformer')
        
        # Override with sampled choices
        execution_config['torch_frame_model_cls'] = search_config['torch_frame_model_cls']
        execution_config['dfs_layer'] = search_config['dfs_layer']
        
        # Model configuration
        execution_config['tab_nn_config'] = {
            'hidden_size': search_config['hidden_size'],
            'dropout': search_config['dropout'],
            'num_layers': search_config['num_layers'],
            'attn_dropout': search_config['attn_dropout'],
            'num_heads': search_config['num_heads'],
            'normalization': search_config['normalization']
        }
        
        # Auto configuration
        execution_config['auto_config'] = {
            'max_train_samples': search_config['max_train_samples'],
            'training_sample_selection_strategy': search_config['training_sample_selection_strategy'],
            'num_trials': search_config['num_trials'],
            'temporal': search_config['temporal']
        }
        
        # Optimizer configuration
        execution_config['optimizer_config']['lr'] = search_config['lr']
        execution_config['optimizer_config']['weight_decay'] = search_config['weight_decay']
        execution_config['optimizer_config']['gamma'] = search_config['gamma']
        
        execution_config['batch_size'] = search_config['batch_size']
        execution_config['negative_sampling_ratio'] = search_config['negative_sampling_ratio']
    
    else:
        raise ValueError(f"Unknown model family type: {model_family_type}")
    
    return execution_config

def normalize_device_name(device: Union[str, int, torch.device, None]) -> Optional[str]:
    if device is None:
        return None
    if isinstance(device, torch.device):
        if device.type == 'cuda':
            if device.index is None:
                return 'cuda'
            return f'cuda:{device.index}'
        return device.type
    if isinstance(device, int):
        return f'cuda:{device}'
    value = str(device).strip()
    if not value:
        return None
    if value.startswith('cuda') or value == 'cpu':
        return value
    if value.isdigit():
        return f'cuda:{value}'
    return value

def resolve_devices(devices: Optional[Sequence[Union[str, int, torch.device]]]) -> List[str]:
    if devices is None:
        if torch.cuda.is_available():
            return [f'cuda:{idx}' for idx in range(torch.cuda.device_count())]
        return []

    if isinstance(devices, str):
        tokens = [tok.strip() for tok in devices.split(',') if tok.strip()]
    else:
        tokens = list(devices)

    normalized: List[str] = []
    seen = set()
    for token in tokens:
        name = normalize_device_name(token)
        if name is None or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def create_objective_function(database_name, task_name, task_type, basic_config_path="configs/default/basic.yaml", hpo_instance=None, skip_unknown_params=False, use_meta_predictor=False):
    """
    Create objective function that integrates with execution.py
    
    Args:
        database_name: Database name
        task_name: Task name  
        task_type: Task type
        basic_config_path: Path to basic config file
        hpo_instance: TaskEmbeddingHPO instance for checking old experiments
    """
    from swap.hpo_exec import execution
    from utils.database import create_swap_db
    from utils.misc import load_yaml
    def objective(config, device_override=None):
        """
        Objective function that runs actual model training via execution.py
        """
        import contextlib
        import io

        # Suppress stdout/stderr to prevent tqdm issues in HPO context
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec_device = device_override or ('cuda' if torch.cuda.is_available() else 'cpu')
            if isinstance(exec_device, torch.device):
                exec_device_str = str(exec_device)
            elif isinstance(exec_device, int):
                exec_device_str = f'cuda:{exec_device}'
            else:
                exec_device_str = str(exec_device).strip() or ('cuda' if torch.cuda.is_available() else 'cpu')

            if exec_device_str.startswith('cuda') and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(torch.device(exec_device_str))
                except (AssertionError, RuntimeError, ValueError):
                    if ':' in exec_device_str:
                        try:
                            torch.cuda.set_device(int(exec_device_str.split(':', 1)[1]))
                        except Exception:
                            pass
            elif exec_device_str == 'cuda' and not torch.cuda.is_available():
                exec_device_str = 'cpu'

            # Extract model type from unified config
            model_family_type = config.get('model_type', 'rdl')
            if model_family_type == 'fs':
                model_family_type = 'dfs'

            # Convert sampled config to execution format
            execution_config = build_execution_config_from_search_config(
                config, model_family_type, task_type
            )

            execution_config['model_type'] = model_family_type

            # Load basic config
            basic_config = load_yaml(basic_config_path)

            # Create data config
            data_config = {
                "dbs": [
                    {
                        "db_name": database_name,
                        "task_name": [task_name],
                        "path": ""
                    }
                ]
            }

            alg = getattr(hpo_instance, 'current_algorithm', '')

            basic_config["result_db_name"] = "loss_metrics"

            result_db = create_swap_db(
                backend=basic_config.get("db_backend", "sqlite"),
                path_to_db=basic_config["db_location"],
                collection_name=basic_config["result_db_name"],
                db_name=basic_config["db_name"],
            )

            # Create model config for checking old experiments
            model_config = {
                **basic_config,
                **data_config,
                'model_params': execution_config
            }


            def _has_primary_metric(metric_dict: Dict[str, Any]) -> bool:
                if not isinstance(metric_dict, dict) or not metric_dict:
                    return False
                if task_type == 'binary_classification':
                    return 'roc_auc' in metric_dict or 'auroc' in metric_dict or 'auc' in metric_dict
                if task_type == 'multiclass_classification':
                    return 'accuracy' in metric_dict
                if task_type == 'regression':
                    return 'mae' in metric_dict
                return True

            def _compute_loss_from_metric(metric_dict: Dict[str, Any]) -> float:
                if task_type == 'binary_classification':
                    if 'roc_auc' in metric_dict:
                        return -float(metric_dict.get('roc_auc', 0.0))
                    elif 'auroc' in metric_dict:
                        return -float(metric_dict.get('auroc', 0.0))
                    # return -float(metric_dict.get('roc_auc', 0.0))
                if task_type == 'multiclass_classification':
                    return -float(metric_dict.get('accuracy', 0.0))
                if task_type == 'regression':
                    return float(metric_dict.get('mae', 1.0))
                return 1.0

            def _format_cached_response(val_metric, test_metric, cache_meta=None):
                has_metric = _has_primary_metric(val_metric)
                response = {
                    'loss': 1.0 if not has_metric else float(_compute_loss_from_metric(val_metric)),
                    'status': STATUS_OK,
                    'val_metric': val_metric if has_metric else {},
                    'test_metric': test_metric if has_metric else {},
                    'exec_config': execution_config,
                    'database_name': database_name,
                    'task_name': task_name,
                    'task_type': task_type,
                    'model_type': model_family_type,
                    'skipped': not has_metric,
                    'device': exec_device_str,
                }
                if cache_meta is not None:
                    response['match_type'] = cache_meta.get('match_type')
                    response['num_matches'] = cache_meta.get('num_matches')
                return response

            cached_result = hpo_instance.get_old_exps(model_config) if hpo_instance is not None else None

            # Check for existing experiments using HPO instance
            if hpo_instance is not None:
                if use_meta_predictor:
                    artifacts = result_db.find_trial_artifacts_by_config(model_config)
                    if artifacts:
                        doc = artifacts[-1]
                        v = (doc.get('results') or {}).get('val_metric', {})
                        t = (doc.get('results') or {}).get('test_metric', {})
                        return _format_cached_response(v, t)
                    # if cached_result is not None:
                    #     return _format_cached_response(
                    #         cached_result.get('val_metric', {}),
                    #         cached_result.get('test_metric', {}),
                    #         cache_meta=cached_result
                    #     )
                else:
                    if cached_result is not None:
                        return _format_cached_response(
                            cached_result.get('val_metric', {}),
                            cached_result.get('test_metric', {}),
                            cache_meta=cached_result
                        )
                    else:
                        # Try to find match in result_db
                        artifacts = result_db.find_trial_artifacts_by_config(model_config)
                        artifacts = [a for a in artifacts if a.get('status') == 'completed' and a.get('artifact_type', "") == 'trial_artifacts']
                        if artifacts:
                            doc = artifacts[-1]
                            v = (doc.get('results') or {}).get('val_metric', {})
                            t = (doc.get('results') or {}).get('test_metric', {})
                            return _format_cached_response(v, t)
            # If no cached match and skipping is enabled, bail out early
            if skip_unknown_params:
                logger.debug(
                    f"Skipping uncached configuration for {database_name}::{task_name} (model={model_family_type}) due to skip_unknown_params",
                )
                return {
                    'loss': 1.0,
                    'status': STATUS_OK,
                    'val_metric': {},
                    'test_metric': {},
                    'skipped': True,
                    'device': exec_device_str,
                }
            try:
                result = execution(
                    search_params=execution_config,
                    basic_config=basic_config,
                    data_config=data_config,
                    save_model=True,
                    recovery_file="",
                    recovery_directory="",
                    result_db=result_db,
                    device=exec_device_str,
                    noexit=True,
                    eval_val=False,
                    text_to_pca=True,
                )
            except SystemExit as exc:
                exit_code = getattr(exc, 'code', None)
                logger.warning(
                    "Execution requested exit (code=%s) for %s::%s; marking trial as skipped",
                    exit_code,
                    database_name,
                    task_name,
                )
                return {
                    'loss': 1.0,
                    'status': STATUS_OK,
                    'val_metric': {},
                    'test_metric': {},
                    'skipped': True,
                    'device': exec_device_str,
                    'exit_code': exit_code,
                }
            except Exception as e:
                # OOM: mark as skipped to avoid counting toward budget
                if 'CUDA out of memory' in str(e):
                    return {
                        'loss': 1.0,
                        'status': STATUS_OK,
                        'val_metric': {},
                        'test_metric': {},
                        'skipped': True,
                        'device': exec_device_str,
                    }
                # Unknown error: count as executed failure
                return {
                    'loss': 1.0,
                    'status': STATUS_OK,
                    'val_metric': {},
                    'test_metric': {},
                    'skipped': False,
                    'device': exec_device_str,
                }
            
            # Get results from database
            results = result.get('results', {}) 
            if not results:
                # Treat empty results as OOM or fatal failure; skip this trial
                return {
                    'loss': 1.0,
                    'status': STATUS_OK,
                    'val_metric': {},
                    'test_metric': {},
                    'skipped': True,
                    'device': exec_device_str,
                }
            # Extract validation metric
            val_metric = results.get('val_metric', {}) or {}
            test_metric = results.get('test_metric', {}) or {}

            if not _has_primary_metric(val_metric):
                logger.warning("Missing validation metric for %s::%s (model=%s); marking trial as skipped.",
                               database_name, task_name, model_family_type)
                return {
                    'loss': 1.0,
                    'status': STATUS_OK,
                    'val_metric': {},
                    'test_metric': {},
                    'skipped': True,
                    'device': exec_device_str,
                }
                            
            # Extract primary metric based on task type
            if task_type == 'binary_classification':
                loss = -val_metric['roc_auc']
            elif task_type == 'multiclass_classification':
                loss = -val_metric['accuracy']
            elif task_type == 'regression':
                loss = val_metric['mae']
            else:
                loss = 1.0

            logger.debug(f"Execution Config: {execution_config}")
            logger.info(f"Loss: {loss}, Val Metric: {val_metric}, Test Metric: {test_metric}")
            
            out = {
                'loss': float(loss), 'status': STATUS_OK,
                'val_metric': val_metric, 'test_metric': test_metric, 'skipped': False,
                'exec_config': execution_config,
                'database_name': database_name,
                'task_name': task_name,
                'task_type': task_type,
                'model_type': model_family_type,
                'device': exec_device_str,
            }
            # Save artifact if we saved model weights (ours-mode)
            # if (hpo_instance is not None) and result.get('model_weight_name'):
            result_db.save_trial_artifacts({
                'artifact_type': 'trial_artifacts',
                'config': model_config,
                'results': {'val_metric': val_metric, 'test_metric': test_metric},
                'status': 'completed',
                'model_weight_name': result.get('model_weight_name')
            })
            # Best-effort cleanup after each trial
            try:
                import gc, torch as _torch
                gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    _torch.cuda.synchronize()
            except Exception:
                pass
            return out    
    return objective

def val_score_of_result(result_dict, task_type):
    vm = result_dict.get('val_metric', {}) or {}
    if not vm:
        return None
    if not isinstance(vm, dict):
        return float(vm)
    else:
        if 'classification' in task_type:
            return float(vm.get('roc_auc', 0.0))
        elif 'regression' in task_type:
            return float(vm.get('mae', 1e9))
        else:
            return None

def compute_landscape_metrics_for_trial(result_dict, basic_config_path="configs/default/basic.yaml"):
    from swap.hpo_exec import execution
    from utils.misc import load_yaml
    from utils.database import create_swap_db
    from data.mtaskdataset import prepare_transductive_rdb_datasets
    from swap.search_and_plot import plot_loss_landscape_and_calculate_metrics, load_torch_frame_loader
    from copy import deepcopy
    import os

    exec_config = result_dict.get('exec_config', {})
    if not exec_config:
        return None
    db = result_dict.get('database_name')
    task = result_dict.get('task_name')
    model_type = exec_config.get('model_type', result_dict.get('model_type','rdl'))
    task_type = result_dict.get('task_type','binary_classification')
    if model_type == 'dfs':
        tf_model = exec_config.get('torch_frame_model_cls') or exec_config.get('model_params', {}).get('torch_frame_model_cls')
        if tf_model and tf_model != 'ft_transformer':
            logger.warning(f"Skipping landscape metrics for unsupported DFS model '{tf_model}'")
            return None

    basic_config = load_yaml(basic_config_path)
    data_config = {"dbs": [{"db_name": db, "task_name": [task], "path": ""}]}
    basic_config["result_db_name"] = "loss_metrics"
    result_db = create_swap_db(
        backend=basic_config.get("db_backend", "sqlite"),
        path_to_db=basic_config["db_location"],
        collection_name=basic_config["result_db_name"],
        db_name=basic_config["db_name"],
    )

    # Compose full model_config matching execution storage
    model_config = {**basic_config, **data_config}
    model_config['model_params'] = exec_config

    # 0) Try to reuse existing artifacts and performance
    existing_artifacts = result_db.find_trial_artifacts_by_config(model_config)
    existing_perf_rows = result_db.get_rows_with_the_same_search_params(model_config)
    model_weight_name = None
    val_metric = None
    test_metric = None
    if existing_artifacts:
        art = existing_artifacts[-1]
        model_weight_name = art.get('model_weight_name')
        res = art.get('results', {}) or {}
        val_metric = res.get('val_metric')
        test_metric = res.get('test_metric')
    if (val_metric is None or test_metric is None) and existing_perf_rows:
        pr = existing_perf_rows[-1]
        res = pr.get('results', {}) or {}
        val_metric = val_metric or res.get('val_metric') or res.get('val_metrics')
        test_metric = test_metric or res.get('test_metric') or res.get('test_metrics')

    # 1) If missing any, run experiment, save model and artifacts
    if (model_weight_name is None) or (not os.path.exists(model_weight_name)):
        raise ValueError("Model weight name, val metric, test metric, or model weight file does not exist")

    # 2) Try to reuse stored loss-landscape metrics
    existing_lm = result_db.find_loss_metrics_by_config(model_config, tag='exp3')
    existing_lm = [x for x in existing_lm if x.get('artifact_type') == 'loss_metrics']
    if existing_lm:
        metrics = existing_lm[-1].get('loss_metrics') or {}
        return {
            'lipschitz_max': float(metrics.get('lipschitz_max', np.inf)),
            'sharpness_lambda_max': float(metrics.get('sharpness_lambda_max', np.inf)),
            'barrier_overall_max': float(metrics.get('barrier_overall_max', np.inf)),
        }

    # 3) Compute loss-landscape metrics and persist
    mtaskdataset = prepare_transductive_rdb_datasets(data_config, basic_config)[0]
    general_config = deepcopy(basic_config)
    general_config['model_params'] = exec_config

    if model_type == 'rdl':
        center_metrics = plot_loss_landscape_and_calculate_metrics(
            general_config, model_weight_name, model_type='rdl', dataset=mtaskdataset, device='cuda', task_name=task, plot_name='exp3'
        )
    else:
        tf_meta = load_torch_frame_loader(mtaskdataset, general_config, True, 3)
        center_metrics = plot_loss_landscape_and_calculate_metrics(
            general_config, model_weight_name, model_type='dfs', dataset=mtaskdataset, device='cuda',
            task_type=tf_meta['task_type'], num_classes=tf_meta['num_classes'], task_name=task,
            train_loader=tf_meta['train_loader'], val_loader=tf_meta['val_loader'], test_loader=tf_meta['test_loader'],
            train_dataset=tf_meta['train_dataset'], scaler_path=tf_meta['scaler_path'], clamp_min=tf_meta['clamp_min'], clamp_max=tf_meta['clamp_max'], plot_name='exp3_dfs'
        )

    if center_metrics is None:
        return None

    result_db.save_loss_metrics({
        'artifact_type': 'loss_metrics',
        'type': 'exp3',
        'task_name': task,
        'db_name': db,
        'config': model_config,
        'model_weight_name': model_weight_name,
        'loss_metrics': center_metrics,
    })

    # Proactive cleanup
    try:
        import gc, torch as _torch
        del mtaskdataset
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()
    except Exception:
        pass

    return {
        'lipschitz_max': float(center_metrics.get('lipschitz_max', np.inf)),
        'sharpness_lambda_max': float(center_metrics.get('sharpness_lambda_max', np.inf)),
        'barrier_overall_max': float(center_metrics.get('barrier_overall_max', np.inf)),
    }

def choose_best_trial_by_landscape(trial_results, task_type='binary_classification', top_k=10,
                                   metrics_provider=None):
    """Select the best trial using landscape metrics within a top-k validation window."""
    if not trial_results:
        return None
    filtered_results = list(trial_results)
    ## filter here, if dfs_layer presents, only keep the ones with dfs_layer = 2
    scored = []
    for r in filtered_results:
        s = val_score_of_result(r, task_type)
        if s is None:
            continue
        scored.append((s, r))
    if not scored:
        return None
    reverse = ('classification' in task_type)
    scored.sort(key=lambda x: x[0], reverse=reverse)
    candidates = [r for _, r in scored[:max(1, top_k)]]

    def metrics_for(r):
        return metrics_provider(r) if metrics_provider is not None else compute_landscape_metrics_for_trial(r)

    metrics_list = []
    for r in candidates:
        m = metrics_for(r)
        if not m:
            continue
        metrics_list.append((r, m))
    if not metrics_list:
        return candidates[0]

    def _val_metric_for(result):
        vm = result.get('val_metric', {}) or {}
        if isinstance(vm, dict):
            if task_type == 'binary_classification':
                return float(vm.get('roc_auc', -np.inf))
            if task_type == 'multiclass_classification':
                return float(vm.get('accuracy', -np.inf))
            if 'regression' in str(task_type):
                return -float(vm.get('mae', np.inf))
            if 'classification' in str(task_type):
                return float(vm.get('roc_auc', -np.inf))
            return None
        return float(vm) if vm is not None else None

    val_scores = []
    for cand, _ in metrics_list:
        score = _val_metric_for(cand)
        if score is not None:
            val_scores.append((score, cand))
    val_best = max(val_scores, key=lambda t: t[0])[1] if val_scores else candidates[0]

    # compute best possible test trial for reference
    def _test_metric_values(result):
        tm = result.get('test_metric') if isinstance(result, dict) else None
        if not isinstance(tm, dict) or not tm:
            return None, None
        if task_type == 'binary_classification':
            val = tm.get('roc_auc')
            return (val, val)
        if task_type == 'multiclass_classification':
            val = tm.get('accuracy')
            return (val, val)
        raw = tm.get('mae')
        if raw is None:
            return None, None
        return (-raw, raw)

    best_test_trial = None
    best_test_metric = None
    best_test_score = -np.inf
    for trial in filtered_results:
        score, raw_val = _test_metric_values(trial)
        if score is None:
            continue
        if score > best_test_score:
            best_test_score = score
            best_test_metric = raw_val
            best_test_trial = trial

    # weighted aggregation between validation score and landscape metrics
    weights = {
        'val': 0.4,
        'lipschitz_max': 0.2,
        'sharpness_lambda_max': 0.2,
        'barrier_overall_max': 0.2,
    }

    def _normalize(values, higher_is_better=True):
        finite_vals = [val for _, val in values if val is not None and np.isfinite(val)]
        if not finite_vals:
            default_val = 0.0 if higher_is_better else 1.0
            return {id_: default_val for id_, _ in values}
        min_v = min(finite_vals)
        max_v = max(finite_vals)
        if np.isclose(max_v, min_v):
            return {id_: 0.5 for id_, _ in values}
        if higher_is_better:
            return {
                id_: ((val - min_v) / (max_v - min_v) if val is not None and np.isfinite(val) else 0.0)
                for id_, val in values
            }
        return {
            id_: ((max_v - val) / (max_v - min_v) if val is not None and np.isfinite(val) else 0.0)
            for id_, val in values
        }

    val_norm = _normalize([(id(r), _val_metric_for(r)) for r, _ in metrics_list], higher_is_better=True)
    lipschitz_norm = _normalize([(id(r), m.get('lipschitz_max', np.nan)) for r, m in metrics_list], higher_is_better=False)
    sharpness_norm = _normalize([(id(r), abs(m.get('sharpness_lambda_max', np.nan))) for r, m in metrics_list], higher_is_better=False)
    barrier_norm = _normalize([(id(r), abs(m.get('barrier_overall_max', np.nan))) for r, m in metrics_list], higher_is_better=False)

    weighted_scores = {}
    for r, _ in metrics_list:
        rid = id(r)
        score = (
            weights['val'] * val_norm.get(rid, 0.0) +
            weights['lipschitz_max'] * lipschitz_norm.get(rid, 0.0) +
            weights['sharpness_lambda_max'] * sharpness_norm.get(rid, 0.0) +
            weights['barrier_overall_max'] * barrier_norm.get(rid, 0.0)
        )
        weighted_scores[rid] = score

    voted_best = max(metrics_list, key=lambda item: weighted_scores.get(id(item[0]), -np.inf))[0]

    return {
        'validation_top': val_best,
        'voted_top': voted_best,
        'best_test_trial': best_test_trial,
        'best_test_metric': best_test_metric,
        'finalists': metrics_list,
        'weighted_scores': weighted_scores,
    }

def aggregate_seed_runs_landscape(seed_runs, task_type='binary_classification', top_k=10, meta_for_landscape=False):
    voted_scores = []
    validation_scores = []
    best_test_scores = []
    selection_details = []
    for r in seed_runs:
        results = r.get('all_results', [])
        if not results:
            continue
        # preferred_family = None
        # if meta_for_landscape:
        #     preferred_family = r.get('preferred_family')
        selection_bundle = choose_best_trial_by_landscape(
            results,
            task_type=task_type,
            top_k=top_k,
            # preferred_family=preferred_family
        )
        if selection_bundle is None:
            continue
        selection_details.append(selection_bundle)

        voted_trial = selection_bundle.get('voted_top') if isinstance(selection_bundle, dict) else selection_bundle
        validation_trial = selection_bundle.get('validation_top') if isinstance(selection_bundle, dict) else selection_bundle
        if validation_trial is None:
            validation_trial = voted_trial

        def _score_from_trial(trial):
            if not trial:
                return None
            tm = trial.get('test_metric') if isinstance(trial, dict) else None
            if not isinstance(tm, dict) or not tm:
                return None
            if task_type == 'binary_classification':
                return tm.get('roc_auc')
            if task_type == 'multiclass_classification':
                return tm.get('accuracy')
            return tm.get('mae')

        voted_score = _score_from_trial(voted_trial)
        validation_score = _score_from_trial(validation_trial)

        if voted_score is not None:
            voted_scores.append(float(voted_score))
        if validation_score is not None:
            validation_scores.append(float(validation_score))
        best_test_metric = selection_bundle.get('best_test_metric') if isinstance(selection_bundle, dict) else None
        if best_test_metric is None:
            best_test_trial = selection_bundle.get('best_test_trial') if isinstance(selection_bundle, dict) else None
            best_test_metric = _score_from_trial(best_test_trial)
        best_test_score = best_test_metric
        if best_test_score is not None:
            best_test_scores.append(float(best_test_score))

    if not voted_scores:
        return None, None, [], {
            'validation_scores': validation_scores,
            'best_test_scores': best_test_scores,
            'selections': selection_details,
        }

    voted_arr = np.array(voted_scores, dtype=float)
    return (
        float(voted_arr.mean()),
        float(voted_arr.std(ddof=0)),
        voted_scores,
        {
            'validation_scores': validation_scores,
            'best_test_scores': best_test_scores,
            'selections': selection_details,
        },
    )

def aggregate_seed_runs_llm(seed_runs, task_type='binary_classification', top_k=10,
                            llm_backend='anthropic', llm_model='claude-sonnet-4-5-20250929'):
    """Aggregate seed runs using LLM-based post-selection.

    Same return signature as aggregate_seed_runs_landscape:
        (voted_mean, voted_std, voted_scores, extra_dict)
    """
    from utils.llm_selector import choose_best_trial_by_llm

    voted_scores = []
    validation_scores = []
    best_test_scores = []
    selection_details = []

    for r in seed_runs:
        results = r.get('all_results', [])
        if not results:
            continue
        selection_bundle = choose_best_trial_by_llm(
            results, task_type=task_type, top_k=top_k,
            llm_backend=llm_backend, llm_model=llm_model,
        )
        if selection_bundle is None:
            continue
        selection_details.append(selection_bundle)

        voted_trial = selection_bundle.get('voted_top') if isinstance(selection_bundle, dict) else selection_bundle
        validation_trial = selection_bundle.get('validation_top') if isinstance(selection_bundle, dict) else selection_bundle
        if validation_trial is None:
            validation_trial = voted_trial

        def _score_from_trial(trial):
            if not trial:
                return None
            tm = trial.get('test_metric') if isinstance(trial, dict) else None
            if not isinstance(tm, dict) or not tm:
                return None
            if task_type == 'binary_classification':
                return tm.get('roc_auc')
            if task_type == 'multiclass_classification':
                return tm.get('accuracy')
            return tm.get('mae')

        voted_score = _score_from_trial(voted_trial)
        validation_score = _score_from_trial(validation_trial)

        if voted_score is not None:
            voted_scores.append(float(voted_score))
        if validation_score is not None:
            validation_scores.append(float(validation_score))
        best_test_metric = selection_bundle.get('best_test_metric') if isinstance(selection_bundle, dict) else None
        if best_test_metric is None:
            best_test_trial = selection_bundle.get('best_test_trial') if isinstance(selection_bundle, dict) else None
            best_test_metric = _score_from_trial(best_test_trial)
        if best_test_metric is not None:
            best_test_scores.append(float(best_test_metric))

    if not voted_scores:
        return None, None, [], {
            'validation_scores': validation_scores,
            'best_test_scores': best_test_scores,
            'selections': selection_details,
        }

    voted_arr = np.array(voted_scores, dtype=float)
    return (
        float(voted_arr.mean()),
        float(voted_arr.std(ddof=0)),
        voted_scores,
        {
            'validation_scores': validation_scores,
            'best_test_scores': best_test_scores,
            'selections': selection_details,
        },
    )

def print_trial_details(seed_runs, database_name, task_name, algorithm, budget):
    """
    Print detailed information about all trials for debugging purposes.
    """
    print(f"\n=== DEBUG: All Trials for {database_name}::{task_name} - {algorithm} ({budget} trials) ===")
    
    for seed_idx, seed_run in enumerate(seed_runs):
        seed = seed_run.get('seed')
        trials = seed_run.get('trials')
        best_val = seed_run.get('best_val')
        best_test = seed_run.get('best_test')

        print(f"\n--- Seed {seed} (Run {seed_idx + 1}) ---")
        print(f"Best Validation: {best_val}")
        print(f"Best Test: {best_test}")
        if trials is None:
            print("Total Trials: cached")
            trial_entries = None
        elif isinstance(trials, list):
            trial_entries = trials
            print(f"Total Trials: {len(trial_entries)}")
        else:
            trial_entries = getattr(trials, 'trials', [])
            print(f"Total Trials: {len(trial_entries)}")

        # Print all trials with their metrics
        if trial_entries is None:
            cached_results = seed_run.get('all_results', [])
            for trial_idx, trial in enumerate(cached_results):
                val_metric = trial.get('val_metric', 'N/A')
                test_metric = trial.get('test_metric', 'N/A')
                loss = trial.get('loss', 'N/A')
                skipped = trial.get('skipped', False)
                print(f"  Trial {trial_idx + 1}: val={val_metric}, test={test_metric}, loss={loss}, skipped={skipped}")
            continue
        for trial_idx, trial in enumerate(trial_entries):
            if isinstance(trial, dict):
                result = trial.get('result', {})
                params = trial.get('config') or trial.get('misc', {}).get('vals')
                loss = trial.get('loss', 'N/A')
                skipped = trial.get('skipped', False)
            else:
                result = getattr(trial, 'result', {})
                params = getattr(getattr(trial, 'misc', None), 'get', lambda x, y=None: {})('vals', {})
            loss = result.get('loss', 'N/A')
            skipped = result.get('skipped', False)
            val_metric = result.get('val_metric', 'N/A')
            test_metric = result.get('test_metric', 'N/A')
            
            print(f"  Trial {trial_idx + 1}: val={val_metric}, test={test_metric}, loss={loss}, skipped={skipped}")
            
            # Print hyperparameters for first few trials or best trial
            if trial_idx < 3 or ('loss' in trial and trial.get('loss') == seed_run.get('best', {}).get('loss')):
                print(f"    Params: {params}")
    
    print("=" * 80)

def aggregate_seed_runs(seed_runs, task_type='binary_classification'):
    """
    From a list of per-seed outputs (as returned by optimize_for_task), compute
    mean/std over the primary *test* metric.
    For each seed, find the trial with best validation metric and return its test metric.
    Returns: (mean, std, per_seed_scores)
    """
    scores = []
    for r in seed_runs:
        all_results = r.get('all_results', [])
        if not all_results:
            continue

        ## filter all skipped trials
        all_results = [r for r in all_results if not r.get('skipped', False) and not r.get('val_metric', {}) == {}]
            
        # Find trial with best validation metric
        # For classification: higher is better, for regression: lower is better
        is_classification = 'classification' in task_type.lower()
        
        best_val_trial = None
        best_val_score = None
        
        for trial_result in all_results:
            val_metric = trial_result.get('val_metric')
            if val_metric is None:
                continue
            
            if isinstance(val_metric, dict):
                if task_type == 'binary_classification':
                    wanted_metrics = ['roc_auc']
                elif task_type == 'multiclass_classification':
                    wanted_metrics = ['accuracy']
                else:
                    wanted_metrics = ['mae']
                val_metric = val_metric.get(wanted_metrics[0], None)
                
            if best_val_trial is None:
                best_val_trial = trial_result
                best_val_score = val_metric
            else:
                if is_classification:
                    # For classification: higher validation score is better
                    if val_metric > best_val_score:
                        best_val_trial = trial_result
                        best_val_score = val_metric
                else:
                    # For regression: lower validation score is better
                    if val_metric < best_val_score:
                        best_val_trial = trial_result
                        best_val_score = val_metric
        
        # Get the test metric from the best validation trial
        if best_val_trial is not None:
            test_metric = best_val_trial.get('test_metric')
            if isinstance(test_metric, dict):
                if task_type == 'binary_classification':
                    wanted_metrics = ['roc_auc']
                elif task_type == 'multiclass_classification':
                    wanted_metrics = ['accuracy']
                else:
                    wanted_metrics = ['mae']
                test_metric = test_metric.get(wanted_metrics[0], None)
            if test_metric is not None:
                scores.append(test_metric)
    
    if not scores:
        return None, None, []
    arr = np.array(scores, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), scores