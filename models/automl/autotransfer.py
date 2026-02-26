import torch 
import numpy as np
from loguru import logger
from tqdm import tqdm
from typing import Dict, Union
from data.mtaskdataset import MultiTabularDataset
from models import get_auto_model
from relbench.base import EntityTask, RecommendationTask
from collections import defaultdict
from swap.execution import pretrain_model, generate_representation, get_loss_fn, METRIC_WEIGHTS
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Union, Optional, List, Callable
from dataclasses import dataclass
from collections import OrderedDict
from utils.type import seed_everything
from utils.misc import load_yaml, timer as time_it
from copy import deepcopy
import math
from dbinfer.solutions.rel_tabnn_solutions import create_ft_transformer_model, convert_all_splits
from dbinfer_bench.pyg_rdb_dataset import DBBRDBDataset
from swap.execution import wrap_searched_params_into_tabnn_config

DEFAULT_ANCHOR_CONFIG = [
    {
        'model_config_path': 'configs/model/anchors/model0.yaml',
        'repeats': 1
    }
]

TWELVE_ANCHOR_CONFIG = [
    {
        'model_config_path': f'configs/model/anchors/model{i+1}.yaml',
        'repeats': 1
    }
    for i in range(12)
]

FOUR_ANCHOR_CONFIG = [
    {
        'model_config_path': f'configs/model/anchors/model{i * 3 + 1}.yaml',
        'repeats': 1
    }
    for i in range(4)
]

SIX_ANCHOR_CONFIG = [
    {
        'model_config_path': f'configs/model/anchors/model{i * 2 + 1}.yaml',
        'repeats': 2
    }
    for i in range(6)
]

DFS_ANCHOR_CONFIG = [
    {
        'model_config_path': f'configs/model/dfsanchors/model{i}.yaml',
        'repeats': 2
    }
    for i in range(3)
]



ANCHOR_SEEDS = [42 + i for i in range(100)]
# ANCHOR_SEEDS = [42]

def configure_rdl(model_init: dict, database: MultiTabularDataset, model_config: dict, task_info: Union[EntityTask, RecommendationTask]):
    ## configure data
    node_to_col_names_dict = database.get_node_to_col_names_dict()
    model_init['node_to_col_names_dict'] = node_to_col_names_dict
    time_types = database.get_time_types()
    model_init['time_types'] = time_types
    num_nodes_dict = database.get_num_nodes_dict()
    model_init['num_nodes_dict'] = num_nodes_dict
    if str(task_info.task_type) == "multiclass_classification":
        model_init['out_channels'] = task_info.num_classes
    else:
        model_init['out_channels'] = task_info.out_channels
    model_init['model_config'] = model_config
    model_init['strategy'] = model_config['gnn_config']['pre_sf']
    node_types, edge_types = database.get_node_edge_types()
    model_init['node_types'] = node_types
    model_init['edge_types'] = edge_types
    model_init['data_metadata'] = database.get_data_metadata()
    return model_init

class StopOnPlateau(object):
    def __init__(self, mode='min', patience=10, cooldown=0, threshold=1e-4, threshold_mode='rel'):
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        
        self._reset()
    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            return 1
        else:
            return 0

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = float('-inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

@dataclass
class _CachedItem:
    z: torch.Tensor           # cached head input (on CPU)
    y: Optional[torch.Tensor] # target (on CPU) if available
    meta: dict                # any extra needed by head (rare)

class _CachedHeadDataset(Dataset):
    def __init__(self, items: List[_CachedItem]):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        return it.z, it.y, it.meta


class TaskEmb:
    def __init__(self, root_dir, task_id, model_type: str = 'rdl'):
        self.root = root_dir
        self.task_id = task_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._feature_cache = []
        self._val_feature_cache = []
        self.model_type = model_type
        self.model = None
        self.task_meta = {}
    
    def configure_dfs_model_and_loaders(self, dfs_dataset: DBBRDBDataset, model_config: Dict):
        from dbinfer.solutions.rel_tabnn_solutions import check_task_type
        from torch_frame.data import DataLoader  
        task = dfs_dataset.tasks[0]
        task_type, num_classes = check_task_type(dfs_dataset.tasks[0])      
        self.task_type = task_type
        self.num_classes = num_classes
        if (task_type == 'classification' or task_type == 'binary_classification') and num_classes > 2:
            task_type = 'multiclass_classification'
        elif task_type == 'binary_classification' and num_classes == 2:
            task_type = 'binary_classification'
        elif task_type == 'classification' and num_classes == 2:
            task_type = 'binary_classification'
        else:
            task_type = 'regression'
        tabnn_model_config = wrap_searched_params_into_tabnn_config(model_config)
        val_sample_batch = 2000
        remove_add_feature = True 
        remove_cat_temporal_information = True
        train_ds, val_ds, test_ds = convert_all_splits(
            dbb_dataset=dfs_dataset,
            task_type=task_type,
            task_name=task.metadata.name,
            transform="none",
            cache_dir="./datasets",
            remove_add_feature=remove_add_feature,
            remove_cat_temporal_information=remove_cat_temporal_information, 
            sample_size = None,
            scaler_path=dfs_dataset.scaler_path,
            val_sample_batch=val_sample_batch
        )
        model = create_ft_transformer_model(train_ds, tabnn_model_config, task_type, num_classes)
        train_loader = DataLoader(train_ds.tensor_frame, batch_size=tabnn_model_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds.tensor_frame, batch_size=tabnn_model_config['batch_size'])
        test_loader = DataLoader(test_ds.tensor_frame, batch_size=tabnn_model_config['eval_batch_size'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=tabnn_model_config['lr'], weight_decay=tabnn_model_config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=tabnn_model_config['gamma'])

        loss_fn = {
            'binary_classification': torch.nn.BCEWithLogitsLoss(),
            # 'bpr': torch.nn.BCEWithLogitsLoss(),
            # 'mse': torch.nn.MSELoss(),
            'regression': torch.nn.L1Loss(),
            'multiclass_classification': torch.nn.CrossEntropyLoss(),
        }[task_type]

        return model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn, task_type, num_classes, tabnn_model_config
    
    @torch.no_grad()
    def cache_features_dfs(self,
                       model,
                       train_loader,
                       model_config,
                       task_type,
                       val_loader = None,
                       max_batches: Optional[int] = None,
                       clear_existing: bool = False,
                       eval_mode: bool = True,
                       debug_mode: bool = True) -> int:
        """
        Run one pass, intercept the **input to the head** with a forward_pre_hook,
        and cache (z, y) on CPU.
        If debug_mode is True, also cache the validation set so that later we can use the validation set to debug the model in fit_head_from_cache_start_end
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if debug_mode:
            assert val_loader is not None, "Validation loader is required when debug_mode is True"
        if clear_existing:
            self._feature_cache = []
            self._val_feature_cache = []
        if self._feature_cache:
            return 0
        model.to(device)
        head, head_name = self._get_head_module(model)

        def _hook(_mod, inputs):
            z = inputs[0]
            if isinstance(z, (tuple, list)): z = z[0]
            # Ensure cached features are float32 to avoid dtype mismatches
            self._last_hook_z = z.detach().cpu().float()

        handle = head.register_forward_pre_hook(_hook)

        prev_mode = model.training
        if eval_mode: model.eval()
        logger.info(f"Caching features at head input via `{head_name}` on train loader...")

        seen = 0
        for i, tf in tqdm(enumerate(train_loader), desc=f'Caching'):
            if i >= model_config['max_steps_per_epoch']:
                break
            tf = tf.to(device)
            pred = model(tf)
            z = self._last_hook_z
            if z is None:
                handle.remove()
                if eval_mode and prev_mode: model.train()
                raise RuntimeError("Head pre-hook did not capture any features; "
                                   "ensure the head is actually called during forward.")
            y = tf.y.view(-1).float()
            # Ensure y is float32 to match model parameters
            y = y.float()
            meta = {}  # put anything your head needs besides z here
            self._feature_cache.append(_CachedItem(z=z, y=y, meta=meta))
            seen += 1
        if debug_mode:
            for i, tf in enumerate(tqdm(val_loader, desc="Caching")):
                if max_batches is not None and i >= max_batches:
                    break
                self._last_hook_z = None
                model.to(self.device)
                tf = tf.to(self.device)
                model(tf)
                z = self._last_hook_z
                if z is None:
                    handle.remove()
                    if eval_mode and prev_mode: model.train()
                    raise RuntimeError("Head pre-hook did not capture any features; "
                                       "ensure the head is actually called during forward.")
                y = tf.y.view(-1).float()
                meta = {}  # put anything your head needs besides z here
                self._val_feature_cache.append(_CachedItem(z=z, y=y, meta=meta))
                seen += 1

        handle.remove()
        if eval_mode and prev_mode: model.train()
        logger.info(f"Cached {seen} batches (CPU).")
        return seen

    def configure_model_and_loaders(self, model_config: Dict, database: MultiTabularDataset, task_idx: int = 0):
        model_params_config = model_config['model_params']
        model = get_auto_model(model_params_config["model_type"])
        model_init = {**model_params_config}
        model_init['col_stats_dict'] = database.get_col_stats_dict()
        task_info = database.tasks[task_idx]
        self.database = database
        ## convert task type to string for easier manipulation
        if not isinstance(task_info.task_type, str):
            task_info.task_type = task_info.task_type.value
        if task_info.task_type == "recommendation" or task_info.task_type == "link_prediction":
            model_init['src_table'] = task_info.src_entity_table
            model_init['dst_table'] = task_info.dst_entity_table
        else:
            model_init['src_table'] = task_info.entity_table
            model_init['dst_table'] = task_info.entity_table
        if model_params_config["model_type"] == "rdl":
            model_init = configure_rdl(model_init, database, model_params_config, task_info)
        self.model_init = model_init
        self.model = model(**model_init)
        loader_dict, num_dst_nodes_dict, task_meta, meta_graph, sparse_tensor, sampled_val_table = database.initialize_data(task_idx, 0, 1, model_params_config['sampler_config']['loader_type'], model_config)
        self.loader_dict = loader_dict
        self.num_dst_nodes_dict = num_dst_nodes_dict
        self.task_meta = task_meta
        self.num_classes = task_meta['out_channels']
        self.meta_graph = meta_graph
        self.sparse_tensor = sparse_tensor
        self.sampled_val_table = sampled_val_table
        self.train_configs = {
            "max_steps_per_epoch": model_config["max_steps_per_epoch"],
            "epochs": model_config["epochs"],
            "share_same_time": True,
            "loss_fn": model_params_config['gnn_config']['loss_fn']
        }

        self.model_config = model_config

    # NEW: utilities -----------------------------------------------------------
    def _get_head_module(self, model: torch.nn.Module):
        # try common names
        for name in ["head", 'decoder']:
            if hasattr(model, name):
                return getattr(model, name), name
        raise AttributeError(
            "Could not find a head module on the model. "
            "Expected one of ['head','decoder']."
        )

    def _extract_targets(self, batch) -> Optional[torch.Tensor]:
        # best-effort: extend this if your batch carries targets differently
        for key in ("y", "target", "labels"):
            if hasattr(batch, key):
                t = getattr(batch, key)
                if torch.is_tensor(t): return t.detach().cpu()
        # mapping-style
        try:
            for key in ("y", "target", "labels"):
                if key in batch:
                    t = batch[key]
                    if torch.is_tensor(t): return t.detach().cpu()
        except Exception:
            pass
        # as a fallback, no targets (e.g., fully unsupervised)
        return None

    def _forward_head_only(self, model, z: torch.Tensor, meta: dict) -> torch.Tensor:
        head, _ = self._get_head_module(model)
        # try common call signatures
        head.to(self.device)
        # Ensure z is float32 to match model parameters
        z = z.to(self.device, dtype=torch.float32)
        try:
            return head(z)
        except TypeError:
            # sometimes head(z, meta)
            try:
                return head(z, meta)
            except TypeError as e:
                raise TypeError(
                    "Head call signature not recognized. "
                    "Override `_forward_head_only` to adapt."
                ) from e

    # NEW: cache head inputs ---------------------------------------------------
    @torch.no_grad()
    def cache_features(self,
                       model,
                       model_init,
                       max_batches: Optional[int] = None,
                       clear_existing: bool = False,
                       eval_mode: bool = True,
                       debug_mode: bool = True) -> int:
        """
        Run one pass, intercept the **input to the head** with a forward_pre_hook,
        and cache (z, y) on CPU.
        If debug_mode is True, also cache the validation set so that later we can use the validation set to debug the model in fit_head_from_cache_start_end
        """
        if clear_existing:
            self._feature_cache = []
            self._val_feature_cache = []
        if self._feature_cache:
            return 0
        data_loader = self.loader_dict['train']
        if debug_mode:
            debug_data_loader = self.loader_dict['val']
        head, head_name = self._get_head_module(model)

        def _hook(_mod, inputs):
            z = inputs[0]
            if isinstance(z, (tuple, list)): z = z[0]
            # Ensure cached features are float32 to avoid dtype mismatches
            self._last_hook_z = z.detach().cpu().float()

        handle = head.register_forward_pre_hook(_hook)

        prev_mode = model.training
        if eval_mode: model.eval()
        logger.info(f"Caching features at head input via `{head_name}` on train loader...")

        seen = 0
        for i, batch in enumerate(tqdm(data_loader, desc="Caching")):
            if max_batches is not None and i >= max_batches:
                break
            self._last_hook_z = None
            model.to(self.device)
            batch = batch.to(self.device) if hasattr(batch, "to") else batch
            # Trigger full forward; hook grabs the head-input.
            seed_time = batch[model_init['src_table']].seed_time
            _ = generate_representation(
                model, self.device, batch,
                model_init['src_table'], model_init['dst_table'], None
            )
            z = self._last_hook_z[:seed_time.size(0)]
            if z is None:
                handle.remove()
                if eval_mode and prev_mode: model.train()
                raise RuntimeError("Head pre-hook did not capture any features; "
                                   "ensure the head is actually called during forward.")
            y = batch[model_init['src_table']].y[:seed_time.size(0)]
            # Ensure y is float32 to match model parameters
            y = y.float()
            meta = {}  # put anything your head needs besides z here
            self._feature_cache.append(_CachedItem(z=z, y=y, meta=meta))
            seen += 1
        if debug_mode:
            for i, batch in enumerate(tqdm(debug_data_loader, desc="Caching")):
                if max_batches is not None and i >= max_batches:
                    break
                self._last_hook_z = None
                model.to(self.device)
                batch = batch.to(self.device) if hasattr(batch, "to") else batch
                seed_time = batch[model_init['src_table']].seed_time
                _ = generate_representation(
                    model, self.device, batch,
                    model_init['src_table'], model_init['dst_table'], None
                )
                z = self._last_hook_z[:seed_time.size(0)]
                if z is None:
                    handle.remove()
                    if eval_mode and prev_mode: model.train()
                    raise RuntimeError("Head pre-hook did not capture any features; "
                                       "ensure the head is actually called during forward.")
                y = batch[model_init['src_table']].y[:seed_time.size(0)]
                # Ensure y is float32 to match model parameters
                y = y.float()
                meta = {}  # put anything your head needs besides z here
                self._val_feature_cache.append(_CachedItem(z=z, y=y, meta=meta))
                seen += 1

        handle.remove()
        if eval_mode and prev_mode: model.train()
        logger.info(f"Cached {seen} batches (CPU).")
        return seen

    def clear_cache(self):
        self._feature_cache = []
        self._val_feature_cache = []
    # NEW: train only the head from cached features ---------------------------
    def fit_head_from_cache(self,
                            optimizer: str = 'adam',
                            learning_rate: float = 1e-2,
                            weight_decay: float = 1e-4,
                            epochs: int = 50,
                            batch_size: Optional[int] = None,
                            shuffle: bool = True,
                            loss_fn: Optional[str] = None,
                            model=None):
        """
        Train head using cached (z, y). No backbone recompute.
        If your head requires extra meta beyond z, stash it in the cache.
        """
        if not self._feature_cache:
            raise ValueError("Feature cache is empty. Call `cache_features()` first.")
        
        if model is not None:
            self.model = model

        head, head_name = self._get_head_module(self.model)
        params = list(head.parameters())
        if len(params) == 0:
            raise ValueError("Head has no parameters to optimize.")

        if optimizer == 'adam':
            opt = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            opt = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer {optimizer}')

        gamma = 1.0
        lr_scheduler = ExponentialLR(opt, gamma=gamma)

        # dataset/dataloader from cache
        ds = _CachedHeadDataset(self._feature_cache)
        if self._val_feature_cache:
            ds_val = _CachedHeadDataset(self._val_feature_cache)
            dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
        batch_size = 1
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        # choose loss
        loss_name = loss_fn or self.train_configs["loss_fn"]
        criterion = get_loss_fn(loss_name)
        logger.info(f"Fitting head from cache (loss={loss_name})...")
        counter = StopOnPlateau()
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Head-only (cached)"):
            epoch_loss = 0.0
            n_batches = 0
            for z, y, meta in dl:
                z = z.view(-1, z.size(-1))
                y = y.view(-1)
                z = z.to(self.device)
                opt.zero_grad()
                logits = self._forward_head_only(self.model, z, meta)
                assert logits.requires_grad, "Logits have no grad; head likely frozen or in no_grad."
                # y can be None (e.g., purely unsupervised). Skip in that case.
                if y is None:
                    continue
                y = y.to(logits.device, dtype=logits.dtype)
                loss = criterion(logits, y.view(*logits.shape))
                assert loss.requires_grad, "Loss has no grad; check loss fn."
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            lr_scheduler.step()

            if n_batches == 0:
                logger.warning("No batches with targets found in cache; "
                               "training did not perform any updates.")
                break

            if counter.step(epoch_loss):
                logger.info(f"[epoch {epoch}]: early stop, loss={epoch_loss:.4f}")
                break
            if epoch % max(1, epochs // 4) == 0:
                logger.info(f"[epoch {epoch}]: loss={epoch_loss:.4f}") 
        if self._val_feature_cache:
            ## do evaluation on the validation set
            ## don't calculate loss, calculate metric 
            pred_list = []
            for z, y, meta in dl_val:
                z = z.view(-1, z.size(-1))
                y = y.view(-1)
                logits = self._forward_head_only(self.model, z, meta)
                pred = logits.view(-1) if logits.size(1) == 1 else logits
                if self.num_classes > 2:
                    pred = torch.softmax(pred, dim=1)
                elif self.task_meta.get('clamp_min', None) is not None:
                    pred = torch.clamp(pred, self.task_meta['clamp_min'], self.task_meta['clamp_max'])
                else:
                    pred = torch.sigmoid(pred)
                pred = pred.view(-1) if (pred.ndim > 1 and pred.size(1) == 1) else pred
                pred_list.append(pred.detach().cpu())
            pred_list = torch.cat(pred_list, dim=0).numpy()
            metric = self.database.tasks[self.task_id].evaluate(pred_list, self.sampled_val_table)
            logger.info(metric)
        return self.model.state_dict()
    
    def fit_head_from_cache_start_end(self,
                            model,
                            optimizer: str = 'adam',
                            learning_rate: float = 1e-2,
                            weight_decay: float = 1e-4,
                            epochs: int = 50,
                            batch_size: Optional[int] = None,
                            shuffle: bool = True,
                            loss_fn: Optional[str] = None,
                            debug_mode: bool = True):
        """
        Train head using cached (z, y). No backbone recompute.
        If your head requires extra meta beyond z, stash it in the cache.
        Returns first epoch parameters and final parameters for loss barrier computation.
        """
        if not self._feature_cache:
            raise ValueError("Feature cache is empty. Call `cache_features()` first.")

        head, head_name = self._get_head_module(model)
        first_epoch_parameters = None

        params = list(head.parameters())
        if len(params) == 0:
            raise ValueError("Head has no parameters to optimize.")

        if optimizer == 'adam':
            opt = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            opt = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer {optimizer}')

        gamma = self.model_config['model_params']["optimizer_config"]["gamma"]
        lr_scheduler = ExponentialLR(opt, gamma=gamma)

        # dataset/dataloader from cache
        ds = _CachedHeadDataset(self._feature_cache)
        if self._val_feature_cache:
            ds_val = _CachedHeadDataset(self._val_feature_cache)
            dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
        batch_size = 1
        # if batch_size is None:
        #     # Heuristic: use same batch size as your train loader if accessible
        #     try:
        #         batch_size = self.loader_dict['train'].batch_size or 1
        #     except Exception:
        #         batch_size = 1
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        if self._val_feature_cache:
            ds_val = _CachedHeadDataset(self._val_feature_cache)
            dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)

        # choose loss
        loss_name = loss_fn or self.train_configs["loss_fn"]
        criterion = get_loss_fn(loss_name)

        logger.info(f"Fitting head from cache (loss={loss_name})...")
        counter = StopOnPlateau()
        model.train()
        for epoch in tqdm(range(epochs), desc="Head-only (cached)"):
            epoch_loss = 0.0
            n_batches = 0
            if epoch == 1:
                first_epoch_parameters = model.state_dict()
            for z, y, meta in dl:
                z = z.view(-1, z.size(-1))
                y = y.view(-1)
                z = z.to(self.device)
                opt.zero_grad()
                logits = self._forward_head_only(model, z, meta)
                assert logits.requires_grad, "Logits have no grad; head likely frozen or in no_grad."
                # y can be None (e.g., purely unsupervised). Skip in that case.
                if y is None:
                    continue
                y = y.to(logits.device, dtype=logits.dtype)
                loss = criterion(logits, y.view(*logits.shape))
                assert loss.requires_grad, "Loss has no grad; check loss fn."
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            lr_scheduler.step()

            if n_batches == 0:
                logger.warning("No batches with targets found in cache; "
                               "training did not perform any updates.")
                break

            if counter.step(epoch_loss):
                logger.info(f"[epoch {epoch}]: early stop, loss={epoch_loss:.4f}")
                break
            if epoch % max(1, epochs // 4) == 0:
                logger.info(f"[epoch {epoch}]: loss={epoch_loss:.4f}") 
        
        if self._val_feature_cache:
            ## do evaluation on the validation set
            ## don't calculate loss, calculate metric 
            pred_list = []
            for z, y, meta in dl_val:
                z = z.view(-1, z.size(-1))
                y = y.view(-1)
                logits = self._forward_head_only(model, z, meta)
                pred = logits.view(-1) if logits.size(1) == 1 else logits
                if self.task_meta['out_channels'] > 1:
                    pred = torch.softmax(pred, dim=1)
                elif self.task_meta.get('clamp_min', None) is not None:
                    pred = torch.clamp(pred, self.task_meta['clamp_min'], self.task_meta['clamp_max'])
                else:
                    pred = torch.sigmoid(pred)
                pred = pred.view(-1) if (pred.ndim > 1 and pred.size(1) == 1) else pred
                pred_list.append(pred.detach().cpu())
            pred_list = torch.cat(pred_list, dim=0)
            metric = self.database.tasks[self.task_id].evaluate(pred_list, self.sampled_val_table)
            logger.info(metric)
        return first_epoch_parameters, model.state_dict()
        
    def fit_full(self, model_config, loss_fn: str = None):
        task = self.database.tasks[self.task_id]
        task_meta = self.task_meta
        if str(task.task_type) == "multiclass_classification":
            task_meta['loss_fn'] = "ce"
        else:
            task_meta['loss_fn'] = loss_fn
        model, _ = self.return_a_newly_randomized_model(self.database, model_config, ANCHOR_SEEDS[0])
        optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=model_config["optimizer_config"]["lr"],
        weight_decay=float(model_config["optimizer_config"]["weight_decay"])
    )
        first_epoch_parameters = None
    ## when gamma is 1, the scheduler is disabled
        lr_scheduler = ExponentialLR(optimizer, gamma=model_config["optimizer_config"]["gamma"])
        model.train()  # Use the new model, not self.model
        train_configs = {
            "max_steps_per_epoch": model_config["max_steps_per_epoch"],
            "epochs": model_config["epochs"],
            "share_same_time": model_config.get("share_same_time", True),
            "loss_fn": task_meta['loss_fn']
        }

        

        best_val_metric = 0
        early_stop_counter = 0
        best_val_metric_record = {}
        
        for epoch in range(model_config["epochs"]):
            average_val_metric = defaultdict(list)
            if epoch == 1:
                first_epoch_parameters = model.state_dict()
            train_loss, val_metric = pretrain_model(
                model,  # Use the new model, not self.model
                self.database, 
                self.database.tasks[self.task_id],
                self.num_dst_nodes_dict,
                self.sampled_val_table, 
                self.task_meta, 
                None, 
                self.sparse_tensor,
                self.loader_dict['train'],
                self.loader_dict['val'],
                train_configs,
                optimizer,
                lr_scheduler, 
                self.device,
                loader_type='node'  # Add missing loader_type parameter
            )

            logger.debug(f"Train loss: {train_loss}, Val metric: {val_metric}")
            for key, value in val_metric.items():
                average_val_metric[key].append(value * METRIC_WEIGHTS.get(key, 1))

            average_val_metric = {key: sum(value) / len(value) for key, value in average_val_metric.items()}

            if np.average(list(average_val_metric.values())) > best_val_metric or best_val_metric == 0:
                early_stop_counter = 0
                best_val_metric = np.average(list(average_val_metric.values()))
                best_val_metric_record = {**val_metric}
                # Save the best model using deepcopy and move to CPU to save memory
                best_model = deepcopy(model).cpu()
                # torch.save(model.state_dict(), model_weight_name)
                logger.debug(f"Early stop counter: {early_stop_counter}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= model_config["early_stop"]:
                    logger.debug(f"Early stop, no improvement for {early_stop_counter} epochs")
                    break
        
        return first_epoch_parameters, best_model.state_dict()
        
            
    def montecarlo_fisher_dfs(self, val_loader, loss_fn_str: str, real_loss):
        for p in self.model.parameters():
            p.grad2_acc = torch.zeros_like(p.data)
            p.grad_counter = 0
        self.model.train()
        self.model.to(self.device)
        for tf in tqdm(val_loader, desc="Computing Monte Carlo Fisher Information"):
            tf.to(self.device)
            target_output = self.model(tf)
            target_y = tf.y.view(-1).float()
            if loss_fn_str == 'cross_entropy':
                # multiclass
                    if target_output.ndim > 1 and target_output.shape[-1] > 1:
                        target = torch.multinomial(F.softmax(target_output, dim=-1), 1).detach().view(-1)
                # binary
                    else:
                        target = torch.bernoulli(F.sigmoid(target_output.view(-1))).detach()
            elif loss_fn_str == 'mse' or loss_fn_str == 'mae':
                target = target_y
            else:
                raise ValueError(f'Unsupported loss function {loss_fn_str}')
                
            loss_fn = real_loss
            loss = loss_fn(target_output, target.view(*target_output.shape))
            self.model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad2_acc += (p.grad.data ** 2).to(p.grad2_acc.device)
                        p.grad_counter += 1
    
    def extract_embedding_dfs(self):
        grad2s = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'grad2_acc'):
                if 'backbone' not in name:
                    continue
                grad2 = module.weight.grad2_acc.cpu().detach().numpy()
                grad2s.append(grad2.flatten())
        grad2s = np.hstack(grad2s)
        moment1 = np.mean(grad2s)
        moment2 = np.mean(grad2s ** 2)
        return moment2 / (moment1 ** 2)
        
        
    def montecarlo_fisher(self, loss_fn_str: str):
        for p in self.model.parameters():
            p.grad2_acc = torch.zeros_like(p.data)
            p.grad_counter = 0
        self.model.train()
        self.model.to(self.device)
        for batch in tqdm(self.loader_dict['val'], desc="Computing Monte Carlo Fisher Information"):
            batch.to(self.device)
            try:
                target_output = generate_representation(self.model, self.device, batch, 
                                                        self.model_init['src_table'], self.model_init['dst_table'], None)
                seed_time = batch[self.model_init['src_table']].seed_time
                target_output = target_output[:seed_time.size(0)]
                target_y = batch[self.model_init['src_table']].y[:seed_time.size(0)]
                if loss_fn_str == 'cross_entropy':
                    # multiclass
                        if target_output.ndim > 1 and target_output.shape[-1] > 1:
                            target = torch.multinomial(F.softmax(target_output, dim=-1), 1).detach().view(-1)
                    # binary
                        else:
                            target = torch.bernoulli(F.sigmoid(target_output.view(-1))).detach()
                elif loss_fn_str == 'mse' or loss_fn_str == 'mae':
                    target = target_y
                else:
                    raise ValueError(f'Unsupported loss function {loss_fn_str}')
                if isinstance(self.task_meta['loss_fn'], str):
                    loss_fn = get_loss_fn(self.task_meta['loss_fn'])
                else:
                    loss_fn = self.task_meta['loss_fn']
                loss = loss_fn(target_output, target.view(*target_output.shape))
                self.model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad2_acc += (p.grad.data ** 2).to(p.grad2_acc.device)
                            p.grad_counter += 1
            except ValueError:
                continue
            
    def extract_embedding(self):
        grad2s = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'grad2_acc'):
                if 'gnn' not in name:
                    continue
                grad2 = module.weight.grad2_acc.cpu().detach().numpy()
                grad2s.append(grad2.flatten())
        grad2s = np.hstack(grad2s)
        moment1 = np.mean(grad2s)
        moment2 = np.mean(grad2s ** 2)
        return moment2 / (moment1 ** 2)
    
    def generate_multi_model_for_loss_barrier(self, model_config: Dict, database: MultiTabularDataset, seed: List[int] = None):
        if seed is None:
            seed = [42, 43]
        model = get_auto_model(model_config["model_type"])
        model_init = {**model_config}
        model_init['col_stats_dict'] = database.get_col_stats_dict()
        task_info = database.tasks[self.task_id]
        self.database = database
        ## convert task type to string for easier manipulation
        if not isinstance(task_info.task_type, str):
            task_info.task_type = task_info.task_type.value
        if task_info.task_type == "recommendation" or task_info.task_type == "link_prediction":
            model_init['src_table'] = task_info.src_entity_table
            model_init['dst_table'] = task_info.dst_entity_table
        else:
            model_init['src_table'] = task_info.entity_table
            model_init['dst_table'] = task_info.entity_table
        if model_config["model_type"] == "rdl":
            model_init = configure_rdl(model_init, database, model_config, task_info)
        model_init = model_init
        seed_everything(seed[0])
        model1 = model(**model_init)
        model1_state_dict = model1.state_dict()
        seed_everything(seed[1])
        model2 = model(**model_init)
        model2_state_dict = model2.state_dict()
        return model1_state_dict, model2_state_dict
        
    def return_a_newly_randomized_model(self, database: MultiTabularDataset, model_config: Dict, seed: int = 42):
        seed_everything(seed)
        model_class = get_auto_model(model_config["model_type"])
        model_init = {**model_config}
        model_init['col_stats_dict'] = database.get_col_stats_dict()
        task_info = database.tasks[self.task_id]
        ## convert task type to string for easier manipulation
        if not isinstance(task_info.task_type, str):
            task_info.task_type = task_info.task_type.value
        if task_info.task_type == "recommendation" or task_info.task_type == "link_prediction":
            model_init['src_table'] = task_info.src_entity_table
            model_init['dst_table'] = task_info.dst_entity_table
        else:
            model_init['src_table'] = task_info.entity_table
            model_init['dst_table'] = task_info.entity_table
        if model_config["model_type"] == "rdl":
            model_init = configure_rdl(model_init, database, model_config, task_info)
        model = model_class(**model_init)
        return model, model_init

    # ---------------- NTK helpers ----------------
    def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
        mu_r = K.mean(dim=1, keepdim=True)
        mu_c = K.mean(dim=0, keepdim=True)
        mu = K.mean()
        return K - mu_r - mu_c + mu

    def _cka(self, K: torch.Tensor, L: torch.Tensor, eps: float = 1e-12) -> float:
        Kc, Lc = self._center_gram(K), self._center_gram(L)
        num = (Kc * Lc).sum()
        den = torch.linalg.norm(Kc, ord='fro') * torch.linalg.norm(Lc, ord='fro') + eps
        return (num / den).item()

    def _hsic_biased(self, K: torch.Tensor, L: torch.Tensor) -> float:
        n = K.size(0)
        Kc, Lc = self._center_gram(K), self._center_gram(L)
        return (Kc * Lc).sum().item() / max((n - 1) ** 2, 1)

    def _label_kernel(self, y: torch.Tensor, task_type: str, num_classes: int | None) -> torch.Tensor:
        y = y.detach().cpu()
        if task_type == "classification":
            if y.ndim == 1:
                C = int(num_classes or int(y.max().item() + 1))
                Y = torch.nn.functional.one_hot(y.to(torch.long), num_classes=C).float()
            else:
                Y = y.float()
        else:
            Y = y.view(-1, 1).float()
        return (Y @ Y.T)

    def _subsample_cache(self, cache_list, subsample_n: int | None, rng: np.random.RandomState | None = None):
        """Flatten cached items to per-row (z_i, y_i) and subsample."""
        zs, ys = [], []
        for item in cache_list:
            z = item.z  # [m, d]
            y = item.y  # [m] or [m,C]
            if z is None or y is None: 
                continue
            m = z.shape[0]
            zs.append(z.view(m, -1).float().cpu())
            ys.append(y.view(m, -1).squeeze(-1).cpu())
        if not zs:
            return None, None
        Z = torch.cat(zs, dim=0)   # [N, d]
        Y = torch.cat(ys, dim=0)   # [N] or [N,C]
        N = Z.size(0)
        if (subsample_n is not None) and (N > subsample_n):
            rng = rng or np.random.RandomState(0)
            idx = torch.from_numpy(rng.choice(N, subsample_n, replace=False))
            Z, Y = Z[idx], Y[idx]
        return Z, Y

    def _per_sample_grad_head(self, head: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
        """
        One sample -> gradient vector wrt head params using scalarization 0.5*||f(z)||^2.
        Returns [P] flatten.
        """
        device = next(head.parameters()).device
        head.zero_grad(set_to_none=True)
        out = head(z.unsqueeze(0).to(device, dtype=torch.float32))  # [1, C] or [1,1]
        scalar = 0.5 * (out.pow(2).sum())
        grads = torch.autograd.grad(
            scalar, [p for p in head.parameters() if p.requires_grad],
            retain_graph=False, create_graph=False, allow_unused=True
        )
        g = []
        for (p, gp) in zip(head.parameters(), grads):
            if not p.requires_grad: 
                continue
            if gp is None:
                g.append(torch.zeros_like(p).reshape(-1))
            else:
                g.append(gp.detach().reshape(-1))
        return torch.cat(g, dim=0)  # [P]

    def ntk_head_from_cache(
        self,
        split: str = "train",                 # "train" | "val"
        subsample_n: int | None = 2048,
        compute_alignment: bool = True,
        compute_hsic: bool = False,
        spectrum_topk: int = 8,
        rng_seed: int = 0,
    ) -> dict:
        """
        Empirical NTK using head-only parameters on cached features (random backbone).
        Returns a dict with spectrum stats, diag/offdiag stats, and optional CKA/HSIC to labels.
        """
        assert split in ("train", "val")
        cache_list = self._feature_cache if split == "train" else self._val_feature_cache
        if not cache_list:
            raise ValueError(f"No cached features for split={split}. Run cache_features() first.")

        # 1) Flatten and subsample (per-row)
        Z, Y = self._subsample_cache(cache_list, subsample_n, np.random.RandomState(rng_seed))
        if Z is None or Z.numel() == 0:
            raise ValueError("Empty Z/Y after subsampling.")

        # 2) Head module & freeze mask is already in your design; only head needs grads
        head, _ = self._get_head_module(self.model)
        head = head.to(self.device)
        head.eval()  # disable dropout/bn randomness; grads still flow
        for p in self.model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

        # 3) Build gradient feature matrix G \in R^{N x P} (per-sample)
        G_rows = []
        with torch.enable_grad():
            for i in range(Z.size(0)):
                g = self._per_sample_grad_head(head, Z[i].to(self.device))
                G_rows.append(g.cpu())
        G = torch.stack(G_rows, dim=0)  # [N, P]
        N, P = G.shape

        # 4) Kernel & quick stats (avoid O(N^2) when not needed)
        # Diagonal = per-sample grad norms^2
        diag = (G * G).sum(dim=1)  # [N]
        mean_diag = diag.mean().item()
        std_diag = diag.std(unbiased=False).item()

        # Spectrum on small matrix: SVD of P x P (since N may be large)
        # eigvals of K = GG^T equal singular_values(G)^2
        # compute via covariance in parameter space:
        C = (G.T @ G) / max(N, 1)                 # [P,P]
        evals = torch.linalg.eigvalsh(C).clamp_min(0).flip(0)   # descending, non-neg
        evals_np = evals.cpu().numpy()
        s_sum = float(evals.sum().item())
        s_topk = evals_np[:min(spectrum_topk, len(evals_np))]

        # Effective rank (entropy of normalized spectrum)
        p_spec = evals / (evals.sum() + 1e-12)
        H = -(p_spec[p_spec > 0] * torch.log(p_spec[p_spec > 0])).sum().item()
        eff_rank = float(math.exp(H))

        spec_stats = {
            "ntk_head_param_dim": int(P),
            "ntk_samples": int(N),
            "ntk_trace_over_N": s_sum,                       # = mean eigenvalue of K/N (since C = G^T G / N)
            "ntk_top1_frac": float((evals[0] / (evals.sum() + 1e-12)).item()),
            "ntk_topk_sum_frac": float((evals[:min(spectrum_topk, len(evals))].sum() / (evals.sum() + 1e-12)).item()),
            "ntk_eff_rank": eff_rank,
            "ntk_diag_mean": mean_diag,
            "ntk_diag_std": std_diag,
        }

        out = {"spec": spec_stats}

        # 5) Optional: alignment to label kernel on the same subsample (O(N^2))
        if compute_alignment:
            # Construct K = G G^T (stay on CPU to save VRAM)
            K = (G @ G.T)  # [N,N]
            # Label kernel (use task meta if available)
            task_type = str(self.task_meta.get("task_type", self.database.tasks[self.task_id].task_type))
            if task_type not in ("classification", "regression"):
                # fallback to your database metadata
                task_type = "classification" if self.task_meta.get("out_channels", 1) > 1 else "regression"
            num_classes = int(self.task_meta.get("out_channels", 1))

            L = self._label_kernel(Y, task_type=task_type, num_classes=num_classes)
            out["align"] = {
                "cka_ntk_label": self._cka(K, L),
            }
            if compute_hsic:
                out["align"]["hsic_ntk_label"] = self._hsic_biased(K, L)

            # handy relative scale stats on K (gives sense of coupling strength)
            off_sum = K.sum() - K.diag().sum()
            off_mean = (off_sum / max(N * (N - 1), 1)).item()
            out["kernel_stats"] = {
                "K_mean_offdiag": off_mean,
                "K_mean_diag": mean_diag,
                "K_offdiag_to_diag_ratio": float(off_mean / (mean_diag + 1e-12)),
            }

        return out

    def compute_loss_barrier(self, barrier_strategy:str = 'random_init', model_config: Dict = None, database: MultiTabularDataset = None, max_batches: int = 100, loss_fn: str = None):
        if barrier_strategy == 'random_init':
            model1_state_dict, model2_state_dict = self.generate_multi_model_for_loss_barrier(model_config, database)
        elif barrier_strategy == 'train_2_only_head':
            #self.cache_features(max_batches=max_batches)
            self.configure_model_and_loaders(model_config, database, ANCHOR_SEEDS[0])
            self.cache_features(max_batches=max_batches, model = self.model, model_init = self.model_init, clear_existing=True)
            model1_state_dict = self.fit_head_from_cache( loss_fn = loss_fn)
            self.configure_model_and_loaders(model_config, database, ANCHOR_SEEDS[1])
            self.cache_features(max_batches=max_batches, model = self.model, model_init = self.model_init, clear_existing=True)
            model2_state_dict = self.fit_head_from_cache(loss_fn = loss_fn)
        elif barrier_strategy == 'start_end_epoch_only_head':
            model, model_init = self.return_a_newly_randomized_model(database, model_config, ANCHOR_SEEDS[0])
            self.cache_features(max_batches=max_batches, model = model, model_init = model_init, clear_existing=True)
            model1_state_dict, model2_state_dict = self.fit_head_from_cache_start_end(model, loss_fn = loss_fn)
        elif barrier_strategy == 'start_end_epoch_full':
            model1_state_dict, model2_state_dict = self.fit_full(model_config, loss_fn = loss_fn)
        else:
            raise ValueError(f'Unsupported barrier strategy {barrier_strategy}')

        result = compute_loss_barrier(
            self.model,
            model1_state_dict,
            model2_state_dict,
            self.loader_dict['val'],
            self.device,
            self.task_meta['loss_fn'],
            forward_kwargs={
                "src_table": self.model_init['src_table'],
                "dst_table": self.model_init['dst_table']
            }
        )

        return result


        


@torch.no_grad()
def compute_loss_barrier(
    model: torch.nn.Module,
    state_a: Union[torch.nn.Module, Dict[str, torch.Tensor]],
    state_b: Union[torch.nn.Module, Dict[str, torch.Tensor]],
    data_loader,
    device: torch.device,
    loss_fn: Callable,
    *,
    forward_fn: Optional[Callable[[torch.nn.Module, object, torch.device], torch.Tensor]] = None,
    forward_kwargs: Optional[Dict] = None,
    target_extractor: Optional[Callable[[object], Optional[torch.Tensor]]] = None,
    num_points: int = 10,
    include_buffers: bool = True,
    weight_by_batch_size: bool = True,
) -> Dict[str, object]:
    """
    Compute the loss barrier between two parameter sets by linear interpolation
    θ(t) = (1 - t) * θ_A + t * θ_B, t ∈ [0, 1].

    Barrier definitions:
      - absolute:  max_t L(t) - max(L(0), L(1))
      - convex:    max_t [ L(t) - ((1 - t) * L(0) + t * L(1)) ]

    Args:
        model: A model instance whose parameters will be temporarily overwritten.
        state_a: A model or a state_dict representing endpoint A.
        state_b: A model or a state_dict representing endpoint B.
        data_loader: Iterable of batches for evaluation (e.g., validation loader).
        device: torch.device to run evaluation on.
        loss_name: Name used by your framework's `get_loss_fn(loss_name)`.
        forward_fn: Function that produces logits from (model, batch, device, **forward_kwargs).
                    If None, will try to call `generate_representation(model, device, batch, ...)`.
        forward_kwargs: Extra kwargs passed to `forward_fn`. For example, if you use
                        `generate_representation`, you likely need {"src_table": ..., "dst_table": ...}.
        target_extractor: Function that returns a target tensor from a batch. If None, a default
                          heuristic will try common keys ('y', 'target', 'labels').
        num_points: Number of interpolation points including endpoints (>= 2).
        include_buffers: If True, also interpolate buffers (e.g., BatchNorm running stats).
                         If False, only interpolate parameters.
        weight_by_batch_size: If True, average the loss weighted by batch size.

    Returns:
        A dict with keys:
          {
            "ts":            1D torch.Tensor of shape [num_points] with interpolation points,
            "losses":        List[float] evaluation loss at each t,
            "loss0":         float, loss at t=0,
            "loss1":         float, loss at t=1,
            "barrier_abs":   float, absolute barrier,
            "barrier_conv":  float, convex-combination barrier,
            "t_at_abs":      float, t where absolute barrier is reached,
            "t_at_conv":     float, t where convex barrier is reached,
          }
    """
    # --- Lazy imports to match your framework (assumes they exist in your env) ---
    # If you prefer, move these to your global imports.
    from swap.execution import get_loss_fn, generate_representation  # noqa: F401

    # Default forward: use generate_representation(model, device, batch, src, dst, None)
    if forward_fn is None:
        def _default_forward_fn(_model, _batch, _device, **kw):
            return generate_representation(
                _model, _device, _batch,
                kw.get("src_table"), kw.get("dst_table"), None
            )
        forward_fn = _default_forward_fn
    if forward_kwargs is None:
        forward_kwargs = {}

    # Default target extractor: try common fields
    if target_extractor is None:
        def _default_target_extractor(batch) -> Optional[torch.Tensor]:
            # Attribute-style
            for key in ("y", "target", "labels"):
                if hasattr(batch, key):
                    t = getattr(batch, key)
                    if torch.is_tensor(t):
                        return t
            # Mapping-style
            try:
                for key in ("y", "target", "labels"):
                    if key in batch:
                        t = batch[key]
                        if torch.is_tensor(t):
                            return t
            except Exception:
                pass
            return None
        target_extractor = _default_target_extractor

    # --- Normalize inputs to state_dicts ---
    def _as_state_dict(x) -> OrderedDict:
        return x.state_dict() if hasattr(x, "state_dict") else x

    sd_a: OrderedDict = _as_state_dict(state_a)
    sd_b: OrderedDict = _as_state_dict(state_b)
    sd_ref: OrderedDict = model.state_dict()

    # --- Sanity check and key alignment ---
    keys = list(sd_ref.keys())
    for k in keys:
        if k not in sd_a or k not in sd_b:
            raise KeyError(f"Key {k} missing in one of the provided states")
        if sd_a[k].shape != sd_b[k].shape or sd_a[k].shape != sd_ref[k].shape:
            raise ValueError(
                f"Shape mismatch at {k}: {sd_a[k].shape} vs {sd_b[k].shape} vs {sd_ref[k].shape}"
            )

    # Identify parameter names to optionally exclude buffers
    param_keyset = set(dict(model.named_parameters()).keys())

    # --- Helper to set interpolated parameters θ(t) ---
    def _set_interp_params(t: float):
        m_sd = model.state_dict()
        one_minus_t = 1.0 - t
        for k in keys:
            # Skip buffers if requested
            if not include_buffers and (k not in param_keyset):
                continue
            dst = m_sd[k]
            a = sd_a[k].to(dst.device, dtype=dst.dtype)
            b = sd_b[k].to(dst.device, dtype=dst.dtype)
            dst.copy_(one_minus_t * a + t * b)
        model.load_state_dict(m_sd, strict=True)

    # --- Loss on loader (evaluation mode, no grad) ---
    criterion = loss_fn

    def _eval_loss() -> float:
        model.eval()
        total_loss, denom = 0.0, 0
        model.to(device)
        for batch in data_loader:
            batch = batch.to(device) if hasattr(batch, "to") else batch
            logits = forward_fn(model, batch, device, **forward_kwargs)
            pred = logits.view(-1) if logits.size(1) == 1 else logits
            seed_time = batch[forward_kwargs['src_table']].seed_time
            pred = pred[:seed_time.size(0)]
            if isinstance(criterion, str):
                loss_fn = get_loss_fn(criterion)
            else:
                loss_fn = criterion
            if criterion == 'cross_entropy':
                loss = loss_fn(pred.float(), batch[forward_kwargs['src_table']].y.long())
            else:
                loss = loss_fn(pred.float(), batch[forward_kwargs['src_table']].y.float())
            if weight_by_batch_size:
                # Weight by batch size (works for vector targets; falls back to 1)
                n = int(pred.shape[0]) if pred.ndim > 0 else 1
                total_loss += float(loss.detach()) * n
                denom += n
            else:
                total_loss += float(loss.detach())
                denom += 1
        return total_loss / max(denom, 1)

    # --- Save original weights and run the t-sweep ---
    orig_state = {k: v.clone() for k, v in sd_ref.items()}
    ts = torch.linspace(0.0, 1.0, steps=max(2, num_points))
    losses = []

    try:
        for t in tqdm(ts.tolist(), desc="Computing loss barrier"):
            _set_interp_params(t)
            losses.append(_eval_loss())
    finally:
        # Always restore the original parameters
        m_sd = model.state_dict()
        for k, v in orig_state.items():
            m_sd[k].copy_(v.to(m_sd[k].device, dtype=m_sd[k].dtype))
        model.load_state_dict(m_sd, strict=True)

    # --- Compute barriers ---
    loss0, loss1 = losses[0], losses[-1]

    # Absolute barrier: max_t L(t) - max(L(0), L(1))
    max_loss = max(losses)
    barrier_abs = max_loss - max(loss0, loss1)
    t_at_abs = ts[int(np.argmax(losses))].item()

    # Convex-combination barrier: max_t [ L(t) - ((1 - t) * L(0) + t * L(1)) ]
    convex_gaps = []
    for i, t in enumerate(ts.tolist()):
        lin = (1.0 - t) * loss0 + t * loss1
        convex_gaps.append(losses[i] - lin)
    barrier_conv = max(convex_gaps)
    t_at_conv = ts[int(np.argmax(convex_gaps))].item()

    # Normalize barrier values by dividing by max(loss0, loss1)
    max_endpoint_loss = max(loss0, loss1)
    barrier_abs_normalized = barrier_abs / max_endpoint_loss if max_endpoint_loss > 0 else 0.0
    barrier_conv_normalized = barrier_conv / max_endpoint_loss if max_endpoint_loss > 0 else 0.0

    # Compute relative gap for every t: Δ_rel(t) = [L(t) - ((1-t)L(0) + tL(1))] / max{L(0), L(1)}
    relative_gaps = []
    for i, t in enumerate(ts.tolist()):
        linear_interp = (1.0 - t) * loss0 + t * loss1
        relative_gap = (losses[i] - linear_interp) / max_endpoint_loss if max_endpoint_loss > 0 else 0.0
        relative_gaps.append(relative_gap)

    # Normalize outputs to ensure consistent data types
    return {
        "ts": ts.detach().cpu().numpy() if torch.is_tensor(ts) else np.array(ts),
        "losses": [float(loss) for loss in losses],
        "loss0": float(loss0),
        "loss1": float(loss1),
        "barrier_abs": float(barrier_abs),
        "barrier_conv": float(barrier_conv),
        "barrier_abs_normalized": float(barrier_abs_normalized),
        "barrier_conv_normalized": float(barrier_conv_normalized),
        "relative_gaps": [float(gap) for gap in relative_gaps],
        "t_at_abs": float(t_at_abs),
        "t_at_conv": float(t_at_conv),
    }


def time_it_with_params(func):
    def wrapper(*args, **kwargs):
        # Extract the parameters we want to log
        method_type = kwargs.get('method_type', 'training-free')
        model_name = kwargs.get('model_name', 'loss-barrier')
        barrier_strategy = kwargs.get('barrier_strategy', 'epoch')
        
        def custom_logger(msg):
            logger.info(f"{msg} | method_type={method_type}, model_name={model_name}, barrier_strategy={barrier_strategy}")
        
        # Apply the time_it decorator with our custom logger
        timed_func = time_it(logger=custom_logger)(func)
        return timed_func(*args, **kwargs)
    return wrapper

@time_it_with_params
def generate_model_task_embedding(database: Union[MultiTabularDataset, DBBRDBDataset], task_id: int, basic_config: dict, anchor_config: dict = DEFAULT_ANCHOR_CONFIG, method_type = 'training-free', model_name = 'loss-barrier', barrier_strategy: str = 'epoch', model_type: str = 'rdl'):
    wanted_loss_fn = {
        'binary_classification': 'cross_entropy',
        'multiclass_classification': 'cross_entropy',
        'regression': 'mae',
        'recommendation': 'cross_entropy',
    }
    training_loss_fn = {
        'binary_classification': 'bce',
        'multiclass_classification': 'ce',
        'regression': 'mae',
        'recommendation': 'bce',
    }
    if method_type == 'training-free':
        task_embedding_list = []
        for model_config in anchor_config:
            model_config_path = model_config['model_config_path']
            repeats = model_config['repeats']
            task_emb = TaskEmb(
                root_dir = f'{model_config_path}',
                task_id = task_id
            )
            for i in range(repeats):
                seed = ANCHOR_SEEDS[i]
                real_model_config = load_yaml(model_config_path)
                real_model_config.update(basic_config)
                task_emb.configure_model_and_loaders(real_model_config, database, task_id)
                # task_emb.fit_head_from_cache()
                if model_name == 'auto-transfer':
                    task_emb.montecarlo_fisher(wanted_loss_fn[database.tasks[task_id].task_type if \
                        isinstance(database.tasks[task_id].task_type, str) else database.tasks[task_id].task_type.value])
                    output = task_emb.extract_embedding()
                elif model_name == 'loss-barrier':
                    output = task_emb.compute_loss_barrier('random_init', real_model_config, database, max_batches=basic_config['max_steps_per_epoch'])
                else:
                    raise ValueError(f'Unsupported model name {model_name}')
                task_embedding_list.append(output)
        return np.array(task_embedding_list)
    elif method_type == 'training-head':
        task_embedding_list = []
        for model_config in anchor_config:
            model_config_path = model_config['model_config_path']
            repeats = model_config['repeats']
            real_model_config = load_yaml(model_config_path)
            basic_config['model_params'] = real_model_config
            task_emb = TaskEmb(
                root_dir = f'{model_config_path}',
                task_id = task_id,
                model_type = model_type
            )
            task_emb.clear_cache()
            if model_type == 'dfs':
                model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn, task_type, num_classes, tabnn_model_config = task_emb.configure_dfs_model_and_loaders(database, basic_config)
                ## doesn't support debug
                task_emb.cache_features_dfs(model, train_loader, tabnn_model_config, task_type, val_loader, max_batches=basic_config['max_steps_per_epoch'], clear_existing = True, debug_mode = False)
            elif model_type == 'rdl':
                task_emb.configure_model_and_loaders(basic_config, database, task_id)
                task_emb.cache_features(max_batches=basic_config['max_steps_per_epoch'], model = task_emb.model, model_init = task_emb.model_init, clear_existing=True)
            for i in range(repeats):
                seed = ANCHOR_SEEDS[i]
                real_model_config = load_yaml(model_config_path)
                real_model_config.update(basic_config)
                if model_name == 'auto-transfer':
                    if model_type == 'rdl':
                        task_emb.fit_head_from_cache(loss_fn = training_loss_fn[database.tasks[task_id].task_type if \
                            isinstance(database.tasks[task_id].task_type, str) else database.tasks[task_id].task_type.value])
                        task_emb.montecarlo_fisher(wanted_loss_fn[database.tasks[task_id].task_type if \
                            isinstance(database.tasks[task_id].task_type, str) else database.tasks[task_id].task_type.value])
                        output = task_emb.extract_embedding()
                    elif model_type == 'dfs':
                        task_emb.fit_head_from_cache(loss_fn = training_loss_fn[task_type], model = model)
                        task_emb.montecarlo_fisher_dfs(val_loader, wanted_loss_fn[task_type], real_loss = loss_fn)
                        output = task_emb.extract_embedding_dfs()
                elif model_name == 'loss-barrier':
                    task_emb.configure_model_and_loaders(real_model_config, database, seed)
                    barrier_selection = 'start_end_epoch_only_head' if barrier_strategy == 'epoch' else 'train_2_only_head'
                    output = task_emb.compute_loss_barrier(barrier_selection, real_model_config, database, max_batches=basic_config['max_steps_per_epoch'], loss_fn = training_loss_fn[database.tasks[task_id].task_type if \
                        isinstance(database.tasks[task_id].task_type, str) else database.tasks[task_id].task_type.value])
                else:
                    raise ValueError(f'Unsupported model name {model_name}')
                task_embedding_list.append(output)
        return np.array(task_embedding_list)
    elif method_type == 'training-full':
        task_embedding_list = []
        for model_config in anchor_config:
            model_config_path = model_config['model_config_path']
            repeats = model_config['repeats']
            task_emb = TaskEmb(
                root_dir = f'{model_config_path}',
                task_id = task_id
            )
            for i in range(repeats):
                seed = ANCHOR_SEEDS[i]
                real_model_config = load_yaml(model_config_path)
                real_model_config.update(basic_config)
                task_emb.configure_model_and_loaders(real_model_config, database, seed)
                # task_emb.cache_features()
                if model_name == 'auto-transfer':
                    task_emb.fit_full(real_model_config, loss_fn = training_loss_fn[database.tasks[task_id].task_type if \
                        isinstance(database.tasks[task_id].task_type, str) else database.tasks[task_id].task_type.value])
                    task_emb.montecarlo_fisher(wanted_loss_fn[database.tasks[task_id].task_type if \
                        isinstance(database.tasks[task_id].task_type, str) else database.tasks[task_id].task_type.value])
                    output = task_emb.extract_embedding()
                elif model_name == 'loss-barrier':
                    barrier_selection = 'start_end_epoch_full' if barrier_strategy == 'epoch' else 'train_2_full'
                    output = task_emb.compute_loss_barrier(barrier_selection, real_model_config, database, max_batches=basic_config['max_steps_per_epoch'], loss_fn = training_loss_fn[database.tasks[task_id].task_type if \
                        isinstance(database.tasks[task_id].task_type, str) else database.tasks[task_id].task_type.value])
                else:
                    raise ValueError(f'Unsupported model name {model_name}')
                task_embedding_list.append(output)
        return np.array(task_embedding_list)
    else:
        raise ValueError(f'Unsupported method type {method_type}')