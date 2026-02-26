from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import wandb
from loguru import logger
from relbench.base import EntityTask, RecommendationTask
from relbench.base.table import Table
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from tqdm import tqdm

from data.mtaskdataset import MultiTabularDataset, TaskType, prepare_transductive_rdb_datasets_save_memory
from dbinfer.device import get_device_info
from dbinfer.solutions.rel_tabnn_solutions import lightgbm_solution, rel_tabnn_solution
from dbinfer.solutions.tabpfn_solutions import TabPFNSolution, TabPFNSolutionConfig
from dbinfer.solutions.tabular_dataset_config import parse_config_from_tabular_dataset
from models import get_auto_model
from utils.database import NullSwapDB, SwapDB, create_swap_db
from models.dfs import generate_dfs_data_pre_post
from utils.input import initialize_wandb
from utils.misc import load_yaml, save_dict_as_yaml
from utils.type import seed_everything

torch_geometric.backend.use_segment_matmul = False
warnings.filterwarnings("ignore")

# Metric weights used when aggregating validation scores.
METRIC_WEIGHTS = {
    "average_precision": 0,
    "acc": 0,
    "f1": 0,
    "roc_auc": 1,
    "rmse": -1,
    "mae": -1,
    "r2": 1,
    "ndcg": 1,
    "map": 1,
}


@dataclass(frozen=True)
class TrainLoopConfig:
    """Configuration bundle for a single training epoch."""

    max_steps_per_epoch: int
    share_same_time: bool


def _resolve_task_tables(task: Union[EntityTask, RecommendationTask]) -> Tuple[str, str]:
    """Return the entity and target table names for the given task."""

    if task.task_type == TaskType.LINK_PREDICTION or task.task_type == "link_prediction":
        return task.src_entity_table, task.dst_entity_table
    return task.entity_table, task.entity_table




def get_loss_fn(loss_fn: str, pos_weight: Optional[float] = None) -> torch.nn.Module:
    """Map a string identifier to a PyTorch loss function."""

    normalized = str(loss_fn).lower()
    if normalized in {"bce", "logistic"} or isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        if pos_weight is not None:
            weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            return torch.nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        return torch.nn.BCEWithLogitsLoss()
    if normalized in {"ce", "cross_entropy"} or isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        return torch.nn.CrossEntropyLoss()
    if normalized == "mse" or isinstance(loss_fn, torch.nn.MSELoss):
        return torch.nn.MSELoss()
    if normalized == "mae" or isinstance(loss_fn, torch.nn.L1Loss):
        return torch.nn.L1Loss()
    raise ValueError(f"Invalid loss function: {loss_fn}")


def train_for_link_loader(
    batch: HeteroData,
    model: torch.nn.Module,
    task: Union[EntityTask, RecommendationTask],
    device: torch.device,
    share_same_time: bool,
    loss_fn: str = "bpr",
) -> tuple[Tensor, int]:
    """Compute the loss for a batch sampled by a LinkNeighborLoader."""

    src_batch, batch_pos_dst, batch_neg_dst = batch
    src_batch, batch_pos_dst, batch_neg_dst = (
        src_batch.to(device),
        batch_pos_dst.to(device),
        batch_neg_dst.to(device),
    )
    x_src = model(
        src_batch,
        task.src_entity_table,
        task.dst_entity_table,
        read_out_table="src",
    )[: src_batch[task.src_entity_table]["seed_time"].size(0)]
    x_pos_dst = model(
        batch_pos_dst,
        task.dst_entity_table,
        task.dst_entity_table,
        read_out_table="dst",
    )[: batch_pos_dst[task.dst_entity_table]["seed_time"].size(0)]
    x_neg_dst = model(
        batch_neg_dst,
        task.dst_entity_table,
        task.dst_entity_table,
        read_out_table="dst",
    )[: batch_neg_dst[task.dst_entity_table]["seed_time"].size(0)]

    normalized_loss_name = loss_fn.lower()
    pos_score = torch.sum(x_src * x_pos_dst, dim=1)
    if share_same_time:
        neg_score = x_src @ x_neg_dst.t()
        pos_score = pos_score.view(-1, 1)
    else:
        neg_score = torch.sum(x_src * x_neg_dst, dim=1)

    if normalized_loss_name == "bpr":
        diff = pos_score.unsqueeze(-1) - neg_score
        loss = F.softplus(-diff).mean()

    elif normalized_loss_name == "logistic":
        neg_flat_score = neg_score.view(-1)
        pos_flat_score = pos_score.view(-1)
        all_scores = torch.cat([pos_flat_score, neg_flat_score], dim=0)
        labels = torch.cat(
            [torch.ones_like(pos_flat_score), torch.zeros_like(neg_flat_score)],
            dim=0,
        )
        loss = F.binary_cross_entropy_with_logits(all_scores, labels)
    else:
        raise ValueError(f"Invalid loss function: {loss_fn}")

    return loss, x_src.size(0)



def generate_representation(
    model: torch.nn.Module,
    device: torch.device,
    batch: HeteroData,
    entity_table: str,
    target_table: str,
    meta_graph: Optional[Tuple[Data, dict]],
) -> Tensor:
    """Return node embeddings while optionally injecting meta-graph context."""

    if meta_graph is None:
        return model(batch, entity_table, target_table, read_out_table="dst")

    meta_g, meta_info = meta_graph
    return model(
        batch,
        entity_table,
        target_table,
        read_out_table="dst",
        meta_g=meta_g.to(device),
        meta_info=meta_info,
    )


def pretrain_model(
    model: torch.nn.Module,
    database: MultiTabularDataset,
    task: Union[EntityTask, RecommendationTask],
    num_dst_nodes_dict: dict,
    val_res_task,
    task_meta,
    meta_graph: Optional[Tuple[Data, dict]],
    sparse_tensor,
    train_loader,
    val_loader,
    train_config: TrainLoopConfig,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: ExponentialLR,
    device: torch.device,
    loader_type: str = "node",
    scaler=None,
) -> tuple[float, dict[str, float]]:
    """Train for one epoch and report validation metrics."""

    model.train()
    model.to(device)
    entity_table, target_table = _resolve_task_tables(task)
    total_steps = min(len(train_loader), train_config.max_steps_per_epoch)
    steps = 0
    pbar = tqdm(train_loader, total=total_steps, desc="Train")

    if sparse_tensor is not None:
        sparse_tensor._crow_indices = sparse_tensor._crow_indices.to(device)
        sparse_tensor._col_indices = sparse_tensor._col_indices.to(device)

    numel = 0
    total_loss = 0.0
    for batch in pbar:
        optimizer.zero_grad()
        try:
            if loader_type == "node":
                batch = batch.to(device)
                target_output = generate_representation(
                    model,
                    device,
                    batch,
                    entity_table,
                    target_table,
                    meta_graph,
                )

                if sparse_tensor is None:
                    pred = target_output.view(-1) if target_output.size(1) == 1 else target_output
                    seed_time = batch[entity_table].seed_time
                    pred = pred[: seed_time.size(0)]
                    loss_fn = get_loss_fn(task_meta['loss_fn'], task_meta.get('pos_weight'))
                    if task_meta['loss_fn'] == 'ce':
                        loss = loss_fn(pred.float(), batch[entity_table].y.long())
                    else:
                        loss = loss_fn(pred.float(), batch[entity_table].y.float())
                    numel += pred.size(0)
                    total_loss += loss.item() * pred.size(0)
                else:
                    if hasattr(model, 'rhs_embedding'):
                        input_id = batch[entity_table].input_id
                        src_batch, dst_index = sparse_tensor[input_id]
                        logits = target_output
                        edge_label_index = torch.stack([src_batch, dst_index], dim=0)
                        loss = sparse_cross_entropy(logits, edge_label_index)
                        numel += len(batch[task.dst_entity_table].batch)
                        total_loss += loss.item() * len(batch[task.dst_entity_table].batch)
                    else:
                        pred = target_output.flatten()
                        batch_size = batch[entity_table].batch_size
                        input_id = batch[entity_table].input_id
                        src_batch, dst_index = sparse_tensor[input_id]
                        target = torch.isin(
                            batch[target_table].batch
                            + batch_size * batch[target_table].n_id,
                            src_batch + batch_size * dst_index,
                        ).float()
                        loss_fn = get_loss_fn('bce')
                        loss = loss_fn(pred, target)
                        numel += pred.size(0)
                        total_loss += loss.item() * pred.size(0)
            else:
                loss, single_numel = train_for_link_loader(
                    batch,
                    model,
                    task,
                    device,
                    train_config.share_same_time,
                    task_meta['loss_fn'],
                )
                total_loss += loss.item() * single_numel
                numel += single_numel
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            steps += 1
            if steps > train_config.max_steps_per_epoch:
                break
        except ValueError as exc:
            logger.error(f"Error: {exc}")
            continue
    train_loss = total_loss / numel if numel > 0 else 0
    if numel == 0:
        warnings.warn(
            f"Did not sample a single '{task.dst_entity_table}' "
            f"node in any mini-batch. Try to increase the number "
            f"of layers/hops and re-try. If you run into memory "
            f"issues with deeper nets, decrease the batch size."
        )
    wandb.log({f"train_loss_{database.database.name}_{task.name}": train_loss})
    metric = test_model(
        model,
        task,
        num_dst_nodes_dict,
        task_meta,
        meta_graph,
        val_loader,
        val_res_task,
        device,
        stage='val',
        loader_type=loader_type,
        scaler=scaler,
    )
    named_metric = {
        f"val_{key}_{database.database.name}_{task.name}": value
        for key, value in metric.items()
    }
    wandb.log(named_metric)
    logger.debug(f"Val metrics: {metric}")
    lr_scheduler.step()
    return train_loss, metric


@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    task: Union[EntityTask, RecommendationTask],
    num_dst_nodes_dict,
    task_meta,
    meta_graph: Optional[Tuple[Data, dict]],
    loader,
    res_task,
    device: torch.device,
    stage: str = "val",
    loader_type: str = "node",
    scaler=None,
    return_predictions: bool = False,
):
    """Evaluate *model* using the provided loader and return metric dict."""

    if len(loader) == 0:
        logger.debug(f"No data for {stage} {task.name}")
        if return_predictions:
            return {}, None
        return {}

    model.eval()
    model.to(device)
    entity_table, target_table = _resolve_task_tables(task)

    if isinstance(loader, tuple):
        src_loader, dst_loader = loader
    pred_list = []

    if loader_type == "node":
        pbar = tqdm(loader, total=len(loader), desc=f"{stage} Test")
        for batch in pbar:
            batch = batch.to(device)
            target_output = generate_representation(
                model,
                device,
                batch,
                entity_table,
                target_table,
                meta_graph,
            )
            if task.task_type == TaskType.LINK_PREDICTION or task.task_type == "link_prediction":
                if hasattr(model, 'rhs_embedding'):
                    pred = target_output.detach()
                    scores = torch.sigmoid(pred)
                else:
                    pred = target_output.detach().flatten()
                    batch_size = batch[entity_table].batch_size
                    scores = torch.zeros(batch_size, num_dst_nodes_dict[stage], device=device)
                    scores[batch[target_table].batch, batch[target_table].n_id] = torch.sigmoid(pred)
                _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
                pred_list.append(pred_mini.cpu())
            else:
                pred = target_output.view(-1) if target_output.size(1) == 1 else target_output
                if task_meta['loss_fn'] == 'ce':
                    pred = torch.softmax(pred, dim=1)
                elif task_meta['loss_fn'] == 'bce':
                    pred = torch.sigmoid(pred)
                # Inverse-transform predictions to original target scale for regression before clamping
                if scaler is not None and (task.task_type == TaskType.REGRESSION or str(task.task_type).lower() == 'regression'):
                    try:
                        mean_val = float(getattr(scaler, 'mean_', [0.0])[0])
                    except Exception:
                        mean_val = 0.0
                    try:
                        scale_val = float(getattr(scaler, 'scale_', [1.0])[0])
                    except Exception:
                        scale_val = 1.0
                    mean_t = torch.as_tensor(mean_val, device=pred.device, dtype=pred.dtype)
                    scale_t = torch.as_tensor(scale_val, device=pred.device, dtype=pred.dtype)
                    pred = pred * scale_t + mean_t
                if task_meta.get('clamp_min', None) is not None and task_meta.get('clamp_max', None) is not None:
                    pred = torch.clamp(pred, task_meta['clamp_min'], task_meta['clamp_max'])
                seed_time = batch[entity_table].seed_time
                pred = pred[: seed_time.size(0)]
                pred = pred.view(-1) if (pred.ndim > 1 and pred.size(1) == 1) else pred
                pred_list.append(pred.detach().cpu())
        predictions_np = torch.cat(pred_list, dim=0).numpy()
        metric = task.evaluate(predictions_np, res_task)
    else:
        dst_embs: list[Tensor] = []
        for batch in tqdm(dst_loader, total=len(dst_loader), desc=f"{stage} Embedding"):
            batch = batch.to(device)
            emb = model(
                batch,
                task.dst_entity_table,
                task.dst_entity_table,
                read_out_table='dst',
            )[: batch[task.dst_entity_table]['seed_time'].size(0)]
            dst_embs.append(emb)
        dst_emb = torch.cat(dst_embs, dim=0)
        del dst_embs

        pred_index_mat_list: list[Tensor] = []
        for batch in tqdm(src_loader, total=len(src_loader), desc=f"{stage} Source"):
            batch = batch.to(device)
            emb = model(
                batch,
                task.src_entity_table,
                task.dst_entity_table,
                read_out_table='src',
            )[: batch[task.src_entity_table]['seed_time'].size(0)]
            _, pred_index_mat = torch.topk(emb @ dst_emb.t(), k=task.eval_k, dim=1)
            pred_index_mat_list.append(pred_index_mat.cpu())
        predictions_np = torch.cat(pred_index_mat_list, dim=0).numpy()
        metric = task.evaluate(predictions_np, res_task)

    if return_predictions:
        return metric, predictions_np
    return metric


def sync_evaluate_model(
    model: torch.nn.Module,
    database: MultiTabularDataset,
    model_config: dict,
    device: torch.device,
    last=False,
    split='test',
    return_predictions: bool = False,
):
    eval_model = deepcopy(model)
    # Fine-tune the model if specified
    loader_dict, num_dst_nodes_dict, task_meta, meta_graph, sparse_tensor, sampled_val_table = database.initialize_data(0, 0, 1,
                                                                  model_config['model_params']['sampler_config']['loader_type'], model_config)
    test_loader = loader_dict[split]
    if split == 'val':
        test_res_task = sampled_val_table
    else:
        test_res_task = database.tasks[0].get_table(split, mask_input_cols=False)

    # Get scaler if normalization is enabled
    scaler = database.get_scaler(0) if database.normalize_target else None

    eval_result = test_model(
        eval_model,
        database.tasks[0],
        num_dst_nodes_dict,
        task_meta,
        meta_graph,
        test_loader,
        test_res_task,
        device,
        stage=split,
        loader_type=model_config['model_params']['sampler_config']['loader_type'],
        scaler=scaler,
        return_predictions=return_predictions,
    )
    if return_predictions:
        test_metric, predictions = eval_result
    else:
        test_metric = eval_result
        predictions = None
    named_test_metric = {f"{split}_{key}_{database.database.name}_{database.tasks[0].name}":
                         value for key, value in test_metric.items()}
    wandb.log(named_test_metric)
    logger.debug(f"{split} metrics: {test_metric}")
    if return_predictions:
        return test_metric, predictions
    return test_metric


def configure_rdl(model_init: dict, database: MultiTabularDataset, model_config: dict, task_info: Union[EntityTask, RecommendationTask]):
    # Populate model-specific metadata
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
    model_init['model_config'] = model_config['model_params']
    model_init['strategy'] = model_config['model_params']['gnn_config']['pre_sf']
    node_types, edge_types = database.get_node_edge_types()
    model_init['node_types'] = node_types
    model_init['edge_types'] = edge_types
    model_init['data_metadata'] = database.get_data_metadata()
    return model_init


def _save_prediction_artifact(
    predictions: Union[np.ndarray, torch.Tensor, dict],
    *,
    db_name: str,
    task_name: str,
    model_name: str,
    split: str = "test",
) -> Path:
    """Persist predictions to disk and return the saved path."""

    ground_truth = None
    if isinstance(predictions, dict):
        ground_truth = predictions.get("ground_truth")
        predictions = predictions.get("predictions")

    predictions_dir = Path("results") / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    artifact_name = f"{db_name}_{task_name}_{model_name}_{split}"
    save_path = predictions_dir / f"{artifact_name}.pt"

    def _to_tensor(value: Union[np.ndarray, torch.Tensor, None]):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return torch.as_tensor(value)

    tensor = _to_tensor(predictions)
    label_tensor = _to_tensor(ground_truth)

    if tensor is None:
        raise ValueError("Predictions payload is empty; nothing to save.")

    payload = {
        "name": artifact_name,
        "db_name": db_name,
        "task_name": task_name,
        "model_name": model_name,
        "split": split,
        "predictions": tensor.cpu(),
        "created_at": datetime.now().isoformat(),
    }
    if label_tensor is not None:
        payload["ground_truth"] = label_tensor.cpu()
    torch.save(payload, save_path)
    logger.info(f"Saved predictions to {save_path}")
    return save_path



def sync_train_process(database: MultiTabularDataset, model_config: dict, full_name,
                       save_model: bool, recovery_file: str, recovery_directory: str,
                       result_db: SwapDB, device: str = 'cuda', eval_val: bool = False, save_results: bool = False, wandb_online: bool = False):
    """Training loop for a single GPU run (recovery arguments kept for compatibility)."""

    # Start overall timing
    total_start_time = time.time()
    
    device = torch.device(device)
    run_command = " ".join(sys.argv)
    model_config["run_command"] = run_command
    if wandb_online:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"
    # always initialize this otherwise wandb.log will throw errors
    initialize_wandb(model_config["wandb_project"], full_name, model_config)

    model = get_auto_model(model_config['model_params']["model_type"])
    model_init = {**model_config['model_params']}
    model_init['col_stats_dict'] = database.get_col_stats_dict()
    task_info = database.tasks[0]
    # Convert task type to string for easier manipulation
    task_info.task_type = task_info.task_type.value if not isinstance(task_info.task_type, str) else task_info.task_type
    if task_info.task_type == "recommendation" or task_info.task_type == "link_prediction":
        model_init['src_table'] = task_info.src_entity_table
        model_init['dst_table'] = task_info.dst_entity_table
    else:
        model_init['src_table'] = task_info.entity_table
        model_init['dst_table'] = task_info.entity_table
    if model_config['model_params']["model_type"] == "rdl":
        model_init = configure_rdl(model_init, database, model_config, task_info)
    model = model(**model_init)
    if model_config['compile_model']:
        model = torch.compile(model)
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=model_config["model_params"]["optimizer_config"]["lr"],
        weight_decay=float(model_config["model_params"]["optimizer_config"]["weight_decay"])
    )

    # When gamma is 1, the scheduler is effectively disabled
    lr_scheduler = ExponentialLR(optimizer, gamma=model_config["model_params"]["optimizer_config"]["gamma"])
    curr_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modelcache_dir = Path("modelcache")
    modelcache_dir.mkdir(exist_ok=True)
    random_str = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    model_weight_name = modelcache_dir / f"{database.database.name}-{database.tasks[0].name}" / (
        f"{full_name}_{model_config['model_params']['model_type']}"
        f"_{curr_timestamp}_{random_str}.pth"
    )
    model_weight_name.parent.mkdir(parents=True, exist_ok=True)
    
    best_model: Optional[torch.nn.Module] = None
    starting_epoch = 1
    best_val_metric = -float('inf')
    early_stop_counter = 0
    best_val_metric_record: dict[str, float] = {}

    # Time data initialization
    data_init_start = time.time()
    loader_dict, num_dst_nodes_dict, task_meta, meta_graph, sparse_tensor, val_res_task = \
            database.initialize_data(0, 0, 1, model_config['model_params']['sampler_config']['loader_type'], model_config)
    data_init_time = time.time() - data_init_start

    # Get scaler if normalization is enabled
    scaler = database.get_scaler(0) if database.normalize_target else None

    # Log data initialization time
    if wandb.run is not None:
        wandb.log({"data_initialization_time": data_init_time})

    epoch = 0
    for epoch in range(starting_epoch, model_config["epochs"] + 1):
        average_val_metric = defaultdict(list)
        
        train_config = TrainLoopConfig(
            max_steps_per_epoch=model_config["max_steps_per_epoch"],
            share_same_time=model_config["model_params"].get("share_same_time", True),
        )

        # Time training phase
        train_start = time.time()
        task = database.tasks[0]
        # weight = -1 if task.task_type == TaskType.REGRESSION else 1
        if (isinstance(task.task_type, str) and task.task_type == "multiclass_classification") or task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            task_meta['loss_fn'] = "ce"
            task_meta['pos_weight'] = None
        ## deal with wrong config of loss_fn
        elif task.task_type == TaskType.REGRESSION or str(task.task_type).lower() == 'regression':
            if model_config['model_params']['gnn_config']['loss_fn'] not in ["mse", "mae"]:
                task_meta['loss_fn'] = "mae"
            task_meta['pos_weight'] = None
        else:
            task_meta['loss_fn'] = model_config['model_params']['gnn_config']['loss_fn']
            if task_meta['loss_fn'] in {"bce", "logistic"}:
                task_meta['pos_weight'] = model_config['model_params'].get('pos_weight')
            else:
                task_meta['pos_weight'] = None

        train_loss, val_metric = pretrain_model(
            model,
            database,
            task,
            num_dst_nodes_dict,
            val_res_task,
            task_meta,
            meta_graph,
            sparse_tensor,
            loader_dict['train'],
            loader_dict['val'],
            train_config,
            optimizer,
            lr_scheduler,
            device,
            loader_type=model_config['model_params']['sampler_config']['loader_type'],
            scaler=scaler,
        )
        train_time = time.time() - train_start

        #result_db.add_a_row(model_config, {"train_loss": train_loss}, "running", epoch, "train_loss")
        #result_db.add_a_row(model_config, val_metric, "running", epoch, "val_metric")

            
        logger.debug(f"Train loss: {train_loss}, Val metric: {val_metric}")
        for key, value in val_metric.items():
            average_val_metric[key].append(value * METRIC_WEIGHTS.get(key, 1))

        average_val_metric = {key: sum(value) / len(value) for key, value in average_val_metric.items()}
        
        # Time evaluation phase if enabled
        eval_time = 0
        if model_config["eval_in_train"] > 0 and epoch % model_config["eval_in_train"] == 0:
            eval_start = time.time()
            test_metric = sync_evaluate_model(model, database, model_config, device)
            eval_time = time.time() - eval_start
            result_db.add_a_row(model_config, test_metric, "running", epoch, "test_metric")
        
        # Log timing metrics to wandb
        if wandb.run is not None:
            wandb.log({
                f"epoch_{epoch}/train_time": train_time,
                f"epoch_{epoch}/eval_time": eval_time,
                f"epoch_{epoch}/total_epoch_time": train_time + eval_time
            })
        
        if np.average(list(average_val_metric.values())) > best_val_metric:
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

                
    logger.debug(f"Performing test on best model")
    
    # Time final test evaluation
    final_test_start = time.time()
    prediction_path: Optional[Path] = None
    test_predictions = None
    # Use the best model directly instead of loading state dict
    if best_model is not None:
        if save_model:
            torch.save(best_model.state_dict(), model_weight_name)
            wandb.log({"model_weight_name": str(model_weight_name)})
        best_model = best_model.to(device)
        if not eval_val:
            if save_results:
                test_metric, test_predictions = sync_evaluate_model(
                    best_model,
                    database,
                    model_config,
                    device,
                    last=model_config.get("last_full", False),
                    return_predictions=True,
                )
            else:
                test_metric = sync_evaluate_model(
                    best_model,
                    database,
                    model_config,
                    device,
                    last=model_config.get("last_full", False),
                )
        else:
            test_metric = {}
            val_metric = sync_evaluate_model(best_model, database, model_config, device, 
                            last=model_config.get("last_full", False), split='val')
    else:
        # Fallback to loading state dict if no best model was saved
        logger.debug(f"No best model found, using the last model")
        if not eval_val:
            if save_results:
                test_metric, test_predictions = sync_evaluate_model(
                    model,
                    database,
                    model_config,
                    device,
                    last=model_config.get("last_full", False),
                    return_predictions=True,
                )
            else:
                test_metric = sync_evaluate_model(model, database, model_config, device, 
                                last=model_config.get("last_full", False))
        else:
            test_metric = {}
            val_metric = sync_evaluate_model(model, database, model_config, device, 
                            last=model_config.get("last_full", False), split='val')
    final_test_time = time.time() - final_test_start
    if save_results and not eval_val and test_predictions is not None:
        prediction_path = _save_prediction_artifact(
            test_predictions,
            db_name=database.database.name,
            task_name=database.tasks[0].name,
            model_name=model_config['model_params']['model_type'],
            split="test",
        )
    
    # Calculate total training time
    total_training_time = time.time() - total_start_time
    
    # Log final timing metrics to wandb
    if wandb.run is not None:
        wandb.log({
            "final_test_time": final_test_time,
            "total_training_time": total_training_time,
            "data_initialization_time": data_init_time,
            "average_epoch_time": (total_training_time - data_init_time - final_test_time) / max(1, epoch - starting_epoch + 1)
        })
    
    result_db.add_a_row(
        model_config,
        {"test_metric": test_metric, "val_metric": best_val_metric_record},
        "completed",
        -1,
    )

    full_dict = {
        "config": model_config,
        "results": {
            "test_metric": test_metric,
            "val_metric": best_val_metric_record if not eval_val else val_metric,
        },
        "status": "completed",
        "epoch": epoch,
        "result_type": "test_metric",
        "model_weight_name": str(model_weight_name),
    }
    if prediction_path is not None:
        full_dict["prediction_path"] = str(prediction_path)

    return full_dict

def wrap_searched_params_into_tabnn_config(model_config: dict):
    """Adapt the search-space config into the format expected by downstream solvers."""

    params = model_config["model_params"]
    # auto_config = params["auto_config"]
    model_type = params["torch_frame_model_cls"]


    if model_type == "tabpfn":
        config = TabPFNSolutionConfig()
        config.max_train_samples = 10000
        config.training_sample_selection_strategy = "random"
        config.remove_add_feature = True
        config.remove_cat_temporal_information = True
        return config

    if model_type == "lightgbm":
        return {
            "num_trials": params.get("num_trials", 20),
            "remove_add_feature": True,
            "remove_cat_temporal_information": True,
        }

    if model_type in {"resnet", "ft_transformer"}:
        config: dict[str, Union[int, float, str]] = {}
        if model_type == "resnet":
            tabnn_cfg = params["tabnn_config"]
            config["channels"] = tabnn_cfg["hidden_size"]
            config["out_channels"] = tabnn_cfg["hidden_size"]
            config["num_layers"] = tabnn_cfg["num_layers"]
            config["normalization"] = tabnn_cfg["normalization"]
            config["dropout_prob"] = tabnn_cfg["dropout"]
        else:
            ft_cfg = params["tab_nn_config"]
            config["channels"] = ft_cfg["hidden_size"]
            config["out_channels"] = ft_cfg["hidden_size"]
            config["num_layers"] = ft_cfg["num_layers"]
        config["nn_config"] = config  # Maintain existing self-reference semantics
        config["lr"] = params["optimizer_config"]["lr"]
        config["batch_size"] = params["batch_size"]
        config["eval_batch_size"] = params["batch_size"]
        config["negative_sampling_ratio"] = params.get("negative_sampling_ratio", 1)
        config["nn_name"] = model_type
        config["weight_decay"] = params["optimizer_config"]["weight_decay"]
        config["scheduler"] = params["optimizer_config"]["scheduler"]
        config["gamma"] = params["optimizer_config"]["gamma"]
        config["loss_fn"] = params["loss_fn"]
        config["remove_add_feature"] = True
        config["remove_cat_temporal_information"] = True
        config["max_steps_per_epoch"] = model_config["max_steps_per_epoch"]
        return config

    raise ValueError(f"Unsupported torch_frame_model_cls: {model_type}")

def sync_train_process_fs(database: MultiTabularDataset, model_config: dict, full_name,
                          save_model: bool, recovery_file: str, result_db: SwapDB,
                          text_to_pca: bool = False, text_to_pca_dim: int = 3, save_results: bool = False, wandb_online: bool = False,
                          task_label_overrides: Optional[Dict[str, Table]] = None,
                          dataset_name_suffix: Optional[str] = None):
    """Feature-store training pipeline (recovery arg kept for compatibility).

    Args:
        dataset_name_suffix: Optional suffix to differentiate cached datasets when using different label configurations
    """

    total_start_time = time.time()

    params = model_config["model_params"]
    params["text_to_pca"] = text_to_pca
    params["text_to_pca_dim"] = text_to_pca_dim
    model_type = params["torch_frame_model_cls"]

    autog_suffix = "autog" if database.autog else ""
    run_name = (
        f"dfs_{database.database.name}_{database.tasks[0].name}_"
        f"{params['dfs_layer']}_{autog_suffix}"
    )
    if wandb_online:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"
    initialize_wandb(model_config["wandb_project"], run_name, model_config)

    scaler_tag = (
        f"scaler_{database.database.name}_{database.tasks[0].name}_"
        f"{params['dfs_layer']}"
    )
    scaler_path = f"./scalers/{scaler_tag}"

    dfs_start = time.time()
    rdb_dataset = generate_dfs_data_pre_post(
        "datasets/dfs",
        database,
        max_depth=params['dfs_layer'],
        load_scalers_from=scaler_tag,
        lte=database.tasks[0].auto,
        text_to_pca=text_to_pca,
        text_to_pca_dim=text_to_pca_dim,
        agg_primitives=params.get('agg_primitives', {}).get('ops', None),
        primitive_suffix=params.get('agg_primitives', {}).get('name', '') if params.get('agg_primitives', None) else '',
    )
    dfs_time = time.time() - dfs_start
    logger.info(f"DFS data generation time: {dfs_time}")

    task_meta = database.get_task_meta(0, 1)
    clamp_min = task_meta.get("clamp_min")
    clamp_max = task_meta.get("clamp_max")

    def run_tabpfn():
        train_start = time.time()
        task_config = parse_config_from_tabular_dataset(rdb_dataset, database.tasks[0].name)
        solver_config = wrap_searched_params_into_tabnn_config(model_config)
        solver_config.use_no_text_datasets = model_config.get("use_no_text_datasets", False)
        tabpfn = TabPFNSolution(solver_config, task_config)
        device = get_device_info()
        tabpfn.fit(
            rdb_dataset,
            database.tasks[0].name,
            os.path.join(model_config['cache_dir'], 'tabpfn'),
            device,
            scaler_path,
        )
        train_time = time.time() - train_start

        eval_start = time.time()
        val_metrics_local = tabpfn.evaluate(
            rdb_dataset.tasks[0].validation_set,
            device,
            scaler_path,
            is_regression=database.tasks[0].task_type == TaskType.REGRESSION,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        test_eval = tabpfn.evaluate(
            rdb_dataset.tasks[0].test_set,
            device,
            scaler_path,
            is_regression=database.tasks[0].task_type == TaskType.REGRESSION,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            return_predictions=save_results,
        )
        eval_time = time.time() - eval_start
        if save_results:
            test_metrics_local, test_predictions, test_labels = test_eval
            test_predictions_payload = {
                'predictions': test_predictions,
                'ground_truth': test_labels,
            }
        else:
            test_metrics_local = test_eval
            test_predictions_payload = None
        return val_metrics_local, test_metrics_local, train_time, eval_time, None, test_predictions_payload

    def run_tabnn():
        train_start = time.time()
        solver_config = wrap_searched_params_into_tabnn_config(model_config)
        task_config = parse_config_from_tabular_dataset(rdb_dataset, database.tasks[0].name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = "".join(random.choices(string.ascii_letters + string.digits, k=5))
        cache_dir = Path("modelcache")
        cache_dir.mkdir(exist_ok=True)
        model_weight_path = cache_dir / (
            f"model_weight_{database.database.name}_{database.tasks[0].name}_"
            f"{params['dfs_layer']}_{timestamp}_{random_str}.pt"
        )
        output = rel_tabnn_solution(
            rdb_dataset,
            database.tasks[0].name,
            model_type,
            solver_config,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            save_model=save_model,
            model_weight_name=str(model_weight_path),
            return_predictions=save_results,
            dataset_name_suffix=dataset_name_suffix,
        )
        if save_model:
            wandb.log({"model_weight_name": str(model_weight_path)})
        train_time = time.time() - train_start
        predictions_payload = None
        if output.get('test_predictions') is not None or output.get('test_ground_truth') is not None:
            predictions_payload = {
                'predictions': output.get('test_predictions'),
                'ground_truth': output.get('test_ground_truth'),
            }
        return (
            output['best_val_metrics'],
            output['best_test_metrics'],
            train_time,
            0.0,
            output['model_weight_name'],
            predictions_payload,
        )

    def run_lightgbm():
        train_start = time.time()
        solver_config = wrap_searched_params_into_tabnn_config(model_config)
        _ = parse_config_from_tabular_dataset(rdb_dataset, database.tasks[0].name)
        result = lightgbm_solution(
            rdb_dataset,
            database.database.name,
            database.tasks[0].name,
            solver_config,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            return_predictions=save_results,
        )
        train_time = time.time() - train_start
        predictions_payload = None
        if result.get('test_predictions') is not None or result.get('test_ground_truth') is not None:
            predictions_payload = {
                'predictions': result.get('test_predictions'),
                'ground_truth': result.get('test_ground_truth'),
            }
        return (
            result['val_metrics'],
            result['test_metrics'],
            train_time,
            0.0,
            None,
            predictions_payload,
        )

    if model_type == "tabpfn":
        val_metrics, test_metrics, model_train_time, model_eval_time, saved_model, test_predictions = run_tabpfn()
    elif model_type in {"resnet", "ft_transformer"}:
        val_metrics, test_metrics, model_train_time, model_eval_time, saved_model, test_predictions = run_tabnn()
    elif model_type == "lightgbm":
        val_metrics, test_metrics, model_train_time, model_eval_time, saved_model, test_predictions = run_lightgbm()
    else:
        raise ValueError(f"Unsupported torch_frame_model_cls: {model_type}")

    total_time = time.time() - total_start_time

    logger.info(f"Test metrics: {test_metrics}")
    logger.info(f"Val metrics: {val_metrics}")

    wandb.log({
        "dfs_data_generation_time": dfs_time,
        "model_training_time": model_train_time,
        "model_evaluation_time": model_eval_time,
        "total_execution_time": total_time,
    })
    wandb.log({f"test_{key}": value for key, value in test_metrics.items()})
    wandb.log({f"val_{key}": value for key, value in val_metrics.items()})

    prediction_path = None
    if save_results and test_predictions is not None:
        prediction_path = _save_prediction_artifact(
            test_predictions,
            db_name=database.database.name,
            task_name=database.tasks[0].name,
            model_name=model_type,
            split="test",
        )

    result_db.add_a_row(
        model_config,
        {'test_metrics': test_metrics, 'val_metrics': val_metrics},
        "completed",
        -1,
    )

    if test_metrics.get('auroc') is not None:
        test_metrics['roc_auc'] = test_metrics['auroc']
    if val_metrics.get('auroc') is not None:
        val_metrics['roc_auc'] = val_metrics['auroc']

    result = {
        "config": model_config,
        "results": {
            "test_metric": test_metrics,
            "val_metric": val_metrics,
        },
        "status": "completed",
        "epoch": -1,
        "result_type": "test_metric",
        "model_weight_name": saved_model,
    }
    if prediction_path is not None:
        result["prediction_path"] = str(prediction_path)
    return result


class SameParams(Exception):
    """Exception raised when duplicate search parameters are found."""
    pass

class SameParamsMoreGPU(Exception):
    """Exception raised when same architecture is found but with more GPUs."""
    pass


def execution(search_params, basic_config, data_config, save_model, recovery_file, recovery_directory,
              result_db: SwapDB, device: str = 'cuda', noexit: bool = False, text_to_pca: bool = False,
              text_to_pca_dim: int = 3, eval_val: bool = False, compile_model: bool = False, test: bool = False, save_results: bool = False, wandb_online: bool = False):
    """Run training for a single configuration (recovery args kept for compatibility)."""
    seed_everything(basic_config['seed'])
    ## transductive is True since we focus on training from scratch
    dataset_iter = prepare_transductive_rdb_datasets_save_memory(
        data_config,
        basic_config,
        normalize_target=basic_config.get("normalize_target", False),
        transductive=True,
        autocomplete_full=search_params.get("autocomplete_full", True),
    )
    database = next(dataset_iter)
    # Exit codes:
    # 0: success
    # 5: unexpected memory error
    # 6: CUDA out of memory 
    try: 
        model_config = {
            **basic_config,
            **data_config
        }
        data_config = data_config["dbs"][0]
        model_config['model_params'] = search_params
        model_config['compile_model'] = compile_model
        full_name = f"{data_config['db_name']}_{data_config['task_name'][0]}"
        if not test and result_db.get_rows_with_the_same_search_params(model_config):
            raise SameParams()
        # elif result_db.find_same_arch_gpu_out_rows(model_config):
        #     raise SameParamsMoreGPU()
        else:
            logger.info(f"New search params found: {model_config}")
        if search_params['model_type'] == 'rdl':
            sync_train_process(database, model_config, full_name, save_model, recovery_file, recovery_directory, result_db, device, eval_val, save_results, wandb_online)
        elif search_params['model_type'] == 'fs':
            sync_train_process_fs(database, model_config, full_name, save_model, recovery_file, result_db, text_to_pca, text_to_pca_dim, save_results, wandb_online)
        elif search_params['model_type'] == 'ng':
            pass 
        # signal.signal(signal.SIGKILL, handle_unexpected_memory_error)
        if result_db.test:
            # Don't exit here otherwise the whole program exits
            return 
        if not noexit:
            sys.exit(0)
    except SameParams as e:
        logger.info(f"Duplicate search params found. Exiting.")
        sys.exit(4)
    except SameParamsMoreGPU as e:
        logger.info(f"Same arch but more gpu found. Exiting.")
        sys.exit(3)
    except MemoryError as e:
        logger.error(f"Memory error. Exiting.")
        sys.exit(5)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            result_db.add_a_row(model_config, {}, "memory_out", -1)
            logger.error(f"CUDA out of memory. Exiting.")
            sys.exit(6)
        else:
            logger.error(f"Error in execution: {e}")
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-name", type=str, default="rel-f1")
    parser.add_argument("--task-name", type=str, default="driver-dnf")
    parser.add_argument("--path", type=str, default="", help="If not null string, \
                        this can load custom dataset")
    parser.add_argument("--rdl-params", type=str, default="{}")
    parser.add_argument("--params_type", type=str, default="json", choices=["json", "file"])
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--basic-config", type=str, default="configs/default/basic.yaml")
    parser.add_argument("--debug-save", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--noexit", action="store_true")
    parser.add_argument("--eval-val", action="store_true")
    parser.add_argument("--text-to-pca", action="store_true")
    parser.add_argument("--text-to-pca-dim", type=int, default=3)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--pos-weight", type=float, default=None, help="Optional positive-class weight for BCE loss")
    parser.add_argument("--wandb-online", action="store_true")
    parser.add_argument("--no-db", action="store_true", help="Disable MongoDB (use NullSwapDB no-op)")
    args = parser.parse_args()
    basic_config = load_yaml(args.basic_config)
    seed_everything(basic_config['seed'])
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if args.params_type == "json":
        rdl_args = json.loads(args.rdl_params)
        if len(rdl_args) == 0:
            # If no RDL params provided, sample one at random
            from utils.space import search
            rdl_args = search(model_type="rdl", task_type="binary_classification")
        if args.debug_save != "":
            save_dict_as_yaml(rdl_args, args.debug_save)
    else:
        rdl_args = load_yaml(args.rdl_params)
    if args.pos_weight is not None:
        rdl_args["pos_weight"] = args.pos_weight
    data_config = {
        "dbs": [
            {
                "db_name": args.db_name,
                "task_name": [args.task_name],
                "path": args.path
            }
        ]
    }
    if args.no_db:
        result_db = NullSwapDB()
    else:
        result_db = create_swap_db(
            backend=basic_config.get("db_backend", "sqlite"),
            path_to_db=basic_config["db_location"],
            collection_name=basic_config["result_db_name"],
            db_name=basic_config["db_name"],
        )
    result_db.test = args.test
    if not rdl_args.get("model_type", None):
        # Default to RDL
        rdl_args["model_type"] = "rdl"
    rdl_args["time_free"] = rdl_args.get("time_free", False)
    rdl_args["zero_init"] = rdl_args.get("zero_init", False)
    logger.info("Starting execution")
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    execution(
        rdl_args,
        basic_config,
        data_config,
        args.save_model,
        None,
        None,
        result_db,
        args.device,
        args.noexit,
        args.text_to_pca,
        args.text_to_pca_dim,
        args.eval_val,
        args.compile_model,
        args.test,
        args.save_results,
        args.wandb_online
    )
