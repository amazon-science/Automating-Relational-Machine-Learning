import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from swap.execution import generate_representation, get_loss_fn
from data.mtaskdataset import MultiTabularDataset, TaskType
from tqdm import tqdm
from torch_frame.data import DataLoader
from torch_frame import Metric
import joblib
from typing import List, Dict, Tuple, Optional
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import contextlib
import math
from dbinfer.evaluator import get_metrics


@torch.no_grad()
def calculate_loss_and_metric(loader, stage, task, loader_type, model, device, 
                              entity_table, target_table, sparse_tensor, task_meta,
                              num_dst_nodes_dict, res_task):
    numel = 0
    total_loss = 0
    pred_list = []
    gt_list = []
    for batch in tqdm(loader):
        ## clean hop to save memory
        # if isinstance(batch, HeteroData) and hasattr(batch[entity_table], 'hop'):
        #     batch[entity_table].hop = batch[entity_table].hop[entity_table]
        if loader_type == 'node':
            batch = batch.to(device)
            target_output = generate_representation(model, device, batch, 
                                                    entity_table, target_table, None)            
            ## entity-classification/ regression task
            pred = target_output.view(-1) if target_output.size(1) == 1 else target_output
            seed_time = batch[entity_table].seed_time
            pred = pred[:seed_time.size(0)]
            if isinstance(task_meta['loss_fn'], str):
                loss_fn = get_loss_fn(task_meta['loss_fn'])
            elif callable(task_meta['loss_fn']):
                loss_fn = task_meta['loss_fn']
            else:
                # If it's a dict or other type, try to extract the loss function name
                loss_fn_name = task_meta.get('loss_fn_name', 'ce')  # default to cross entropy
                loss_fn = get_loss_fn(loss_fn_name)
            loss = loss_fn(pred.float(), batch[entity_table].y.float())
            numel += pred.size(0)
            total_loss += loss.item() * pred.size(0)
            pred_list.append(pred.detach().cpu())
            gt_list.append(batch[entity_table].y.detach().cpu())
    pred_list = torch.cat(pred_list, dim=0)
    gt_list = torch.cat(gt_list, dim=0)
    # metric = task.evaluate(pred_list, gt_list)
    metric = {fn.__name__: fn(gt_list, pred_list) for fn in task.metrics}
    train_loss = total_loss / numel if numel > 0 else 0
    return train_loss, metric


@torch.no_grad()
def calculate_ftt_loss_and_metric(loader, model, device, task_type, num_classes, loss_fn, 
                                 scaler_path=None, clamp_min=None, clamp_max=None):
    """
    Calculate loss and metrics for FT-Transformer models following the style from rel_tabnn_solutions.py.
    
    Args:
        loader: DataLoader containing tensor frames
        model: FT-Transformer model
        device: Device to run on
        task_type: Type of task ('classification', 'binary_classification', 'regression', 'multiclass_classification')
        num_classes: Number of classes (1 for regression, >=2 for classification)
        loss_fn: Loss function to use
        scaler_path: Path to scaler for regression tasks (optional)
        clamp_min: Minimum value for clamping regression predictions (optional)
        clamp_max: Maximum value for clamping regression predictions (optional)
        
    Returns:
        Tuple of (loss, metric_dict) where metric_dict contains various metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_count = 0
    
    is_classification = task_type in ['classification', 'binary_classification', 'multiclass_classification']
    
    for tf in tqdm(loader, desc="Evaluating FTT"):
        tf = tf.to(device)
        pred = model(tf)
        
        # Calculate loss
        if is_classification:
            if pred.dim() == 2 and pred.shape[1] == 1:
                pred_loss = pred.squeeze(1)
                loss = loss_fn(pred_loss, tf.y.view(-1).float())
            else:
                loss = loss_fn(pred, tf.y.view(-1).long())
        else:
            loss = loss_fn(pred.view(-1), tf.y.view(-1))
        
        total_loss += float(loss) * len(tf.y)
        total_count += len(tf.y)
        
        # Store predictions and labels
        all_preds.append(pred.cpu())
        all_labels.append(tf.y.cpu())
    
    # Combine all predictions and labels
    pred = torch.cat(all_preds, dim=0)
    label = torch.cat(all_labels, dim=0)
    
    # Calculate average loss
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    
    # Get appropriate metrics for the task
    metric_fns = get_metrics(task_type, num_classes)
    
    # Calculate metrics
    metrics = {}
    for metric_name, metric_fn in metric_fns.items():
        try:
            if task_type == 'classification' or task_type == 'binary_classification':
                # For classification, ensure proper shapes
                if num_classes == 2:
                    # Binary classification: for torchmetrics, we need probabilities for both classes
                    if pred.dim() == 2 and pred.shape[1] == 1:
                        # Convert single logit to 2-class probabilities
                        probs_neg = torch.sigmoid(-pred.squeeze(1))  # P(class=0)
                        probs_pos = torch.sigmoid(pred.squeeze(1))   # P(class=1)
                        pred_metric = torch.stack([probs_neg, probs_pos], dim=1)
                    elif pred.dim() == 1:
                        # Convert single logit to 2-class probabilities
                        probs_neg = torch.sigmoid(-pred)  # P(class=0)
                        probs_pos = torch.sigmoid(pred)   # P(class=1)
                        pred_metric = torch.stack([probs_neg, probs_pos], dim=1)
                    elif pred.dim() == 2 and pred.shape[1] == 2:
                        # Already in correct format
                        pred_metric = pred
                    else:
                        pred_metric = pred.reshape(-1, num_classes)
                    
                    # Ensure label is 1D
                    if label.dim() > 1:
                        label_metric = label.squeeze()
                    else:
                        label_metric = label
                else:
                    # Multi-class classification: pred should be (N, num_classes), label should be (N,)
                    pred_metric = pred.reshape(-1, num_classes)
                    if label.dim() > 1:
                        label_metric = label.squeeze()
                    else:
                        label_metric = label
                
                metrics[metric_name] = metric_fn(None, pred_metric, label_metric).item()
                
            elif task_type == 'regression':
                # Regression: both pred and label should be 1D
                pred_flat = pred.reshape(-1)
                label_flat = label.reshape(-1)
                
                # Apply scaler inverse transform if available
                if scaler_path is not None:
                    try:
                        loaded_scaler = list(joblib.load(scaler_path / 'target_scalers.pkl').values())[0]
                        pred_flat = torch.from_numpy(loaded_scaler.inverse_transform(pred_flat.reshape(-1, 1))).reshape(-1)
                        label_flat = torch.from_numpy(loaded_scaler.inverse_transform(label_flat.reshape(-1, 1))).reshape(-1)
                    except Exception as e:
                        print(f"Warning: Could not apply scaler inverse transform: {e}")
                
                # Apply clamping if specified
                if clamp_min is not None and clamp_max is not None:
                    pred_flat = torch.clamp(pred_flat, clamp_min, clamp_max)
                    label_flat = torch.clamp(label_flat, clamp_min, clamp_max)
                
                metrics[metric_name] = metric_fn(None, pred_flat, label_flat).item()
                
            elif task_type == 'multiclass_classification':
                pred_metric = pred.reshape(-1, num_classes)
                if label.dim() > 1:
                    label_metric = label.squeeze()
                else:
                    label_metric = label
                metrics[metric_name] = metric_fn(None, pred_metric, label_metric).item()
                
            else:
                # Other task types
                metrics[metric_name] = metric_fn(None, pred, label).item()
                
        except Exception as e:
            print(f"Warning: Could not calculate {metric_name}: {e}")
            metrics[metric_name] = 0.0
    
    # Return primary metric for compatibility with landscape analysis
    # Use the first available metric as the primary one
    primary_metric = list(metrics.values())[0] if metrics else 0.0
    
    return avg_loss, primary_metric

# --- Small wrapper so we can pass a fixed set of batches as a "loader"-like object ---
class _StaticBatchesLoader:
    def __init__(self, batches: List):
        self._batches = batches
    def __iter__(self):
        for b in self._batches:
            yield b
    def __len__(self):
        return len(self._batches)

class GNNLossLandscape:
    """
    Scalable loss-landscape analysis for GNNs on your MultiTabularDataset.

    Features:
      - Filter-wise normalization for direction vectors (scale-invariant).
      - Flat-vector param updates for fast inner loops.
      - Probe loader to evaluate on a fixed small subset (repeatable & cheap).
      - Metrics-only mode (Hessian + multi-ray) -> O(9 + M*K) evals, not G^2.
      - Optional AMP and torch.compile for faster inference.
      - Accelerated full-grid plotting retained (if you still want visuals).
    """

    def __init__(self, model: nn.Module, dataset, device: str, model_config: dict):
        if not isinstance(model, nn.Module):
            raise TypeError("The 'model' argument must be a PyTorch nn.Module.")

        self.model = model
        self.device = device
        self.dataset = dataset

        task_type_to_metrics = {
            'regression': 'mae',
            'classification': 'bce',
            'recommendation': 'map',
            'binary_classification': 'bce',
        }

        # get_loss_fn must exist in your codebase
        from swap.execution import get_loss_fn  # adjust import path if needed
        first_task_type = str(dataset.tasks[0].task_type)
        self.criterion = get_loss_fn(str(task_type_to_metrics[first_task_type]))
        self.initial_state = deepcopy(model.state_dict())
        self.task = dataset.tasks[0]

        loader_type = model_config['model_params']['sampler_config']['loader_type']
        (
            loader_dict,
            num_dst_nodes_dict,
            task_meta,
            meta_graph,
            sparse_tensor,
            sampled_val_table
        ) = dataset.initialize_data(
            0, 0, 1, loader_type, model_config
        )

        self.train_loader = loader_dict['train']
        self.val_loader = loader_dict['val']
        self.test_loader = loader_dict['test']
        self.num_dst_nodes_dict = num_dst_nodes_dict
        self.task_meta = task_meta
        self.meta_graph = meta_graph
        self.train_sparse_tensors = sparse_tensor
        self.val_res_task = sampled_val_table
        self.test_res_task = dataset.tasks[0].get_table('test', mask_input_cols=False)

        # cache for probe subsets by (loader_id, k)
        self._probe_cache: Dict[Tuple[int, int], _StaticBatchesLoader] = {}
        self._temp_model = None  # set in methods that need it


        self._calc = calculate_loss_and_metric

    # ----------------------------
    # Utilities
    # ----------------------------
    def _params_list(self, model: Optional[nn.Module] = None):
        m = self.model if model is None else model
        return [p for p in m.parameters() if p.requires_grad]

    def _create_random_directions(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Create two filter-wise normalized random directions.
        Direction 2 is made orthogonal to direction 1 per-parameter tensor.
        """
        params = self._params_list(self.model)

        # Direction 1
        direction_1 = []
        for p in params:
            rand_d1 = torch.randn_like(p, device=self.device)
            p_norm = torch.norm(p.data, p='fro')
            d1_norm = torch.norm(rand_d1, p='fro')
            if d1_norm > 1e-8:
                rand_d1.mul_(p_norm / d1_norm)
            direction_1.append(rand_d1)

        # Direction 2 (orthogonal to d1)
        direction_2 = []
        for i, p in enumerate(params):
            rand_d2 = torch.randn_like(p, device=self.device)
            d1 = direction_1[i]
            dot = torch.sum(d1 * rand_d2)
            norm_sq = torch.sum(d1 * d1)
            rand_d2.sub_((dot / (norm_sq + 1e-12)) * d1)

            p_norm = torch.norm(p.data, p='fro')
            d2_norm = torch.norm(rand_d2, p='fro')
            if d2_norm > 1e-8:
                rand_d2.mul_(p_norm / d2_norm)
            direction_2.append(rand_d2)

        return direction_1, direction_2

    @torch.no_grad()
    def _precompute_flat_base_and_dirs(self):
        """
        Flatten base params and two directions for fast updates.
        Returns a temp_model (deepcopy, eval), its param-list, and three flat vectors.
        """
        params = self._params_list(self.model)
        d1_tensors, d2_tensors = self._create_random_directions()

        base_vec = parameters_to_vector(params).detach()
        dir1_vec = parameters_to_vector(d1_tensors).detach()
        dir2_vec = parameters_to_vector(d2_tensors).detach()

        temp_model = deepcopy(self.model).to(self.device).eval()
        temp_params = self._params_list(temp_model)
        return temp_model, temp_params, base_vec, dir1_vec, dir2_vec

    @torch.no_grad()
    def _load_alpha_beta(self, temp_params, base_vec, dir1_vec, dir2_vec, alpha: float, beta: float):
        vec = base_vec.add(alpha, dir1_vec).add(beta, dir2_vec)  # fused axpy style
        vector_to_parameters(vec, temp_params)

    def _resolve_entity_target_tables(self) -> Tuple[str, str]:
        if self.task_meta['task_type'] == (TaskType.LINK_PREDICTION if TaskType else "link_prediction") or \
           self.task_meta['task_type'] == "link_prediction":
            entity_table = self.task.src_entity_table
            target_table = self.task.dst_entity_table
        else:
            entity_table = self.task.entity_table
            target_table = self.task.entity_table
        return entity_table, target_table

    def _get_stage_loader(self, stage: str):
        if stage == "train":
            return self.train_loader
        elif stage == "val":
            return self.val_loader
        elif stage == "test":
            return self.test_loader
        raise ValueError(f"Unknown stage: {stage}")

    def _make_probe_loader(self, loader, *, max_batches: int = 8) -> _StaticBatchesLoader:
        """
        Materialize a fixed small subset of batches from `loader`.
        This list is cached so repeated evals are consistent and cheap.
        """
        key = (id(loader), int(max_batches))
        if key in self._probe_cache:
            return self._probe_cache[key]

        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i + 1 >= max_batches:
                break
        if not batches:
            raise RuntimeError("Probe loader could not collect any batches (is the loader empty?)")

        static_loader = _StaticBatchesLoader(batches)
        self._probe_cache[key] = static_loader
        return static_loader

    # ----------------------------
    # Fast eval at a single (alpha, beta)
    # ----------------------------
    @torch.no_grad()
    def _eval_loss_at(self, temp_params, base_vec, d1_vec, d2_vec,
                      alpha: float, beta: float, loader_like, stage: str,
                      entity_table: str, target_table: str):
        self._load_alpha_beta(temp_params, base_vec, d1_vec, d2_vec, alpha, beta)
        loss, metric = self._calc(
            loader_like, stage, self.task, 'node', self._temp_model, self.device,
            entity_table, target_table, self.train_sparse_tensors,
            self.task_meta, self.num_dst_nodes_dict, self.val_res_task
        )
        # metric may be a tensor, float, or dict
        if isinstance(metric, dict):
            metric_val = float(list(metric.values())[0])
        else:
            metric_val = float(metric.item() if hasattr(metric, "item") else metric)
        return float(loss), metric_val

    # ----------------------------
    # Metrics-only mode (O(9 + M*K) evals)
    # ----------------------------
    def compute_metrics_only(
        self,
        stage: str = "val",
        epsilons: Tuple[float, ...] = (0.01, 0.02, 0.05),
        ray_angles_deg: Tuple[float, ...] = (0, 30, 60, 90, 120, 150),  # 6 rays (add more if you want)
        radii: np.ndarray = np.linspace(0.0, 1.0, 21),
        probe_batches: int = 8,
        use_amp: bool = True,
        compile_model: bool = True
    ) -> Dict[str, float]:
        """
        Compute quantitative metrics without building a full 2D grid.
        Returns a dict of numbers you can log/compare across runs.
        """
        base_loader = self._get_stage_loader(stage)
        probe_loader = self._make_probe_loader(base_loader, max_batches=probe_batches)
        entity_table, target_table = self._resolve_entity_target_tables()

        temp_model, temp_params, base_vec, d1_vec, d2_vec = self._precompute_flat_base_and_dirs()
        if compile_model:
            try:
                temp_model = torch.compile(temp_model, mode="reduce-overhead")
            except Exception:
                pass
        self._temp_model = temp_model

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_amp and torch.cuda.is_available()) else contextlib.nullcontext()

        with amp_ctx, torch.no_grad():
            # Finite differences around 0 for Hessian (decoupled step size)
            h = 1.0 / 30.0
            f00, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, 0.0, 0.0, probe_loader, stage, entity_table, target_table)
            fpp, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, +h, 0.0, probe_loader, stage, entity_table, target_table)
            fpm, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, -h, 0.0, probe_loader, stage, entity_table, target_table)
            fp2, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, 0.0, +h, probe_loader, stage, entity_table, target_table)
            fm2, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, 0.0, -h, probe_loader, stage, entity_table, target_table)
            fppmm, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, +h, +h, probe_loader, stage, entity_table, target_table)
            fpmmp, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, +h, -h, probe_loader, stage, entity_table, target_table)
            fmpmp, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, -h, +h, probe_loader, stage, entity_table, target_table)
            fmmmm, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, -h, -h, probe_loader, stage, entity_table, target_table)

            faa = (fpp - 2*f00 + fpm) / (h**2)
            fbb = (fp2 - 2*f00 + fm2) / (h**2)
            fab = (fppmm - fpmmp - fmpmp + fmmmm) / (4*h*h)

            H = np.array([[faa, fab], [fab, fbb]], dtype=np.float64)
            evals, _ = np.linalg.eigh(H)
            lam_min, lam_max = float(evals[0]), float(evals[1])
            condH = abs(lam_max) / (abs(lam_min) + 1e-12)
            anisotropy = abs(faa - fbb) / (abs(faa) + abs(fbb) + 1e-12)
            froH = float(np.sqrt(faa*faa + fbb*fbb + 2.0*fab*fab))

            # Rays (barriers + widths)
            angles = np.deg2rad(np.array(ray_angles_deg, dtype=np.float64))
            barriers = []
            widths_eps = {eps: [] for eps in epsilons}
            for theta in angles:
                ca, cb = math.cos(theta), math.sin(theta)
                losses = []
                for r in radii:
                    a, b = (float(r * ca), float(r * cb))
                    fl, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, a, b,
                                               probe_loader, stage, entity_table, target_table)
                    losses.append(fl)
                losses = np.array(losses, dtype=np.float64)
                barriers.append(float(np.max(losses) - f00))
                for eps in epsilons:
                    thr = f00 * (1.0 + eps)
                    ok = np.where(losses <= thr)[0]
                    rad = float(radii[ok[-1]]) if len(ok) else 0.0
                    widths_eps[eps].append(rad)

        metrics = {
            "loss_center": float(f00),
            "sharpness_lambda_max": float(lam_max),
            "curvature_lambda_min": float(lam_min),
            "traceH": float(faa + fbb),
            "froH": froH,
            "condH": float(condH),
            "anisotropy": float(anisotropy),
            "barrier_ray_mean": float(np.mean(barriers)),
            "barrier_ray_max": float(np.max(barriers)),
        }
        for eps, vals in widths_eps.items():
            metrics[f"ray_width_mean@{int(eps*100)}%"] = float(np.mean(vals))
            metrics[f"ray_width_min@{int(eps*100)}%"]  = float(np.min(vals))
            metrics[f"ray_width_max@{int(eps*100)}%"]  = float(np.max(vals))

        return metrics

    # ----------------------------
    # Full-grid plot (accelerated)
    # ----------------------------
    def plot(
        self,
        grid_points: int = 30,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = (-1.0, 1.0),
        stage: str = "val",
        figure_name: str = "loss_landscape.svg",
        probe_batches: int = 8,
        use_amp: bool = True,
        compile_model: bool = False
    ):
        """
        Compute a grid (still O(G^2), but accelerated) and save a contour plot.
        Returns (loss_grid, alphas, betas, center_metrics_dict).
        """
        base_loader = self._get_stage_loader(stage)
        probe_loader = self._make_probe_loader(base_loader, max_batches=probe_batches)
        entity_table, target_table = self._resolve_entity_target_tables()

        temp_model, temp_params, base_vec, dir1_vec, dir2_vec = self._precompute_flat_base_and_dirs()
        if compile_model:
            try:
                temp_model = torch.compile(temp_model, mode="reduce-overhead")
            except Exception:
                pass
        self._temp_model = temp_model

        alphas = np.linspace(x_range[0], x_range[1], grid_points)
        betas  = np.linspace(y_range[0], y_range[1], grid_points)
        loss_grid = np.empty((grid_points, grid_points), dtype=np.float32)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_amp and torch.cuda.is_available()) else contextlib.nullcontext()

        with amp_ctx, torch.no_grad():
            total = tqdm(total=grid_points * grid_points, desc="Calculating loss")
            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    self._load_alpha_beta(temp_params, base_vec, dir1_vec, dir2_vec, float(a), float(b))
                    loss, _ = self._calc(
                        probe_loader, stage, self.task, 'node', temp_model, self.device,
                        entity_table, target_table, self.train_sparse_tensors,
                        self.task_meta, self.num_dst_nodes_dict, self.val_res_task
                    )
                    loss_grid[j, i] = float(loss)
                    total.update(1)

        # --- Center metrics from the grid (optional but handy) ---
        center_metrics = self.compute_landscape_metrics(loss_grid, alphas, betas, epsilons=(0.01, 0.02, 0.05))

        # --- Plotting ---
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8))

        min_loss = float(np.min(loss_grid))
        max_loss = float(np.max(loss_grid))
        eps = 1e-8
        # Safe levels for strictly positive losses (BCE/MAE are >=0)
        lo = max(min_loss + eps, eps)
        hi = max(max_loss, lo * 1.05)
        levels = np.logspace(np.log10(lo), np.log10(hi), num=20)

        cs = ax.contour(alphas, betas, loss_grid, levels=levels, cmap='viridis')
        ax.clabel(cs, inline=True, fontsize=9, fmt='%1.3f')

        ax.set_title('GNN Loss Landscape', fontsize=18, fontweight='bold')
        ax.set_xlabel(r'Direction 1 ($\alpha$)', fontsize=14)
        ax.set_ylabel(r'Direction 2 ($\beta$)', fontsize=14)
        plt.savefig(figure_name, format=figure_name.split(".")[-1])
        # also save PDF for vector quality (optional)
        if not figure_name.endswith(".pdf"):
            plt.savefig(figure_name.rsplit(".", 1)[0] + ".pdf", format='pdf')
        plt.close(fig)

        return loss_grid, alphas, betas, center_metrics

    # ----------------------------
    # Rich grid-based metrics (if you already have a grid)
    # ----------------------------
    @staticmethod
    def _nearest_index(vec: np.ndarray, val: float) -> int:
        return int(np.abs(vec - val).argmin())

    @staticmethod
    def _contiguous_span_around_center(mask: np.ndarray, center: int) -> Tuple[int, int]:
        L = R = center
        n = len(mask)
        while L-1 >= 0 and mask[L-1]:
            L -= 1
        while R+1 < n and mask[R+1]:
            R += 1
        return L, R

    def _finite_difference_hessian(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> dict:
        ia0 = self._nearest_index(alphas, 0.0)
        ib0 = self._nearest_index(betas, 0.0)
        if not (1 <= ia0 < len(alphas)-1 and 1 <= ib0 < len(betas)-1):
            raise ValueError("Grid must include ±1 steps around (0,0). Expand range or grid_points.")

        f00 = float(loss_grid[ib0, ia0])
        da = float(alphas[1] - alphas[0])
        db = float(betas[1] - betas[0])

        faa = (loss_grid[ib0, ia0+1] - 2.0 * f00 + loss_grid[ib0, ia0-1]) / (da**2)
        fbb = (loss_grid[ib0+1, ia0] - 2.0 * f00 + loss_grid[ib0-1, ia0]) / (db**2)
        fab = (loss_grid[ib0+1, ia0+1] - loss_grid[ib0+1, ia0-1]
               - loss_grid[ib0-1, ia0+1] + loss_grid[ib0-1, ia0-1]) / (4.0 * da * db)

        H = np.array([[faa, fab], [fab, fbb]], dtype=np.float64)
        evals, evecs = np.linalg.eigh(H)
        lam_min, lam_max = float(evals[0]), float(evals[1])
        cond = abs(lam_max) / (abs(lam_min) + 1e-12)
        anisotropy = abs(faa - fbb) / (abs(faa) + abs(fbb) + 1e-12)

        return {
            "f00": f00,
            "H": H,
            "eigvals": (lam_min, lam_max),
            "traceH": float(np.trace(H)),
            "froH": float(np.sqrt(faa*faa + fbb*fbb + 2.0*fab*fab)),
            "condH": float(cond),
            "anisotropy": float(anisotropy),
            "principal_dir": evecs[:, 1].tolist(),
            "ia0": ia0, "ib0": ib0, "da": da, "db": db
        }

    def _flatness_and_widths(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                             f00: float, ia0: int, ib0: int, eps_rel: float = 0.05) -> dict:
        rel = (loss_grid - f00) / max(f00, 1e-12)
        da = float(alphas[1] - alphas[0])
        db = float(betas[1] - betas[0])
        area_cell = da * db
        area_total = (alphas[-1] - alphas[0]) * (betas[-1] - betas[0])

        area_eps = float(np.sum(rel <= eps_rel) * area_cell)
        area_ratio = float(area_eps / (area_total + 1e-12))

        # α-axis width
        row_mask = (rel[ib0, :] <= eps_rel)
        La, Ra = self._contiguous_span_around_center(row_mask, ia0)
        width_alpha = float(alphas[Ra] - alphas[La])

        # β-axis width
        col_mask = (rel[:, ia0] <= eps_rel)
        Lb, Rb = self._contiguous_span_around_center(col_mask, ib0)
        width_beta = float(betas[Rb] - betas[Lb])

        # main diagonal
        kmax1 = int(min(ia0, ib0))
        kmax2 = int(min(len(alphas)-ia0-1, len(betas)-ib0-1))
        diag_main_mask = []
        for k in range(-kmax1, kmax2+1):
            diag_main_mask.append(rel[ib0 + k, ia0 + k] <= eps_rel)
        center_k = kmax1
        Lk, Rk = self._contiguous_span_around_center(np.array(diag_main_mask, dtype=bool), center_k)
        width_diag_main = float(np.sqrt(
            (alphas[ia0+(Rk-center_k)] - alphas[ia0-(center_k-Lk)])**2 +
            (betas[ib0+(Rk-center_k)]  - betas[ib0-(center_k-Lk)])**2
        ))

        # anti-diagonal
        kmax1b = int(min(ia0, len(betas)-ib0-1))
        kmax2b = int(min(len(alphas)-ia0-1, ib0))
        diag_anti_mask = []
        for k in range(-kmax1b, kmax2b+1):
            diag_anti_mask.append(rel[ib0 - k, ia0 + k] <= eps_rel)
        center_k2 = kmax1b
        Lk2, Rk2 = self._contiguous_span_around_center(np.array(diag_anti_mask, dtype=bool), center_k2)
        width_diag_anti = float(np.sqrt(
            (alphas[ia0+(Rk2-center_k2)] - alphas[ia0-(center_k2-Lk2)])**2 +
            (betas[ib0-(Rk2-center_k2)]  - betas[ib0+(center_k2-Lk2)])**2
        ))

        robust_radius_eps = float(0.5 * min(width_alpha, width_beta))

        return {
            f"area@{int(eps_rel*100)}%": area_eps,
            f"area_ratio@{int(eps_rel*100)}%": area_ratio,
            f"width_alpha@{int(eps_rel*100)}%": width_alpha,
            f"width_beta@{int(eps_rel*100)}%": width_beta,
            f"width_diag_main@{int(eps_rel*100)}%": width_diag_main,
            f"width_diag_anti@{int(eps_rel*100)}%": width_diag_anti,
            f"robust_radius@{int(eps_rel*100)}%": robust_radius_eps,
        }

    def _global_sensitivity_and_ruggedness(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> dict:
        da = float(alphas[1] - alphas[0])
        db = float(betas[1] - betas[0])
        dfdβ, dfdα = np.gradient(loss_grid, db, da, edge_order=2)
        grad_norm = np.sqrt(dfdα**2 + dfdβ**2)
        lipschitz_max = float(np.nanmax(grad_norm))
        lipschitz_p95 = float(np.nanpercentile(grad_norm, 95))
        lipschitz_mean = float(np.nanmean(grad_norm))
        tv = float(np.mean(np.abs(dfdα)) + np.mean(np.abs(dfdβ)))

        lap = (np.roll(loss_grid, 1, axis=0) + np.roll(loss_grid, -1, axis=0) +
               np.roll(loss_grid, 1, axis=1) + np.roll(loss_grid, -1, axis=1) -
               4.0 * loss_grid) / ((da**2 + db**2) / 2.0)
        rugged_rms = float(np.sqrt(np.mean(lap**2)))

        return {
            "lipschitz_max": lipschitz_max,
            "lipschitz_p95": lipschitz_p95,
            "lipschitz_mean": lipschitz_mean,
            "total_variation": tv,
            "rugged_laplacian_rms": rugged_rms
        }

    def _barrier_indices(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                         ia0: int, ib0: int, f00: float) -> dict:
        row = loss_grid[ib0, :]
        col = loss_grid[:, ia0]
        B_alpha_pos = float(np.max(row[ia0:]) - f00)
        B_alpha_neg = float(np.max(row[:ia0+1]) - f00)
        B_beta_pos  = float(np.max(col[ib0:]) - f00)
        B_beta_neg  = float(np.max(col[:ib0+1]) - f00)

        kmax1 = int(min(ia0, ib0))
        kmax2 = int(min(len(alphas)-ia0-1, len(betas)-ib0-1))
        diag_main_vals_pos = [loss_grid[ib0 + k, ia0 + k] for k in range(0, kmax2+1)]
        diag_main_vals_neg = [loss_grid[ib0 - k, ia0 - k] for k in range(0, kmax1+1)]
        B_diag_main_pos = float(np.max(diag_main_vals_pos) - f00)
        B_diag_main_neg = float(np.max(diag_main_vals_neg) - f00)

        kmax1b = int(min(ia0, len(betas)-ib0-1))
        kmax2b = int(min(len(alphas)-ia0-1, ib0))
        diag_anti_vals_pos = [loss_grid[ib0 - k, ia0 + k] for k in range(0, kmax2b+1)]
        diag_anti_vals_neg = [loss_grid[ib0 + k, ia0 - k] for k in range(0, kmax1b+1)]
        B_diag_anti_pos = float(np.max(diag_anti_vals_pos) - f00)
        B_diag_anti_neg = float(np.max(diag_anti_vals_neg) - f00)

        return {
            "barrier_alpha_pos": B_alpha_pos,
            "barrier_alpha_neg": B_alpha_neg,
            "barrier_beta_pos":  B_beta_pos,
            "barrier_beta_neg":  B_beta_neg,
            "barrier_diag_main_pos": B_diag_main_pos,
            "barrier_diag_main_neg": B_diag_main_neg,
            "barrier_diag_anti_pos": B_diag_anti_pos,
            "barrier_diag_anti_neg": B_diag_anti_neg,
            "barrier_axis_mean": float(np.mean([B_alpha_pos, B_alpha_neg, B_beta_pos, B_beta_neg])),
            "barrier_diag_mean": float(np.mean([B_diag_main_pos, B_diag_main_neg, B_diag_anti_pos, B_diag_anti_neg])),
            "barrier_overall_max": float(np.max([
                B_alpha_pos, B_alpha_neg, B_beta_pos, B_beta_neg,
                B_diag_main_pos, B_diag_main_neg, B_diag_anti_pos, B_diag_anti_neg
            ]))
        }

    def _non_quadraticity_rmse(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                               ia0: int, ib0: int, f00: float, patch_radius: int = 2) -> float:
        i0, j0 = ia0, ib0
        iL = max(0, i0 - patch_radius); iR = min(len(alphas)-1, i0 + patch_radius)
        jL = max(0, j0 - patch_radius); jR = min(len(betas)-1,  j0 + patch_radius)

        A, y = [], []
        for j in range(jL, jR+1):
            for i in range(iL, iR+1):
                a = float(alphas[i])
                b = float(betas[j])
                A.append([1.0, a, b, 0.5*a*a, a*b, 0.5*b*b])
                y.append(float(loss_grid[j, i]))
        A = np.asarray(A, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ theta
        rmse = float(np.sqrt(np.mean((yhat - y)**2)))
        return rmse / max(f00, 1e-12)

    def compute_landscape_metrics(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                                  epsilons: Tuple[float, ...] = (0.01, 0.02, 0.05)) -> dict:
        """
        Quantitative metrics from an already-computed grid (loss_grid, alphas, betas).
        """
        hess = self._finite_difference_hessian(loss_grid, alphas, betas)
        ia0, ib0, f00 = hess["ia0"], hess["ib0"], hess["f00"]

        glb = {
            "loss_min": float(np.min(loss_grid)),
            "loss_max": float(np.max(loss_grid)),
            "loss_range": float(np.max(loss_grid) - np.min(loss_grid)),
            "loss_center": float(f00),
        }
        sens = self._global_sensitivity_and_ruggedness(loss_grid, alphas, betas)
        barriers = self._barrier_indices(loss_grid, alphas, betas, ia0, ib0, f00)

        flat = {}
        for eps in epsilons:
            flat.update(self._flatness_and_widths(loss_grid, alphas, betas, f00, ia0, ib0, eps_rel=eps))

        nonquad = {"nonquadraticity_rmse@5x5": self._non_quadraticity_rmse(loss_grid, alphas, betas, ia0, ib0, f00)}

        lam_min, lam_max = hess["eigvals"]
        curv = {
            "sharpness_lambda_max": float(lam_max),
            "curvature_lambda_min": float(lam_min),
            "traceH": hess["traceH"],
            "froH": hess["froH"],
            "condH": hess["condH"],
            "anisotropy": hess["anisotropy"],
        }

        out = {}
        out.update(glb)
        out.update(curv)
        out.update(flat)
        out.update(sens)
        out.update(barriers)
        out.update(nonquad)
        return out


class FTTLossLandscape:
    """
    Scalable loss-landscape analysis for FT-Transformer-style tabular models.

    Highlights:
      - Filter-wise normalized random directions (scale-invariant).
      - Flat-vector param updates (fast inner loop).
      - Probe loader (evaluate on a fixed small subset).
      - Metrics-only mode (Hessian + multi-ray) => O(9 + M*K) evals.
      - Optional AMP and torch.compile for faster inference.
      - Full-grid plotting retained but accelerated.

    Requirements:
      - A callable `calculate_ftt_loss_and_metric(loader, model, device, task_type, num_classes, loss_fn, scaler_path, clamp_min, clamp_max)`
        returning (loss, metric).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        task_type: str,
        num_classes: int,
        model_config: dict,
        *,
        scaler_path=None,
        clamp_min=None,
        clamp_max=None,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("The 'model' argument must be a PyTorch nn.Module.")

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task_type = task_type
        self.num_classes = num_classes
        self.model_config = model_config
        self.scaler_path = scaler_path
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        loss_fn_map = {
            'bce': torch.nn.BCEWithLogitsLoss(),
            'bpr': torch.nn.BCEWithLogitsLoss(),
            'mse': torch.nn.MSELoss(),
            'mae': torch.nn.L1Loss(),
            'ce':  torch.nn.CrossEntropyLoss(),
        }
        if model_config.get('loss_fn') not in loss_fn_map:
            raise ValueError(f"Unknown loss_fn '{model_config.get('loss_fn')}'. "
                             f"Choose from {list(loss_fn_map.keys())}.")
        self.loss_fn = loss_fn_map[model_config['loss_fn']]

        self.initial_state = deepcopy(model.state_dict())
        self._probe_cache: Dict[Tuple[int, int], _StaticBatchesLoader] = {}
        self._temp_model: Optional[nn.Module] = None


        self._calc = calculate_ftt_loss_and_metric

    # ----------------------------
    # Utilities
    # ----------------------------
    def _stage_loader(self, stage: str) -> DataLoader:
        if stage == "train":
            return self.train_loader
        if stage == "val":
            return self.val_loader
        if stage == "test":
            return self.test_loader
        raise ValueError(f"Invalid stage: {stage}. Must be 'train', 'val', or 'test'.")

    def _params_list(self, model: Optional[nn.Module] = None):
        m = self.model if model is None else model
        return [p for p in m.parameters() if p.requires_grad]

    def _create_random_directions(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Filter-wise normalized random directions; d2 ⟂ d1 per-parameter tensor."""
        params = self._params_list(self.model)

        d1 = []
        for p in params:
            r = torch.randn_like(p, device=self.device)
            pn = torch.norm(p.data, p='fro')
            rn = torch.norm(r, p='fro')
            if rn > 1e-8:
                r.mul_(pn / rn)
            d1.append(r)

        d2 = []
        for i, p in enumerate(params):
            r = torch.randn_like(p, device=self.device)
            d1i = d1[i]
            dot = torch.sum(d1i * r)
            n2 = torch.sum(d1i * d1i)
            r.sub_((dot / (n2 + 1e-12)) * d1i)
            pn = torch.norm(p.data, p='fro')
            rn = torch.norm(r, p='fro')
            if rn > 1e-8:
                r.mul_(pn / rn)
            d2.append(r)

        return d1, d2

    @torch.no_grad()
    def _precompute_flat_base_and_dirs(self):
        """Flatten base params and two directions for fast updates."""
        params = self._params_list(self.model)
        d1_t, d2_t = self._create_random_directions()

        base_vec = parameters_to_vector(params).detach()
        dir1_vec = parameters_to_vector(d1_t).detach()
        dir2_vec = parameters_to_vector(d2_t).detach()

        temp_model = deepcopy(self.model).to(self.device).eval()
        temp_params = self._params_list(temp_model)
        return temp_model, temp_params, base_vec, dir1_vec, dir2_vec

    @torch.no_grad()
    def _load_alpha_beta(self, temp_params, base_vec, dir1_vec, dir2_vec, alpha: float, beta: float):
        vec = base_vec.add(alpha, dir1_vec).add(beta, dir2_vec)
        vector_to_parameters(vec, temp_params)

    def _make_probe_loader(self, loader: DataLoader, *, max_batches: int = 8) -> _StaticBatchesLoader:
        """Take the first k batches to form a small, deterministic eval set."""
        key = (id(loader), int(max_batches))
        if key in self._probe_cache:
            return self._probe_cache[key]
        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i + 1 >= max_batches:
                break
        if not batches:
            raise RuntimeError("Probe loader: no batches collected (loader empty?).")
        static_loader = _StaticBatchesLoader(batches)
        self._probe_cache[key] = static_loader
        return static_loader

    # ----------------------------
    # Single (alpha, beta) evaluation using the provided metric fn
    # ----------------------------
    @torch.no_grad()
    def _eval_loss_at(self, temp_params, base_vec, d1_vec, d2_vec,
                      alpha: float, beta: float, loader_like, stage: str):
        self._load_alpha_beta(temp_params, base_vec, d1_vec, d2_vec, alpha, beta)
        loss, metric = self._calc(
            loader_like, self._temp_model, self.device,
            self.task_type, self.num_classes, self.loss_fn,
            self.scaler_path, self.clamp_min, self.clamp_max
        )
        return float(loss), float(metric)

    # ----------------------------
    # Metrics-only mode: O(9 + M*K) forward passes
    # ----------------------------
    def compute_metrics_only(
        self,
        stage: str = "val",
        epsilons: Tuple[float, ...] = (0.01, 0.02, 0.05),
        ray_angles_deg: Tuple[float, ...] = (0, 30, 60, 90, 120, 150),
        radii: np.ndarray = np.linspace(0.0, 1.0, 21),
        probe_batches: int = 8,
        use_amp: bool = True,
        compile_model: bool = True,
    ) -> Dict[str, float]:
        """
        Compute quantitative metrics (sharpness, anisotropy, barriers, ray widths)
        without building a full grid.
        """
        base_loader = self._stage_loader(stage)
        probe_loader = self._make_probe_loader(base_loader, max_batches=probe_batches)

        temp_model, temp_params, base_vec, d1_vec, d2_vec = self._precompute_flat_base_and_dirs()
        if compile_model:
            try:
                temp_model = torch.compile(temp_model, mode="reduce-overhead")
            except Exception:
                pass
        self._temp_model = temp_model

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_amp and torch.cuda.is_available()) else contextlib.nullcontext()

        with amp_ctx, torch.no_grad():
            # 3x3 finite differences around 0 for Hessian
            h = 1.0 / 30.0
            f00, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, 0.0, 0.0, probe_loader, stage)
            fpp, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, +h, 0.0, probe_loader, stage)
            fpm, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, -h, 0.0, probe_loader, stage)
            fp2, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, 0.0, +h, probe_loader, stage)
            fm2, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, 0.0, -h, probe_loader, stage)
            fppmm, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, +h, +h, probe_loader, stage)
            fpmmp, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, +h, -h, probe_loader, stage)
            fmpmp, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, -h, +h, probe_loader, stage)
            fmmmm, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, -h, -h, probe_loader, stage)

            faa = (fpp - 2*f00 + fpm) / (h**2)
            fbb = (fp2 - 2*f00 + fm2) / (h**2)
            fab = (fppmm - fpmmp - fmpmp + fmmmm) / (4*h*h)

            H = np.array([[faa, fab], [fab, fbb]], dtype=np.float64)
            evals, _ = np.linalg.eigh(H)
            lam_min, lam_max = float(evals[0]), float(evals[1])
            condH = abs(lam_max) / (abs(lam_min) + 1e-12)
            anisotropy = abs(faa - fbb) / (abs(faa) + abs(fbb) + 1e-12)
            froH = float(np.sqrt(faa*faa + fbb*fbb + 2.0*fab*fab))

            # multi-ray scan
            angles = np.deg2rad(np.array(ray_angles_deg, dtype=np.float64))
            barriers = []
            widths_eps = {eps: [] for eps in epsilons}
            for theta in angles:
                ca, cb = math.cos(theta), math.sin(theta)
                losses = []
                for r in radii:
                    a, b = float(r*ca), float(r*cb)
                    fl, _ = self._eval_loss_at(temp_params, base_vec, d1_vec, d2_vec, a, b, probe_loader, stage)
                    losses.append(fl)
                losses = np.asarray(losses, dtype=np.float64)
                barriers.append(float(np.max(losses) - f00))
                for eps in epsilons:
                    thr = f00 * (1.0 + eps)
                    ok = np.where(losses <= thr)[0]
                    rad = float(radii[ok[-1]]) if len(ok) else 0.0
                    widths_eps[eps].append(rad)

        metrics = {
            "loss_center": float(f00),
            "sharpness_lambda_max": float(lam_max),
            "curvature_lambda_min": float(lam_min),
            "traceH": float(faa + fbb),
            "froH": froH,
            "condH": float(condH),
            "anisotropy": float(anisotropy),
            "barrier_ray_mean": float(np.mean(barriers)),
            "barrier_ray_max": float(np.max(barriers)),
        }
        for eps, vals in widths_eps.items():
            metrics[f"ray_width_mean@{int(eps*100)}%"] = float(np.mean(vals))
            metrics[f"ray_width_min@{int(eps*100)}%"]  = float(np.min(vals))
            metrics[f"ray_width_max@{int(eps*100)}%"]  = float(np.max(vals))
        return metrics

    # ----------------------------
    # Accelerated full-grid plot (O(G^2), but fast inner loop + probe)
    # ----------------------------
    def plot(
        self,
        grid_points: int = 25,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = (-1.0, 1.0),
        stage: str = "val",
        figure_name: str = "ftt_loss_landscape.png",
        probe_batches: int = 8,
        use_amp: bool = False,
        compile_model: bool = False,
    ):
        """
        Compute a 2-D grid of losses and save a contour plot.
        Returns (loss_grid, alphas, betas, center_metrics_dict).
        """
        base_loader = self._stage_loader(stage)
        probe_loader = self._make_probe_loader(base_loader, max_batches=probe_batches)

        temp_model, temp_params, base_vec, dir1_vec, dir2_vec = self._precompute_flat_base_and_dirs()
        if compile_model:
            try:
                temp_model = torch.compile(temp_model, mode="reduce-overhead")
            except Exception:
                pass
        self._temp_model = temp_model

        alphas = np.linspace(x_range[0], x_range[1], grid_points)
        betas  = np.linspace(y_range[0], y_range[1], grid_points)
        loss_grid = np.empty((grid_points, grid_points), dtype=np.float32)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_amp and torch.cuda.is_available()) else contextlib.nullcontext()

        with amp_ctx, torch.no_grad():
            total = tqdm(total=grid_points * grid_points, desc="Calculating FTT loss")
            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    self._load_alpha_beta(temp_params, base_vec, dir1_vec, dir2_vec, float(a), float(b))
                    loss, _ = self._calc(
                        probe_loader, temp_model, self.device,
                        self.task_type, self.num_classes, self.loss_fn,
                        self.scaler_path, self.clamp_min, self.clamp_max
                    )
                    loss_grid[j, i] = float(loss)
                    total.update(1)

        center_metrics = self.compute_landscape_metrics(loss_grid, alphas, betas, epsilons=(0.01, 0.02, 0.05))

        # Plot
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8))
        min_loss = float(np.min(loss_grid))
        max_loss = float(np.max(loss_grid))
        eps = 1e-8
        lo = max(min_loss + eps, eps)
        hi = max(max_loss, lo * 1.05)
        levels = np.logspace(np.log10(lo), np.log10(hi), num=20)
        cs = ax.contour(alphas, betas, loss_grid, levels=levels, cmap='viridis')
        ax.clabel(cs, inline=True, fontsize=9, fmt='%1.3f')
        ax.set_title('FT-Transformer Loss Landscape', fontsize=18, fontweight='bold')
        ax.set_xlabel(r'Direction 1 ($\alpha$)', fontsize=14)
        ax.set_ylabel(r'Direction 2 ($\beta$)', fontsize=14)
        plt.savefig(figure_name, format=figure_name.split(".")[-1])
        if not figure_name.endswith(".pdf"):
            plt.savefig(figure_name.rsplit(".", 1)[0] + ".pdf", format='pdf')
        plt.close(fig)

        return loss_grid, alphas, betas, center_metrics

    # ----------------------------
    # Grid-derived metrics (if you already have the grid)
    # ----------------------------
    @staticmethod
    def _nearest_index(vec: np.ndarray, val: float) -> int:
        return int(np.abs(vec - val).argmin())

    @staticmethod
    def _contiguous_span_around_center(mask: np.ndarray, center: int) -> Tuple[int, int]:
        L = R = center
        n = len(mask)
        while L-1 >= 0 and mask[L-1]:
            L -= 1
        while R+1 < n and mask[R+1]:
            R += 1
        return L, R

    def _finite_difference_hessian(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> dict:
        ia0 = self._nearest_index(alphas, 0.0)
        ib0 = self._nearest_index(betas, 0.0)
        if not (1 <= ia0 < len(alphas)-1 and 1 <= ib0 < len(betas)-1):
            raise ValueError("Grid must include ±1 steps around (0,0). Adjust range or grid_points.")
        f00 = float(loss_grid[ib0, ia0])
        da = float(alphas[1] - alphas[0])
        db = float(betas[1] - betas[0])

        faa = (loss_grid[ib0, ia0+1] - 2.0*f00 + loss_grid[ib0, ia0-1]) / (da**2)
        fbb = (loss_grid[ib0+1, ia0] - 2.0*f00 + loss_grid[ib0-1, ia0]) / (db**2)
        fab = (loss_grid[ib0+1, ia0+1] - loss_grid[ib0+1, ia0-1]
               - loss_grid[ib0-1, ia0+1] + loss_grid[ib0-1, ia0-1]) / (4.0 * da * db)

        H = np.array([[faa, fab], [fab, fbb]], dtype=np.float64)
        evals, evecs = np.linalg.eigh(H)
        lam_min, lam_max = float(evals[0]), float(evals[1])
        cond = abs(lam_max) / (abs(lam_min) + 1e-12)
        anisotropy = abs(faa - fbb) / (abs(faa) + abs(fbb) + 1e-12)

        return {
            "f00": f00,
            "H": H,
            "eigvals": (lam_min, lam_max),
            "traceH": float(np.trace(H)),
            "froH": float(np.sqrt(faa*faa + fbb*fbb + 2.0*fab*fab)),
            "condH": float(cond),
            "anisotropy": float(anisotropy),
            "principal_dir": evecs[:, 1].tolist(),
            "ia0": ia0, "ib0": ib0, "da": da, "db": db
        }

    def _flatness_and_widths(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                             f00: float, ia0: int, ib0: int, eps_rel: float = 0.05) -> dict:
        rel = (loss_grid - f00) / max(f00, 1e-12)
        da = float(alphas[1] - alphas[0])
        db = float(betas[1] - betas[0])
        area_cell = da * db
        area_total = (alphas[-1] - alphas[0]) * (betas[-1] - betas[0])

        area_eps = float(np.sum(rel <= eps_rel) * area_cell)
        area_ratio = float(area_eps / (area_total + 1e-12))

        row_mask = (rel[ib0, :] <= eps_rel)
        La, Ra = self._contiguous_span_around_center(row_mask, ia0)
        width_alpha = float(alphas[Ra] - alphas[La])

        col_mask = (rel[:, ia0] <= eps_rel)
        Lb, Rb = self._contiguous_span_around_center(col_mask, ib0)
        width_beta = float(betas[Rb] - betas[Lb])

        # main diagonal
        kmax1 = int(min(ia0, ib0))
        kmax2 = int(min(len(alphas)-ia0-1, len(betas)-ib0-1))
        diag_main_mask = []
        for k in range(-kmax1, kmax2+1):
            diag_main_mask.append(rel[ib0 + k, ia0 + k] <= eps_rel)
        center_k = kmax1
        Lk, Rk = self._contiguous_span_around_center(np.array(diag_main_mask, dtype=bool), center_k)
        width_diag_main = float(np.sqrt(
            (alphas[ia0+(Rk-center_k)] - alphas[ia0-(center_k-Lk)])**2 +
            (betas[ib0+(Rk-center_k)]  - betas[ib0-(center_k-Lk)])**2
        ))

        # anti-diagonal
        kmax1b = int(min(ia0, len(betas)-ib0-1))
        kmax2b = int(min(len(alphas)-ia0-1, ib0))
        diag_anti_mask = []
        for k in range(-kmax1b, kmax2b+1):
            diag_anti_mask.append(rel[ib0 - k, ia0 + k] <= eps_rel)
        center_k2 = kmax1b
        Lk2, Rk2 = self._contiguous_span_around_center(np.array(diag_anti_mask, dtype=bool), center_k2)
        width_diag_anti = float(np.sqrt(
            (alphas[ia0+(Rk2-center_k2)] - alphas[ia0-(center_k2-Lk2)])**2 +
            (betas[ib0-(Rk2-center_k2)]  - betas[ib0+(center_k2-Lk2)])**2
        ))

        robust_radius_eps = float(0.5 * min(width_alpha, width_beta))

        return {
            f"area@{int(eps_rel*100)}%": area_eps,
            f"area_ratio@{int(eps_rel*100)}%": area_ratio,
            f"width_alpha@{int(eps_rel*100)}%": width_alpha,
            f"width_beta@{int(eps_rel*100)}%": width_beta,
            f"width_diag_main@{int(eps_rel*100)}%": width_diag_main,
            f"width_diag_anti@{int(eps_rel*100)}%": width_diag_anti,
            f"robust_radius@{int(eps_rel*100)}%": robust_radius_eps,
        }

    def _global_sensitivity_and_ruggedness(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> dict:
        da = float(alphas[1] - alphas[0])
        db = float(betas[1] - betas[0])
        dfdβ, dfdα = np.gradient(loss_grid, db, da, edge_order=2)
        grad_norm = np.sqrt(dfdα**2 + dfdβ**2)
        lipschitz_max = float(np.nanmax(grad_norm))
        lipschitz_p95 = float(np.nanpercentile(grad_norm, 95))
        lipschitz_mean = float(np.nanmean(grad_norm))
        tv = float(np.mean(np.abs(dfdα)) + np.mean(np.abs(dfdβ)))

        lap = (np.roll(loss_grid, 1, axis=0) + np.roll(loss_grid, -1, axis=0) +
               np.roll(loss_grid, 1, axis=1) + np.roll(loss_grid, -1, axis=1) -
               4.0 * loss_grid) / ((da**2 + db**2) / 2.0)
        rugged_rms = float(np.sqrt(np.mean(lap**2)))

        return {
            "lipschitz_max": lipschitz_max,
            "lipschitz_p95": lipschitz_p95,
            "lipschitz_mean": lipschitz_mean,
            "total_variation": tv,
            "rugged_laplacian_rms": rugged_rms
        }

    def _barrier_indices(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                         ia0: int, ib0: int, f00: float) -> dict:
        row = loss_grid[ib0, :]
        col = loss_grid[:, ia0]
        B_alpha_pos = float(np.max(row[ia0:]) - f00)
        B_alpha_neg = float(np.max(row[:ia0+1]) - f00)
        B_beta_pos  = float(np.max(col[ib0:]) - f00)
        B_beta_neg  = float(np.max(col[:ib0+1]) - f00)

        kmax1 = int(min(ia0, ib0))
        kmax2 = int(min(len(alphas)-ia0-1, len(betas)-ib0-1))
        diag_main_vals_pos = [loss_grid[ib0 + k, ia0 + k] for k in range(0, kmax2+1)]
        diag_main_vals_neg = [loss_grid[ib0 - k, ia0 - k] for k in range(0, kmax1+1)]
        B_diag_main_pos = float(np.max(diag_main_vals_pos) - f00)
        B_diag_main_neg = float(np.max(diag_main_vals_neg) - f00)

        kmax1b = int(min(ia0, len(betas)-ib0-1))
        kmax2b = int(min(len(alphas)-ia0-1, ib0))
        diag_anti_vals_pos = [loss_grid[ib0 - k, ia0 + k] for k in range(0, kmax2b+1)]
        diag_anti_vals_neg = [loss_grid[ib0 + k, ia0 - k] for k in range(0, kmax1b+1)]
        B_diag_anti_pos = float(np.max(diag_anti_vals_pos) - f00)
        B_diag_anti_neg = float(np.max(diag_anti_vals_neg) - f00)

        return {
            "barrier_alpha_pos": B_alpha_pos,
            "barrier_alpha_neg": B_alpha_neg,
            "barrier_beta_pos":  B_beta_pos,
            "barrier_beta_neg":  B_beta_neg,
            "barrier_diag_main_pos": B_diag_main_pos,
            "barrier_diag_main_neg": B_diag_main_neg,
            "barrier_diag_anti_pos": B_diag_anti_pos,
            "barrier_diag_anti_neg": B_diag_anti_neg,
            "barrier_axis_mean": float(np.mean([B_alpha_pos, B_alpha_neg, B_beta_pos, B_beta_neg])),
            "barrier_diag_mean": float(np.mean([B_diag_main_pos, B_diag_main_neg, B_diag_anti_pos, B_diag_anti_neg])),
            "barrier_overall_max": float(np.max([
                B_alpha_pos, B_alpha_neg, B_beta_pos, B_beta_neg,
                B_diag_main_pos, B_diag_main_neg, B_diag_anti_pos, B_diag_anti_neg
            ]))
        }

    def _non_quadraticity_rmse(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                               ia0: int, ib0: int, f00: float, patch_radius: int = 2) -> float:
        i0, j0 = ia0, ib0
        iL = max(0, i0 - patch_radius); iR = min(len(alphas)-1, i0 + patch_radius)
        jL = max(0, j0 - patch_radius); jR = min(len(betas)-1,  j0 + patch_radius)

        A, y = [], []
        for j in range(jL, jR+1):
            for i in range(iL, iR+1):
                a = float(alphas[i]); b = float(betas[j])
                A.append([1.0, a, b, 0.5*a*a, a*b, 0.5*b*b])
                y.append(float(loss_grid[j, i]))
        A = np.asarray(A, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ theta
        rmse = float(np.sqrt(np.mean((yhat - y)**2)))
        return rmse / max(f00, 1e-12)

    def compute_landscape_metrics(self, loss_grid: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                                  epsilons: Tuple[float, ...] = (0.01, 0.02, 0.05)) -> dict:
        """Quantitative metrics from an already-computed grid."""
        hess = self._finite_difference_hessian(loss_grid, alphas, betas)
        ia0, ib0, f00 = hess["ia0"], hess["ib0"], hess["f00"]

        glb = {
            "loss_min": float(np.min(loss_grid)),
            "loss_max": float(np.max(loss_grid)),
            "loss_range": float(np.max(loss_grid) - np.min(loss_grid)),
            "loss_center": float(f00),
        }
        sens = self._global_sensitivity_and_ruggedness(loss_grid, alphas, betas)
        barriers = self._barrier_indices(loss_grid, alphas, betas, ia0, ib0, f00)

        flat = {}
        for eps in epsilons:
            flat.update(self._flatness_and_widths(loss_grid, alphas, betas, f00, ia0, ib0, eps_rel=eps))

        nonquad = {"nonquadraticity_rmse@5x5": self._non_quadraticity_rmse(loss_grid, alphas, betas, ia0, ib0, f00)}

        lam_min, lam_max = hess["eigvals"]
        curv = {
            "sharpness_lambda_max": float(lam_max),
            "curvature_lambda_min": float(lam_min),
            "traceH": hess["traceH"],
            "froH": hess["froH"],
            "condH": hess["condH"],
            "anisotropy": hess["anisotropy"],
        }

        out = {}
        out.update(glb)
        out.update(curv)
        out.update(flat)
        out.update(sens)
        out.update(barriers)
        out.update(nonquad)
        return out