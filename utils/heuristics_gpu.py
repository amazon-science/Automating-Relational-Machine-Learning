import math
import numpy as np
import pandas as pd
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import degree
from tqdm import tqdm
from typing import List, Tuple, Any, Optional, Literal, Dict
from dataclasses import dataclass
from loguru import logger


def _get_field(tbl, key, default=None):
    """Safely read attribute or dict-style field."""
    if hasattr(tbl, key):
        return getattr(tbl, key)
    if isinstance(tbl, dict):
        return tbl.get(key, default)
    return default

def _pick_fk_for_entity(labels_tbl_meta, task_entity_table: str, override_fk: str | None = None):
    """Find the FK column in the labels table that maps to the target entity table."""
    if override_fk:
        return override_fk
    fmap = _get_field(labels_tbl_meta, "fkey_col_to_pkey_table", {}) or {}
    candidates = [fk for fk, tgt in fmap.items() if str(tgt) == str(task_entity_table)]
    if not candidates:
        raise ValueError(
            f"No FK in labels table maps to '{task_entity_table}'. "
            f"Available mapping: {fmap}"
        )
    if len(candidates) > 1:
        # If multiple, pick the first; you can pass override_fk to disambiguate.
        return candidates[0]
    return candidates[0]

# ------------------ soft label builder ------------------

import math
import numpy as np
import pandas as pd
import torch

def build_label_soft_from_event_table_df(
    labels_df: pd.DataFrame,
    *,
    fkey_col: str,
    node_pkeys: np.ndarray,            # aligned with graph node index order
    label_col: str,
    time_col: str | None = None,
    num_classes: int | None = None,
    half_life_days: float | None = None,
    end_time: pd.Timestamp | None = None,
    fill_unlabeled: str = "uniform",   # 'uniform' | 'prop' | 'zeros'
    task_type: str = "classification",
    is_autocomplete: bool = False,
    db_obj=None,                       # needed for autocomplete mode
    task_entity: str | None = None,    # needed for autocomplete mode
    graph_num_nodes: int | None = None,  # graph node count for autocomplete mode
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turn a (entity_id, label[, time]) event table into [N, C] soft labels aligned to node indices.
    Fast, vectorized version.

    Autocomplete:
      - labeled_mask is True for all nodes
      - P rows for nodes not present in labels_df set to -1 (all classes or the single reg value)
    """
    # --- Select only needed cols
    cols = [fkey_col, label_col]
    if time_col:
        cols.append(time_col)
    df = labels_df[cols].copy()

    # --- Time weighting
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if end_time is None:
            end_time = df[time_col].max()
    if time_col and half_life_days:
        lam = math.log(2.0) / float(half_life_days)
        dt_days = (end_time - df[time_col]).dt.total_seconds() / 86400.0
        w = np.exp(-lam * dt_days.to_numpy(dtype=np.float64))
    else:
        w = np.ones(len(df), dtype=np.float64)

    # --- Map entity ids to node indices (vectorized)
    #     idx = position in node_pkeys, or -1 if not present
    node_index = pd.Index(node_pkeys)
    ids = df[fkey_col].to_numpy()
    idx = node_index.get_indexer(ids)  # int64, -1 for unknown

    if is_autocomplete:
        # For autocomplete tasks, we need to align labels with graph node indices
        # Use graph_num_nodes if provided (correct graph size), otherwise fall back to node_pkeys length
        if graph_num_nodes is not None:
            N = graph_num_nodes
        else:
            N = len(node_pkeys)

        # Create mapping from primary key to graph node index using node_pkeys
        # (node_index already created above at line 88)

        # Map labels_df foreign keys to graph node indices
        task_ids = df[fkey_col].to_numpy()
        task_labels = df[label_col].to_numpy()

        # Get indices in the graph for each task ID (-1 if not found)
        # Use the idx already computed above which maps fkey_col values to node_pkeys indices
        graph_indices = idx  # already computed as node_index.get_indexer(ids)

        # Valid entries are those that exist in the graph
        valid_mask = graph_indices >= 0
        valid_graph_indices = graph_indices[valid_mask]
        valid_labels = task_labels[valid_mask]

        if task_type == "classification":
            # Infer num_classes from labels
            num_classes = int(np.nanmax(valid_labels)) + 1
            C = num_classes

            # Initialize P matrix with zeros (unlabeled nodes)
            P_np = np.full((N, C), 0.0, dtype=np.float64)

            # Filter valid labels and create one-hot encoding
            valid_label_mask = (~np.isnan(valid_labels)) & (valid_labels >= 0) & (valid_labels < C)
            valid_indices = valid_graph_indices[valid_label_mask]
            valid_labels_clean = valid_labels[valid_label_mask].astype(int)

            # Use advanced indexing for one-hot encoding
            P_np[valid_indices, valid_labels_clean] = 1.0
        else:  # regression
            # Initialize P matrix with -1 (indicating unlabeled nodes)
            P_np = np.full((N, 1), -1.0, dtype=np.float64)

            # Filter valid labels
            valid_label_mask = ~np.isnan(valid_labels)
            valid_indices = valid_graph_indices[valid_label_mask]
            valid_labels_clean = valid_labels[valid_label_mask]

            # Use advanced indexing for assignment
            P_np[valid_indices, 0] = valid_labels_clean

        labeled_mask_np = np.zeros(N, dtype=bool)
        labeled_mask_np[valid_indices] = True
    else:
        # --- Normalize/validate labels per task
        if task_type == "classification":
            # infer num_classes if needed
            if num_classes is None:
                u = df[label_col].dropna().astype(np.int64).unique()
                if set(u.tolist()) <= {0, 1}:
                    num_classes = 2
                else:
                    num_classes = int(np.nanmax(df[label_col].to_numpy(dtype=np.float64))) + 1
            C = int(num_classes)

            # valid labels (integer in [0, C))
            yi_raw = df[label_col].to_numpy()
            # Coerce to int safely: mask non-numeric first
            yi_num = pd.to_numeric(yi_raw, errors="coerce")
            valid_lab = ~np.isnan(yi_num)
            yi = yi_num.astype(np.int64, copy=False)
            valid = (idx >= 0) & valid_lab & (yi >= 0) & (yi < C) & (w > 0)

            idx_v = idx[valid]
            yi_v  = yi[valid]
            w_v   = w[valid]

            N = int(len(node_pkeys))

            # --- Prepare baseline fill
            if fill_unlabeled == "uniform":
                P_np = np.full((N, C), 1.0 / C, dtype=np.float64)
            elif fill_unlabeled == "prop":
                # global class proportions (weighted)
                gcounts = np.bincount(yi_v, weights=w_v, minlength=C).astype(np.float64, copy=False)
                if gcounts.sum() > 0:
                    P_np = np.broadcast_to((gcounts / gcounts.sum())[None, :], (N, C)).copy()
                else:
                    P_np = np.full((N, C), 1.0 / C, dtype=np.float64)
            elif fill_unlabeled == "zeros":
                P_np = np.zeros((N, C), dtype=np.float64)
            else:
                raise ValueError("fill_unlabeled must be 'uniform' | 'prop' | 'zeros'")

            # --- Accumulate weighted class counts per node (very fast; all in C)
            #     We add into P_np_counts and then row-normalize for nodes with events.
            P_counts = np.zeros_like(P_np)   # shape [N, C]
            # np.add.at on 2D via tuple indexing (no Python loops)
            np.add.at(P_counts, (idx_v, yi_v), w_v)

            # totals per node (sum of weights)
            totals = np.bincount(idx_v, weights=w_v, minlength=N).astype(np.float64, copy=False)
            rows = totals > 0

            # normalize rows with observations
            if rows.any():
                P_np[rows, :] = P_counts[rows, :] / totals[rows, None]

            labeled_mask_np = rows

        elif task_type == "regression":
            # numeric labels
            y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=np.float64)
            valid = (idx >= 0) & (~np.isnan(y)) & (w > 0)

            idx_v = idx[valid]
            y_v   = y[valid]
            w_v   = w[valid]

            N = int(len(node_pkeys))

            # global weighted mean for fills (if needed)
            if w_v.size > 0 and w_v.sum() > 0:
                gmean = np.sum(y_v * w_v) / np.sum(w_v)
            else:
                gmean = 0.0

            if fill_unlabeled in ("uniform", "prop"):
                P_np = np.full((N, 1), gmean, dtype=np.float64)
            elif fill_unlabeled == "zeros":
                P_np = np.zeros((N, 1), dtype=np.float64)
            else:
                raise ValueError("fill_unlabeled must be 'uniform' | 'prop' | 'zeros'")

            # weighted sums and totals per node
            wsum = np.bincount(idx_v, weights=(y_v * w_v), minlength=N).astype(np.float64, copy=False)
            totals = np.bincount(idx_v, weights=w_v, minlength=N).astype(np.float64, copy=False)
            rows = totals > 0
            if rows.any():
                P_np[rows, 0] = wsum[rows] / totals[rows]

            labeled_mask_np = rows

        else:
            raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type}")

    # --- Torch tensors (float64 to match your downstream expectations)
    P = torch.from_numpy(P_np).to(torch.float64)
    labeled_mask = torch.from_numpy(labeled_mask_np)

    if torch.cuda.is_available():
        P = P.cuda()
        labeled_mask = labeled_mask.cuda()

    return P, labeled_mask


# ------------------ soft-label homophily (2-hop, chunked) ------------------

# --- RFF for 1D Gaussian kernel k(y,y') = exp(-(y-y')^2 / (2 sigma^2)) ---


def rff_embed_scalar_1d(y: torch.Tensor, dim_half: int, sigma: float) -> torch.Tensor:
    """
    Random Fourier Features for 1D Gaussian kernel k(a,b)=exp(-(a-b)^2 / (2 sigma^2)).
    Returns [N, 2*dim_half] with cos/sin blocks, scaled so dot ~ kernel.
    """
    if sigma <= 0 or not math.isfinite(sigma):
        sigma = float(y.std(unbiased=True).item() or 1.0)
    # w ~ Normal(0, 1/sigma^2)
    w = torch.randn(dim_half, dtype=torch.float64, device=y.device) / max(sigma, 1e-12)
    xw = y.to(torch.float64)[:, None] * w[None, :]
    # sqrt(2/H) normalization so E[phi(x)·phi(y)] ≈ k(x,y)
    scale = math.sqrt(2.0 / dim_half)
    return scale * torch.cat([torch.cos(xw), torch.sin(xw)], dim=1)  # [N, 2H]


def _agg_homophily_soft_cls(
    DN: torch.Tensor,          # [N, C] neighbor soft-label mean
    P: torch.Tensor,           # [N, C] node soft labels
    used: torch.Tensor,        # [N] bool mask of nodes to score
    proportions: torch.Tensor  # [C] class prior (on labeled nodes)
) -> float:
    """
    Aggregation homophily (classification, soft labels).
    Score per node u: sim_same = <P_u, DN_u>; baseline = <proportions, DN_u>.
    Return fraction where sim_same >= baseline, averaged over `used`.
    """
    if used.dtype != torch.bool:
        used = used.to(torch.bool)
    if not used.any():
        return float("nan")
    U = used.nonzero(as_tuple=False).squeeze(1)
    Pu = P[U]        # [M, C]
    DNu = DN[U]      # [M, C]
    sim_same = (Pu * DNu).sum(dim=1)                  # [M]
    base = (proportions.to(DNu.dtype)[None, :] * DNu).sum(dim=1)  # [M]
    return float((sim_same >= base).float().mean().item())


# ----- main -----

@torch.no_grad()
def calcHomophily_soft(
    data,
    task_entity: str,
    label_soft: torch.Tensor,      # [N, C] (classification) or [N,1] (regression)
    labeled_mask: torch.Tensor,    # [N] bool (True = labeled)
    chunk_size: int = 500,
    task_type: Literal["classification", "regression"] = "classification",
    max_group: int = 300,
    return_per_sample: bool = False,
    regression_mode: Literal["gaussian", "fake_cls", "corr"] = "gaussian",
    regression_bins: Optional[List[float]] = None,
    empty_as_0: bool = False,
):
    """
    Soft-label homophily on 2-hop metapaths that start and end at `task_entity`, computed in chunks.

    Returns (lists aligned to metapaths):
        h_edges         : expected edge similarity (soft)
        h_insensitives  : class-insensitive variant
        h_adjs          : degree-adjusted variant
        h_aggs          : aggregation homophily (LOO-approx; cls+reg supported)
        sparsities      : number of labeled nodes that have at least one valid neighbor along this metapath
        metapath_degrees: list of (metapath, deg_u, deg_v) where
                          deg_u: [N] labeled-neighbor counts per source u
                          deg_v: [N] valid-edge counts per target v
        per_sample_homophily: (optional, if return_per_sample=True) dict mapping metapath to [N] tensor

    Args:
        regression_mode: For regression tasks choose between original Gaussian kernel
            ("gaussian"), threshold-based fake classification ("fake_cls"), or correlation-based
            similarity ("corr").
        regression_bins: Thresholds used when `regression_mode="fake_cls"` to bin the target.
        empty_as_0: If True, incorporate labeled nodes without neighbors by treating their homophily
            contribution as zero when averaging metrics (including the adjusted variant). Defaults False.
    """
    # --- enumerate only 2-hop metapaths that RETURN to task_entity ---
    metas: List[List[Tuple[str, str, str]]] = []

    def _ext(edge_types, cur, mp, hop):
        if hop < 1:
            if cur == task_entity:
                metas.append(mp)
            return
        for et in edge_types:
            s, _, t = et
            if s == cur:
                _ext(edge_types, t, mp + [et], hop - 1)

    _ext(data.edge_types, task_entity, [], 2)  # only collect ... -> task_entity (2 hops)

    device = label_soft.device
    labeled_mask = labeled_mask.to(torch.bool).to(device)
    total_labeled_nodes = int(labeled_mask.sum().item())

    # --- prepare labels / embeddings ---
    if task_type == "classification":
        P = label_soft.to(device=device, dtype=torch.float64)  # [N, C]
        N, C = P.shape
        # class prior over labeled nodes only (ignore -1 rows if caller uses that convention)
        if (P < 0).any():
            proportions = torch.nanmean(torch.where(P >= 0, P, torch.nan), dim=0, keepdim=True)
        else:
            proportions = P[labeled_mask].mean(dim=0, keepdim=True)
        proportions = torch.nan_to_num(proportions, nan=0.0)
        if float(proportions.sum()) <= 0:
            proportions = torch.full_like(proportions, 1.0 / C)
    else:
        # regression
        P = label_soft.to(device=device, dtype=torch.float64)  # [N, 1]
        N = P.shape[0]
        if P.shape[1] != 1:
            raise ValueError(f"regression expects label_soft [N,1], got {tuple(P.shape)}")
        y_all = P.squeeze(1)  # [N], float64
        P_multi: Optional[torch.Tensor] = None
        reg_bins_tensor: Optional[torch.Tensor] = None
        p_cols: Optional[torch.Tensor] = None
        baseline_reg: Optional[float] = None
        fake_cls_dim = 0

        # corr mode uses globally normalized labels for stability
        valid_y_mask = labeled_mask & (y_all != -1.0)
        if valid_y_mask.any():
            mu_global = y_all[valid_y_mask].mean()
            sigma_global = y_all[valid_y_mask].std(unbiased=True)
            if (not torch.isfinite(sigma_global)) or (float(sigma_global) <= 0.0):
                sigma_global = torch.tensor(1.0, device=device, dtype=torch.float64)
        else:
            mu_global = torch.tensor(0.0, device=device, dtype=torch.float64)
            sigma_global = torch.tensor(1.0, device=device, dtype=torch.float64)
        y_norm = y_all.clone()
        if valid_y_mask.any():
            y_norm[valid_y_mask] = (y_all[valid_y_mask] - mu_global) / sigma_global

        if regression_mode == "gaussian":
            # robust kernel width
            sigma_y = float(y_all[labeled_mask].std(unbiased=True).item() or 1.0)
            rff_dim_half = 32
            Z_all = rff_embed_scalar_1d(y_all, rff_dim_half, sigma_y)  # [N, 2H], float64
            var_y = float(y_all[labeled_mask].var(unbiased=True).item() or 0.0)
        elif regression_mode == "fake_cls":
            if not regression_bins:
                raise ValueError("regression_bins must be provided when regression_mode='fake_cls'")
            reg_bins_tensor = torch.as_tensor(
                sorted(regression_bins),
                dtype=torch.float64,
                device=device,
            )
            if reg_bins_tensor.numel() == 0:
                raise ValueError("regression_bins must be a non-empty list when using fake_cls mode")

            fake_cls_dim = int(reg_bins_tensor.numel())
            y_col = y_all.unsqueeze(1)
            P_multi = (y_col >= reg_bins_tensor.unsqueeze(0)).to(torch.float64)  # [N, K]

            if labeled_mask.any():
                p_cols = P_multi[labeled_mask].mean(dim=0)
            else:
                p_cols = P_multi.mean(dim=0)

            baseline_reg = float(((p_cols ** 2 + (1.0 - p_cols) ** 2).mean()).item())
            # placeholders for gaussian-specific vars
            sigma_y = 0.0
            var_y = 0.0
            Z_all = None
        elif regression_mode == "corr":
            sigma_y = 0.0
            var_y = 1.0  # normalized labels have unit variance on labeled nodes
            Z_all = None
            baseline_reg = 0.0
        else:
            raise ValueError(f"Unknown regression_mode: {regression_mode}")

    h_edges, h_insensitives, h_adjs, h_aggs = [], [], [], []
    sparsities: List[int] = []
    metapath_degrees = []  # (mp, deg_u, deg_v)
    per_sample_homophily_dict = {}  # metapath -> [N] tensor of per-sample homophily

    for mp in tqdm(metas, desc="Metapaths"):
        # mp = [(task_entity, *, mid), (mid, *, task_entity)]
        try:
            s0, edge_name, m1 = mp[0]
            m2, edge_name2, t2 = mp[1]
            if s0 != task_entity or t2 != task_entity or m1 != m2:
                # we only support 2-hop cycles back to task_entity
                continue
            # if 'f2p' in edge_name:
            #     # pk-fk edges can't be used for homophily
            #     continue

            m = data.num_nodes_dict[s0]   # == N
            n = data.num_nodes_dict[m1]
            k = data.num_nodes_dict[t2]   # == N

            e1 = data[mp[0]].edge_index
            e2 = data[mp[1]].edge_index
            A1 = SparseTensor(row=e1[0], col=e1[1], sparse_sizes=(m, n)).to(device)
            A2 = SparseTensor(row=e2[0], col=e2[1], sparse_sizes=(n, k)).to(device)

            # Optional: thin A2 to only columns that are labeled on the target side
            # (saves compute under very sparse labels)
            r2, c2, _ = A2.coo()
            keep_c = labeled_mask[c2]
            if keep_c.any():
                A2 = SparseTensor(
                    row=r2[keep_c], col=c2[keep_c], sparse_sizes=(n, k)
                ).to(device)
            else:
                # no reachable labeled targets -> skip this metapath
                continue

            n_rows = A1.size(0)  # == N

            # --- edge-level accumulators ---
            total_same_exp = torch.zeros((), device=device, dtype=torch.float64)
            total_edges    = torch.zeros((), device=device, dtype=torch.float64)
            deg_v = torch.zeros(N, device=device, dtype=torch.float64)     # degree over target v

            sum_yu = sum_yv = sum_yu2 = sum_yv2 = sum_yuyv = M_edges = None
            if task_type == "regression" and regression_mode == "corr":
                sum_yu = torch.zeros((), device=device, dtype=torch.float64)
                sum_yv = torch.zeros((), device=device, dtype=torch.float64)
                sum_yu2 = torch.zeros((), device=device, dtype=torch.float64)
                sum_yv2 = torch.zeros((), device=device, dtype=torch.float64)
                sum_yuyv = torch.zeros((), device=device, dtype=torch.float64)
                M_edges = torch.zeros((), device=device, dtype=torch.float64)

            if task_type == "classification":
                numer_k = torch.zeros(C, device=device, dtype=torch.float64)
                denom_k = torch.zeros(C, device=device, dtype=torch.float64)

            # --- aggregation accumulators ---
            if task_type == "classification":
                neigh_sum_u = torch.zeros((N, C), device=device, dtype=torch.float64)
            else:
                if regression_mode == "gaussian":
                    Dz = Z_all.shape[1]
                    neigh_sum_u_z = torch.zeros((N, Dz), device=device, dtype=torch.float64)
                elif regression_mode == "fake_cls":
                    if P_multi is None:
                        raise RuntimeError("P_multi is required for regression_mode='fake_cls'")
                    neigh_sum_u_multi = torch.zeros((N, fake_cls_dim), device=device, dtype=torch.float64)
                elif regression_mode == "corr":
                    neigh_sum_u_z = None
                    neigh_sum_u_multi = None
                else:
                    raise RuntimeError(f"Unsupported regression_mode='{regression_mode}' for aggregation accumulators")

            deg_u = torch.zeros(N, device=device, dtype=torch.float64)     # labeled-neighbor counts per source u

            # ---- label-aware chunk selection (robust under sparse labels) ----
            L = torch.nonzero(labeled_mask, as_tuple=False).squeeze(1)  # global labeled rows (task_entity)
            if L.numel() == 0:
                # no labeled nodes at all for this metapath; skip safely
                continue

            total_groups = math.ceil(n_rows / chunk_size)
            groups_with_labels = torch.div(L, chunk_size, rounding_mode='floor').unique()
            if groups_with_labels.numel() == 0:
                continue

            if groups_with_labels.numel() > max_group:
                perm = torch.randperm(groups_with_labels.numel(), device=device)
                chosen_groups = groups_with_labels[perm[:max_group]]
            else:
                chosen_groups = groups_with_labels

            pbar = tqdm(chosen_groups.tolist(), desc="Metapath (label-aware chunks)")
            for g in pbar:
                start = int(g) * chunk_size
                end = min(start + chunk_size, n_rows)

                # 2-hop adjacency slice: rows in [start, end)
                R = A1[start:end] @ A2
                row, col, _ = R.coo()
                if row.numel() == 0:
                    continue

                u = row + start  # global task-entity source indices
                v = col          # global task-entity target indices

                # ----- edge-level metrics (both endpoints must be labeled) -----
                valid_e = labeled_mask[u] & labeled_mask[v]
                if valid_e.any():
                    if task_type == "classification":
                        Pu_valid = P[u][valid_e]  # [E, C]
                        Pv_valid = P[v][valid_e]  # [E, C]
                        if (u == v).all():
                            continue
                        s = (Pu_valid * Pv_valid).sum(dim=1)  # [E]
                        total_same_exp += s.sum()
                        total_edges    += s.new_tensor(s.numel())
                        numer_k        += (Pv_valid * s.unsqueeze(1)).sum(dim=0)
                        denom_k        += Pv_valid.sum(dim=0)
                        deg_v.index_add_(0, v[valid_e], torch.ones_like(s, dtype=deg_v.dtype))
                    else:
                        yu_all = y_all[u]
                        yv_all = y_all[v]
                        pm = (yu_all != -1.0) & (yv_all != -1.0)  # ignore test/unset labels if caller uses -1
                        ve = valid_e & pm
                        if ve.any():
                            if regression_mode == "gaussian":
                                diff2 = (yu_all - yv_all).pow(2)[ve]
                                s = torch.exp(-diff2 / (2.0 * (sigma_y ** 2)))
                                total_same_exp += s.sum()
                                total_edges    += s.new_tensor(s.numel())
                                deg_v.index_add_(0, v[ve], torch.ones_like(s, dtype=deg_v.dtype))
                            elif regression_mode == "fake_cls":
                                if reg_bins_tensor is None or reg_bins_tensor.numel() == 0:
                                    raise RuntimeError("regression_bins tensor is missing for fake classification mode")
                                yu_ve = yu_all[ve].unsqueeze(1)
                                yv_ve = yv_all[ve].unsqueeze(1)
                                low = torch.minimum(yu_ve, yv_ve)
                                high = torch.maximum(yu_ve, yv_ve)
                                T = reg_bins_tensor.view(1, -1)
                                disagree = (T >= low) & (T < high)
                                D = disagree.sum(dim=1, dtype=torch.float64)
                                K_fake = max(float(reg_bins_tensor.numel()), 1.0)
                                s = 1.0 - D / K_fake
                                total_same_exp += s.sum()
                                total_edges    += s.new_tensor(s.numel())
                                deg_v.index_add_(0, v[ve], torch.ones_like(s, dtype=deg_v.dtype))
                            elif regression_mode == "corr":
                                if y_norm is None or sum_yu is None or M_edges is None:
                                    raise RuntimeError("corr mode expects normalized labels and accumulators")
                                yu_corr = y_norm[u][ve]
                                yv_corr = y_norm[v][ve]
                                sum_yu += yu_corr.sum()
                                sum_yv += yv_corr.sum()
                                sum_yu2 += (yu_corr * yu_corr).sum()
                                sum_yv2 += (yv_corr * yv_corr).sum()
                                sum_yuyv += (yu_corr * yv_corr).sum()
                                M_edges += yu_corr.new_tensor(yu_corr.numel())
                                deg_v.index_add_(0, v[ve], torch.ones_like(yu_corr, dtype=deg_v.dtype))
                            else:
                                raise RuntimeError(f"Unsupported regression_mode='{regression_mode}'")

                # ----- aggregation accumulators: neighbors must be labeled on v side -----
                valid_n = labeled_mask[v]
                if valid_n.any():
                    idx_u = u[valid_n]
                    if task_type == "classification":
                        Pv_u = P[v[valid_n]]  # [E_valid, C]
                        pm = (Pv_u >= 0).all(dim=1) if (P < 0).any() else torch.ones(Pv_u.size(0), dtype=torch.bool, device=device)
                        if pm.any():
                            Pv_u_valid = Pv_u[pm]
                            idx_u_valid = idx_u[pm]
                            neigh_sum_u.index_add_(0, idx_u_valid, Pv_u_valid)
                            deg_u.index_add_(0, idx_u_valid, torch.ones(idx_u_valid.numel(), device=device, dtype=deg_u.dtype))
                    else:
                        neighbor_idx = v[valid_n]
                        yv_u = y_all[neighbor_idx]
                        pm = (yv_u != -1.0)
                        if pm.any():
                            idx_u_valid = idx_u[pm]
                            if regression_mode == "gaussian":
                                Zv_u_valid = Z_all[neighbor_idx][pm]
                                neigh_sum_u_z.index_add_(0, idx_u_valid, Zv_u_valid)
                            elif regression_mode == "fake_cls":
                                if P_multi is None:
                                    raise RuntimeError("P_multi is required for regression_mode='fake_cls'")
                                Pv_valid = P_multi[neighbor_idx][pm]
                                neigh_sum_u_multi.index_add_(0, idx_u_valid, Pv_valid)
                            elif regression_mode == "corr":
                                # corr mode only needs degrees for coverage; no neighbor embeddings
                                pass
                            else:
                                raise RuntimeError(f"Unsupported regression_mode='{regression_mode}' for aggregation accumulation")
                            deg_u.index_add_(0, idx_u_valid, torch.ones(idx_u_valid.numel(), device=device, dtype=deg_u.dtype))

                if (task_type == "classification" or regression_mode != "corr") and float(total_edges) > 0.0:
                    pbar.set_postfix(h_edge=float((total_same_exp / (total_edges + 1e-12)).item()))

            # If no valid edges accumulated for this metapath, skip its finalize step (except corr mode).
            if not (task_type == "regression" and regression_mode == "corr"):
                if total_edges.item() == 0:
                    continue
        except Exception as e:
            ## usually this is becuase too much memory is used
            print(f"Error in metapath {mp}: {e}")
            continue

        # ---- finalize edge-level metrics ----
        h_edge = float("nan")
        baseline_value = 0.0

        # if h_edge == 1.0:
        #     ## such kind of metapath is not useful, skip it
        #     continue

        if task_type == "classification":
            h_edge = float((total_same_exp / total_edges).item())
            # class-insensitive: compare class-conditional edge similarity vs class prior
            h_k = torch.nan_to_num(numer_k / torch.clamp_min(denom_k, 1e-12))  # [C]
            prior = proportions.squeeze(0)                                      # [C]
            # avg positive excess above prior, normalized by (C-1)
            h_ins = (h_k - prior).clamp_min_(0).sum().item() / max(C - 1, 1)

            # degree-adjusted normalization removes degree-driven chance
            # Only count classes on labeled nodes; deg_v has mass only on labeled v
            P_l = P[labeled_mask]                     # [N_l, C]
            deg_v_l = deg_v[labeled_mask]             # [N_l]
            degree_sums_by_class = (P_l.T @ deg_v_l)  # [C]
            num_edges = total_edges.item()
            adjust = float((degree_sums_by_class.pow(2).sum() / (num_edges ** 2)).item())
            h_adj = (h_edge - adjust) / (1.0 - adjust + 1e-12)
            baseline_value = adjust
        else:
            if regression_mode == "corr":
                if M_edges is None or M_edges.item() == 0:
                    h_edge = float("nan")
                    h_ins = float("nan")
                    h_adj = float("nan")
                else:
                    mu_u = sum_yu / M_edges
                    mu_v = sum_yv / M_edges
                    var_u = sum_yu2 / M_edges - mu_u * mu_u
                    var_v = sum_yv2 / M_edges - mu_v * mu_v
                    cov = sum_yuyv / M_edges - mu_u * mu_v
                    eps = 1e-12
                    denom = (var_u * var_v + eps).sqrt()
                    r_edge = cov / denom
                    h_edge = float(r_edge.item())
                    h_ins = h_edge
                    h_adj = h_edge
                baseline_value = 0.0
            elif regression_mode == "gaussian":
                h_edge = float((total_same_exp / total_edges).item())
                baseline = 1.0 / math.sqrt(1.0 + 2.0 * (var_y / (sigma_y ** 2) if sigma_y > 0 else 0.0))
                h_ins = h_edge
                h_adj = (h_edge - baseline) / (1.0 - baseline + 1e-12)
                baseline_value = baseline
            elif regression_mode == "fake_cls":
                h_edge = float((total_same_exp / total_edges).item())
                if baseline_reg is None:
                    raise RuntimeError("baseline_reg is undefined for fake classification mode")
                baseline = baseline_reg
                h_ins = h_edge
                h_adj = (h_edge - baseline) / (1.0 - baseline + 1e-12)
                baseline_value = baseline
            else:
                raise RuntimeError(f"Unsupported regression_mode='{regression_mode}'")

        # ---- aggregation homophily (LOO-approx) ----
        mask_deg = deg_u > 0
        DN = None
        DNz = None
        DN_multi: Optional[torch.Tensor] = None

        if task_type == "classification":
            DN = torch.zeros_like(neigh_sum_u)  # [N, C]
            DN[mask_deg] = neigh_sum_u[mask_deg] / deg_u[mask_deg].unsqueeze(1)

            # nodes we can score: labeled, has neighbors, and (if using -1) labels are valid
            if (P < 0).any():
                valid_p_rows = (P >= 0).all(dim=1)
            else:
                valid_p_rows = torch.ones(N, dtype=torch.bool, device=device)

            used = labeled_mask & mask_deg & valid_p_rows
            h_agg = _agg_homophily_soft_cls(DN, P, used, proportions.squeeze(0))
        else:
            if regression_mode == "gaussian":
                Dz = Z_all.shape[1]
                DNz = torch.zeros((N, Dz), device=device, dtype=torch.float64)
                DNz[mask_deg] = neigh_sum_u_z[mask_deg] / deg_u[mask_deg].unsqueeze(1)

                valid_p_rows = (y_all != -1.0)
                used = labeled_mask & mask_deg & valid_p_rows
                if used.any():
                    U = used.nonzero(as_tuple=False).squeeze(1)   # [M]
                    M = U.numel()
                    # small anchor set to keep complexity O(MA)
                    max_anchors = 1024
                    if M <= max_anchors:
                        A = U
                    else:
                        A = U[torch.randperm(M, device=device)[:max_anchors]]

                    ZU = DNz[U]                    # [M, Dz]
                    ZA = DNz[A]                    # [A, Dz]
                    S = ZU @ ZA.T                  # [M, A] (dot of neighbor distributions)

                    yu = y_all[U]                  # [M]
                    ya = y_all[A]                  # [A]
                    W = torch.exp(- (yu[:, None] - ya[None, :]).pow(2) / (2.0 * (sigma_y ** 2) + 1e-12))  # [M,A]
                    # strict LOO for anchors
                    eq = (U[:, None] == A[None, :])
                    if eq.any():
                        W = W.masked_fill(eq, 0.0)

                    W_sum  = W.sum(dim=1)                     # [M]
                    WC_sum = (1.0 - W).sum(dim=1)             # [M]
                    same_avg = ((W * S).sum(dim=1) / (W_sum + 1e-12))
                    diff_avg = (((1.0 - W) * S).sum(dim=1) / (WC_sum + 1e-12))

                    ok = (W_sum > 0) & (WC_sum > 0)
                    h_agg = float(((same_avg[ok] >= diff_avg[ok]).float().mean().item())) if ok.any() else float('nan')
                else:
                    h_agg = float('nan')
            elif regression_mode == "fake_cls":
                if P_multi is None or p_cols is None:
                    raise RuntimeError("Fake classification mode requires multi-label encodings and marginals")
                DN_multi = torch.zeros((N, fake_cls_dim), device=device, dtype=torch.float64)
                DN_multi[mask_deg] = neigh_sum_u_multi[mask_deg] / deg_u[mask_deg].unsqueeze(1)
                valid_p_rows = (y_all != -1.0)
                used = labeled_mask & mask_deg & valid_p_rows
                if used.any():
                    h_agg = _agg_homophily_soft_cls(DN_multi, P_multi, used, p_cols)
                else:
                    h_agg = float("nan")
            elif regression_mode == "corr":
                h_agg = float("nan")
            else:
                raise RuntimeError(f"Unsupported regression_mode='{regression_mode}' for aggregation scoring")

        # ---- compute per-sample homophily if requested ----
        if return_per_sample:
            # Per-sample homophily: for each node, compute homophily with its neighbors
            per_sample_h = torch.zeros(N, device=device, dtype=torch.float64)

            if task_type == "classification":
                # For each node with neighbors, compute similarity with neighbor distribution
                if mask_deg.any():
                    # Compute similarity between node label and neighbor distribution
                    sim = (P[mask_deg] * DN[mask_deg]).sum(dim=1)  # [num_nodes_with_neighbors]
                    per_sample_h[mask_deg] = sim
            else:
                if regression_mode == "gaussian":
                    # For regression, compute similarity using RFF embeddings
                    if mask_deg.any():
                        Zu = Z_all[mask_deg]  # [num_nodes_with_neighbors, 2H]
                        DNzu = DNz[mask_deg]  # [num_nodes_with_neighbors, 2H]
                        norm_Zu = torch.linalg.norm(Zu, dim=1, keepdim=True).clamp_min(1e-12)
                        norm_DNzu = torch.linalg.norm(DNzu, dim=1, keepdim=True).clamp_min(1e-12)
                        sim = (Zu * DNzu).sum(dim=1) / (norm_Zu.squeeze() * norm_DNzu.squeeze())
                        per_sample_h[mask_deg] = sim.clamp(0.0, 1.0)
                elif regression_mode == "fake_cls":
                    if mask_deg.any():
                        if DN_multi is None or P_multi is None:
                            raise RuntimeError("Per-sample fake classification requires DN_multi and P_multi")
                        K_fake = max(float(fake_cls_dim), 1.0)
                        sim = (P_multi[mask_deg] * DN_multi[mask_deg]).sum(dim=1) / K_fake
                        per_sample_h[mask_deg] = sim.clamp(0.0, 1.0)
                elif regression_mode == "corr":
                    if mask_deg.any():
                        per_sample_h[mask_deg] = float("nan")
                else:
                    raise RuntimeError(f"Unsupported regression_mode='{regression_mode}' for per-sample homophily")

            # Store per-sample homophily for this metapath
            # Convert list to tuple to make it hashable for dict key
            per_sample_homophily_dict[tuple(mp)] = per_sample_h.cpu()

        active_labeled = int(((mask_deg) & labeled_mask).sum().item())
        sparsities.append(active_labeled)
        if empty_as_0 and total_labeled_nodes > 0:
            coverage = active_labeled / total_labeled_nodes
            baseline_scaled = baseline_value * coverage
            h_edge = h_edge * coverage
            h_ins = h_ins * coverage
            h_adj = (h_edge - baseline_scaled) / (1.0 - baseline_scaled + 1e-12)
            if not math.isnan(h_agg):
                h_agg = h_agg * coverage
            elif active_labeled == 0:
                h_agg = 0.0

        # collect per-metapath metrics
        h_edges.append(h_edge)
        h_insensitives.append(h_ins)
        h_adjs.append(h_adj)
        h_aggs.append(h_agg)
        metapath_degrees.append((mp, deg_u.clone(), deg_v.clone()))

    if len(h_edges) == 0:
        if return_per_sample:
            return [], [], [], [], [], [], {}
        return [], [], [], [], [], []

    if return_per_sample:
        return (
            h_edges,
            h_insensitives,
            h_adjs,
            h_aggs,
            sparsities,
            metapath_degrees,
            per_sample_homophily_dict,
        )
    return h_edges, h_insensitives, h_adjs, h_aggs, sparsities, metapath_degrees



# ------------------ the simplified wrapper you asked for ------------------

def homophily_from_table_db(
    data,
    labels_table_obj,
    *,
    db_obj,                         # has .table_dict[table_name] with fields: df, fkey_col_to_pkey_table, pkey_col, time_col
    task_entity: str,               # graph node type, and also the corresponding table name in db_obj
    entity_fk_override: str | None = None,
    label_col: str = "qualifying",
    half_life_days: float | None = None,
    fill_unlabeled: str = "uniform",
    chunk_size: int = 100,
    num_classes: int | None = None,
    task_type: str = "classification",
    is_autocomplete: bool = False,
    max_group: int = 300,
    return_per_sample: bool = False,
    regression_mode: Literal["gaussian", "fake_cls", "corr"] = "gaussian",
    regression_bins: Optional[List[float]] = None,
    empty_as_0: bool = False,
):
    """
    Wrapper that:
      (1) infers the FK column -> task_entity,
      (2) aligns node primary keys,
      (3) builds soft labels [N,C] (classification) or [N,1] (regression),
      (4) runs metapath-based soft homophily and aggregation homophily.

    For autocomplete tasks (is_autocomplete=True):
    - labeled_mask will be True for all nodes (train, validation, and test)
    - P values will be set to -1 for test nodes to indicate they should not be used for training
    - Homophily calculations will properly handle test nodes by filtering out edges with -1 values

    Returns:
      h_edges, h_insensitives, h_adjs, h_aggs, sparsities  (each a per-metapath list)

    Args:
      regression_mode: Controls regression treatment ("gaussian", "fake_cls", or "corr").
      regression_bins: Thresholds for the fake classification mode.
      empty_as_0: When True, labeled nodes without neighbors count as zero when averaging homophily.
    """
    ent_meta = db_obj.table_dict[task_entity]

    lbl_df = labels_table_obj.df
    lbl_time_col = None  # keep as in your original wrapper

    ent_pkey_col = _get_field(ent_meta, "pkey_col", None)
    ent_df = _get_field(ent_meta, "df", None)
    if ent_pkey_col is None:
        logger.error(f"task_entity table '{task_entity}' must provide 'pkey_col'.")
        ## add a pk column to the table
        ent_meta.df[task_entity + '_pk'] = np.arange(len(ent_df))
        ent_pkey_col = task_entity + '_pk'
        db_obj.table_dict[task_entity].df[task_entity + '_pk'] = ent_meta.df[task_entity + '_pk']
        setattr(db_obj.table_dict[task_entity], "pkey_col", ent_pkey_col)

    # figure out the FK column in labels_df that targets the task_entity table
    fkey_col = _pick_fk_for_entity(labels_table_obj, task_entity, override_fk=entity_fk_override)

    # get node primary keys aligned to node index order
    if ent_pkey_col in data[task_entity]:
        node_pkeys = np.asarray(data[task_entity][ent_pkey_col].cpu().numpy())
    else:
        if ent_df is None or ent_pkey_col not in ent_df.columns:
            raise ValueError(
                f"Cannot find primary keys for node type '{task_entity}'. "
                f"Neither data['{task_entity}']['{ent_pkey_col}'] nor "
                f"db_obj.table_dict['{task_entity}'].df['{ent_pkey_col}'] exists."
            )
        node_pkeys = ent_df[ent_pkey_col].to_numpy()

    # build soft labels [N, C] (or [N,1] for regression)
    # For autocomplete, we need to pass graph_num_nodes to ensure P matrix matches graph size
    graph_num_nodes = data.num_nodes_dict.get(task_entity) if is_autocomplete else None

    P, labeled_mask = build_label_soft_from_event_table_df(
        labels_df=lbl_df,
        fkey_col=fkey_col,
        node_pkeys=node_pkeys,
        label_col=label_col,
        time_col=lbl_time_col,
        num_classes=num_classes,
        half_life_days=half_life_days,
        end_time=None,
        fill_unlabeled=fill_unlabeled,
        task_type=task_type,
        is_autocomplete=is_autocomplete,
        db_obj=db_obj,
        task_entity=task_entity,
        graph_num_nodes=graph_num_nodes,
    )

    # run homophily (soft) + aggregation (classification) on 2-hop metapaths
    homophily_result = calcHomophily_soft(
        data,
        task_entity=task_entity,
        label_soft=P,
        labeled_mask=labeled_mask,
        chunk_size=chunk_size,
        task_type=task_type,
        max_group=max_group,
        return_per_sample=return_per_sample,
        regression_mode=regression_mode,
        regression_bins=regression_bins,
        empty_as_0=empty_as_0,
    )
    return homophily_result, P, labeled_mask


def enumerate_two_hop_metas(data, task_entity: str) -> List[Tuple[Tuple[str,str,str], Tuple[str,str,str]]]:
    metas = []
    def ext(cur: str, mp: List[Tuple[str,str,str]], hop: int):
        if hop < 1:
            if cur == task_entity: metas.append(tuple(mp))
            return
        for et in data.edge_types:
            s, _, t = et
            if s == cur: ext(t, mp + [et], hop - 1)
    ext(task_entity, [], 2)
    return [m for m in metas if (m[0][0] == task_entity and m[0][2] == m[1][0] and m[1][2] == task_entity)]


# =============================================================================
#  Type-aware Feature Similarity (timestamp / embedding / categorical / numerical)
# =============================================================================

def _get_feat(tf, key, default=None):
    fd = getattr(tf, "feat_dict", None)
    if fd is None and hasattr(tf, "tf"):
        fd = getattr(tf.tf, "feat_dict", None)
    if fd is None and isinstance(tf, dict):
        fd = tf.get("feat_dict", None)
    if isinstance(fd, dict):
        if key in fd: return fd[key]
        for k in fd.keys():
            if str(k) == str(key):
                return fd[k]
    return default

def _as_2d_float(x: torch.Tensor) -> torch.Tensor:
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if x.dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    if x.ndim == 1:
        x = x[:, None]
    return x

def _as_2d_long(x: torch.Tensor) -> torch.Tensor:
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if x.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
        x = x.long()
    elif x.dtype == torch.bool:
        x = x.long()
    if x.ndim == 1:
        x = x[:, None]
    return x

@dataclass
class FeatureViews:
    timestamp: Optional[torch.Tensor]   # [N, d_t]
    embeddings: torch.Tensor      # list of [N, d_ej]
    categorical: Optional[torch.Tensor] # [N, d_c] int codes (missing = -1 recommended)
    numerical: Optional[torch.Tensor]   # [N, d_n] float

def extract_views_from_tensorframe(
    tf,
    *,
    ts_key: Any = "timestamp",
    emb_key: Any = "embedding",
    cat_key: Any = "categorical",
    num_key: Any = "numerical",
) -> FeatureViews:
    dev = getattr(tf, "device", "cpu")
    ts = _get_feat(tf, ts_key, None)
    emb = _get_feat(tf, emb_key, None)
    cat = _get_feat(tf, cat_key, None)
    num = _get_feat(tf, num_key, None)

    # timestamp
    if ts is not None:
        ts = torch.as_tensor(ts, device=dev)
        if ts.ndim == 1: ts = ts[:, None]
        elif ts.ndim > 2: ts = ts.view(ts.shape[0], -1)

    # embeddings
    if emb is not None:
        embeddings: torch.Tensor = emb.values
    else:
        # Create empty embeddings tensor if none exist
        embeddings = torch.empty(0, 0, device=dev)

    # categorical
    if cat is not None:
        cat = torch.as_tensor(cat, device=dev)
        cat = _as_2d_long(cat)

    # numerical
    if num is not None:
        num = torch.as_tensor(num, device=dev)
        num = _as_2d_float(num)

    fv = FeatureViews(timestamp=ts, embeddings=embeddings, categorical=cat, numerical=num)
    fv.num_elements = tf.num_rows
    return fv

def _timestamp_to_days(ts_row: torch.Tensor) -> torch.Tensor:
    if ts_row.numel() == 1:
        x = ts_row.squeeze()
        if x.abs() > 10_000:  # seconds→days heuristic
            return x / 86_400.0
        return x.to(torch.float32)
    elif ts_row.numel() >= 3:
        y, m, d = ts_row[0].float(), ts_row[1].float(), ts_row[2].float()
        return 365.0 * y + 30.0 * m + d
    ws = torch.logspace(0, -2, steps=ts_row.numel(), base=10.0, device=ts_row.device)
    return (ts_row.float() * ws).sum()

def _timestamp_similarity(ts: Optional[torch.Tensor], u: torch.Tensor, v: torch.Tensor,
                          *, sigma_days: Optional[float] = None) -> torch.Tensor:
    B = u.numel()
    if ts is None:
        return torch.zeros(B, dtype=torch.float32, device=u.device)
    days_u = torch.stack([_timestamp_to_days(ts[i]) for i in u], dim=0)
    days_v = torch.stack([_timestamp_to_days(ts[j]) for j in v], dim=0)
    diffs = (days_u - days_v).abs()
    if sigma_days is None:
        all_days = torch.cat([days_u, days_v], dim=0)
        q75, q25 = torch.quantile(all_days, 0.75), torch.quantile(all_days, 0.25)
        iqr = (q75 - q25).item()
        sigma_days = (iqr / 1.349) if iqr > 1e-9 else float(all_days.std().item() + 1e-6)
    sigma_days = max(float(sigma_days), 1e-6)
    return torch.exp(- (diffs ** 2) / (2.0 * sigma_days ** 2))

def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    an = torch.linalg.norm(a, dim=-1).clamp_min(eps)
    bn = torch.linalg.norm(b, dim=-1).clamp_min(eps)
    return (a * b).sum(dim=-1) / (an * bn)

def _embedding_similarity(embeddings: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B = u.numel()
    if embeddings.numel() == 0:
        return torch.zeros(B, dtype=torch.float32, device=u.device)
    
    # Memory-efficient approach: compute similarity in chunks if needed
    device = u.device
    max_chunk_size = 10000  # Adjust based on available GPU memory
    
    if B <= max_chunk_size:
        # Small batch: transfer to GPU and compute directly
        embeddings_gpu = embeddings.to(device)
        s = _cosine_sim(embeddings_gpu[u, :], embeddings_gpu[v, :])
    else:
        # Large batch: compute in chunks to avoid OOM
        s = torch.zeros(B, dtype=torch.float32, device=device)
        for i in range(0, B, max_chunk_size):
            end_idx = min(i + max_chunk_size, B)
            chunk_u = u[i:end_idx]
            chunk_v = v[i:end_idx]
            
            # Transfer only the needed embeddings for this chunk
            unique_indices = torch.cat([chunk_u, chunk_v]).unique()
            embeddings_chunk = embeddings[unique_indices].to(device)
            
            # Create mapping from original indices to chunk indices
            index_map = {idx.item(): i for i, idx in enumerate(unique_indices)}
            chunk_u_mapped = torch.tensor([index_map[idx.item()] for idx in chunk_u], device=device)
            chunk_v_mapped = torch.tensor([index_map[idx.item()] for idx in chunk_v], device=device)
            
            s[i:end_idx] = _cosine_sim(embeddings_chunk[chunk_u_mapped, :], embeddings_chunk[chunk_v_mapped, :])
    
    return torch.clamp((s + 1.0) * 0.5, 0.0, 1.0)

def _categorical_similarity(cat: Optional[torch.Tensor], u: torch.Tensor, v: torch.Tensor,
                            *, missing_code: int = -1, reweight: Literal["none","idf"]="none") -> torch.Tensor:
    B = u.numel()
    if cat is None:
        return torch.zeros(B, dtype=torch.float32, device=u.device)
    Cu = cat[u, :]
    Cv = cat[v, :]
    valid = (Cu != missing_code) & (Cv != missing_code)
    same = (Cu == Cv) & valid
    if reweight == "idf":
        # Lightweight IDF: column-wise frequency → weight = 1 / sqrt(freq)
        C = cat.shape[1]
        w_cols = []
        for c in range(C):
            col = cat[:, c]
            mask = (col != missing_code)
            vals, counts = torch.unique(col[mask], return_counts=True)
            # map each value to weight 1/sqrt(freq)
            wmap = {int(vals[i]): 1.0 / math.sqrt(float(counts[i].item())) for i in range(vals.numel())}
            w = torch.tensor([wmap.get(int(x.item()), 0.0) for x in Cu[:, c]], device=cat.device).float()
            w_cols.append(w * valid[:, c].float())
        W = torch.stack(w_cols, dim=1)
        num = (same.float() * W).sum(dim=1)
        den = W.sum(dim=1).clamp_min(1e-12)
        return (num / den).clamp(0.0, 1.0)
    else:
        num = same.float().sum(dim=1)
        den = valid.float().sum(dim=1).clamp_min(1e-12)
        return (num / den).clamp(0.0, 1.0)

def _numerical_similarity(num: Optional[torch.Tensor], u: torch.Tensor, v: torch.Tensor,
                          *, metric: Literal["gaussian","l1"]="gaussian") -> torch.Tensor:
    B = u.numel()
    if num is None:
        return torch.zeros(B, dtype=torch.float32, device=u.device)
    Nu = num[u, :]
    Nv = num[v, :]
    q75, q25 = torch.quantile(num, 0.75, dim=0), torch.quantile(num, 0.25, dim=0)
    s = (q75 - q25) / 1.349
    s = torch.where(s > 1e-9, s, num.std(dim=0) + 1e-6).clamp_min(1e-6)
    diff = (Nu - Nv).abs()
    if metric == "gaussian":
        sim_cols = torch.exp(- (diff / s)**2 / 2.0)
    else:
        sim_cols = 1.0 / (1.0 + diff / s)
    return sim_cols.mean(dim=1).clamp(0.0, 1.0)

@dataclass
class FeatureSimilarityConfig:
    sigma_days: Optional[float] = None
    cat_missing_code: int = -1
    cat_reweight: Literal["none","idf"] = "none"
    num_metric: Literal["gaussian","l1"] = "gaussian"

class FeatureSimilarity:
    def __init__(self, cfg: FeatureSimilarityConfig | None = None):
        self.cfg = cfg or FeatureSimilarityConfig()

    @torch.no_grad()
    def pairwise(self, views: FeatureViews, u_idx: torch.Tensor, v_idx: torch.Tensor) -> torch.Tensor:
        u = u_idx.long(); v = v_idx.long()
        # ts_sim  = _timestamp_similarity(views.timestamp, u, v, sigma_days=self.cfg.sigma_days)
        emb_sim = _embedding_similarity(views.embeddings, u, v)
        cat_sim = _categorical_similarity(views.categorical, u, v,
                                          missing_code=self.cfg.cat_missing_code,
                                          reweight=self.cfg.cat_reweight)
        num_sim = _numerical_similarity(views.numerical, u, v, metric=self.cfg.num_metric)
        return torch.stack([emb_sim, cat_sim, num_sim], dim=1)
