import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch_geometric.data import HeteroData

class RandomFeatureRegressor(nn.Module):
    """
    Closed-form prediction heads on frozen random GNN features.

    - task_kind='regression': ridge regression (closed-form)
    - task_kind='binary':     LDA (closed-form) -> single logit
    """

    def __init__(
        self,
        base_model: nn.Module,
        task_kind: str = "regression",
        ridge_lambda: float = 1.0,
        lda_shrinkage: float = 1e-3,
        standardize: bool = True,
        max_points: int = 10000,
    ):
        super().__init__()
        assert task_kind in ("regression", "binary")
        # Keep a handle to the base model (we will call its encoders + GNN)
        self.base = base_model.eval()
        self.task_kind: str = task_kind
        self.ridge_lambda = float(ridge_lambda)
        self.lda_shrinkage = float(lda_shrinkage)
        self.standardize = bool(standardize)
        self.max_points = int(max_points)

        # Flag so external utilities can detect this wrapper
        self._is_rfr = True
        self._rfr_task_kind = task_kind

        # Buffers for standardization & linear head params
        self.register_buffer("mu", torch.tensor([]))        # [D]
        self.register_buffer("inv_std", torch.tensor([]))   # [D]
        self.register_buffer("w", torch.tensor([]))         # [D]
        self.register_buffer("b", torch.tensor([0.0]))      # [1]

    # ------------------------------ Embedding path ------------------------------

    @torch.no_grad()
    def _embed_batch(
        self,
        batch: HeteroData,
        entity_table: str,
        dst_table: str,
        read_out_table: str = "dst",
        device: torch.device = torch.device("cuda"),
        meta_graph=None,
    ) -> torch.Tensor:
        """
        Build pre-head embedding φ(x) by replaying the base model's
        encoder -> (temporal) -> GNN pipeline. Do NOT call the base head.
        """
        # Make sure base modules and batch are on the same device
        self.base.to(device)
        batch = batch.to(device)

        seed_time = batch[entity_table].seed_time

        # 1) Heterogeneous tabular features
        x_dict = self.base.feature_encoder(batch.tf_dict)

        # 2) Anchor / structural feature injection
        x_dict = self.base.apply_auto_structural_feature(
            x_dict, batch, entity_table, dst_table,
            self.base.model_config['gnn_config']['pre_sf']
        )

        # 3) Temporal encoding (if enabled)
        if not getattr(self.base, "time_free", False):
            rel_time_dict = self.base.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for nt, rel in rel_time_dict.items():
                x_dict[nt] = x_dict[nt] + rel

        # 4) Message passing → node-type embeddings
        x_dict = self.base.gnn(x_dict, batch.edge_index_dict)

        # 5) Read-out selection and crop to seed
        out_table = dst_table if read_out_table == "dst" else entity_table
        emb = x_dict[out_table][: seed_time.size(0)]  # [B, D]
        return emb

    # ---------------------------- Data collection -----------------------------

    @torch.no_grad()
    def _collect_phi_y(
        self,
        loader,
        entity_table: str,
        dst_table: str,
        device: torch.device,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iterate the train loader once to collect features Phi and targets y.
        """
        phis, ys = [], []
        seen = 0
        self.base.eval()
        self.base.to(device)

        for batch in loader:
            batch = batch.to(device)
            phi = self._embed_batch(batch, entity_table, dst_table, device=device)  # [B, D]
            # y from the seed part only
            y = batch[entity_table].y.view(-1)[: phi.size(0)].float()

            # Optional clamp for regression targets
            if (clamp_min is not None) or (clamp_max is not None):
                y = torch.clamp(
                    y,
                    min=clamp_min if clamp_min is not None else float("-inf"),
                    max=clamp_max if clamp_max is not None else float("inf"),
                )

            phis.append(phi.detach())
            ys.append(y.detach())

            seen += phi.size(0)
            if seen >= self.max_points:
                break

        if len(phis) == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        Phi = torch.cat(phis, dim=0).to(device)  # [N, D]
        y = torch.cat(ys, dim=0).to(device)      # [N]
        return Phi, y

    # ------------------------------ Solvers -----------------------------------

    @torch.no_grad()
    def _solve_ridge_regression(self, Phi: torch.Tensor, y: torch.Tensor, lam: float) -> None:
        """
        min_w ||Phi w - y||^2 + lam ||w||^2  (bias unregularized)
        """
        device = Phi.device
        N, D = Phi.shape

        if N == 0 or D == 0:
            # Constant fallback
            self.w = torch.zeros(D, device=device)
            self.b = torch.zeros(1, device=device)
            self.mu = torch.zeros(D, device=device)
            self.inv_std = torch.ones(D, device=device)
            return

        # Standardization
        if self.standardize:
            mu = Phi.mean(dim=0)
            std = Phi.std(dim=0, unbiased=False).clamp_min(1e-6)
            Phi_std = (Phi - mu) / std
            self.mu = mu
            self.inv_std = 1.0 / std
        else:
            Phi_std = Phi
            self.mu = torch.zeros(D, device=device)
            self.inv_std = torch.ones(D, device=device)

        # Bias column (not regularized)
        ones = torch.ones(N, 1, device=device, dtype=Phi_std.dtype)
        Phi_aug = torch.cat([Phi_std, ones], dim=1)  # [N, D+1]

        reg = torch.zeros(D + 1, device=device, dtype=Phi_std.dtype)
        reg[:D] = lam

        if (D + 1) <= N:
            A = Phi_aug.T @ Phi_aug
            A = A + torch.diag(reg)
            b = Phi_aug.T @ y
            try:
                L = torch.linalg.cholesky(A)
                w_aug = torch.cholesky_solve(b.unsqueeze(1), L).squeeze(1)
            except RuntimeError:
                w_aug = torch.linalg.solve(A, b)
        else:
            # Dual for N << D
            K = Phi_aug @ Phi_aug.T
            K = K + lam * torch.eye(N, device=device, dtype=Phi_std.dtype)
            alpha = torch.linalg.solve(K, y)
            w_aug = Phi_aug.T @ alpha  # [D+1]

        self.w, self.b = w_aug[:-1], w_aug[-1]

    @torch.no_grad()
    def _solve_binary_lda(self, Phi: torch.Tensor, y: torch.Tensor, shrink: float) -> None:
        """
        LDA for binary classification on standardized features.
        Produces a single logit = (w^T z + b).
        """
        device = Phi.device
        N, D = Phi.shape
        if N == 0 or D == 0:
            self.w = torch.zeros(D, device=device)
            self.b = torch.zeros(1, device=device)
            self.mu = torch.zeros(D, device=device)
            self.inv_std = torch.ones(D, device=device)
            return

        # Standardize features
        if self.standardize:
            mu_all = Phi.mean(dim=0)
            std_all = Phi.std(dim=0, unbiased=False).clamp_min(1e-6)
            Z = (Phi - mu_all) / std_all   # [N, D]
            self.mu = mu_all
            self.inv_std = 1.0 / std_all
        else:
            Z = Phi
            self.mu = torch.zeros(D, device=device)
            self.inv_std = torch.ones(D, device=device)

        # Split by class
        y_bin = (y > 0.5).to(torch.bool)
        Z0 = Z[~y_bin]
        Z1 = Z[y_bin]
        n0, n1 = Z0.shape[0], Z1.shape[0]
        eps_eye = shrink * torch.eye(D, device=device, dtype=Z.dtype)

        if n0 == 0 or n1 == 0:
            # Degenerate: constant logit
            prior1 = (n1 + 1e-8) / (n0 + n1 + 2e-8)
            self.w = torch.zeros(D, device=device)
            self.b = torch.tensor([torch.logit(torch.tensor(prior1, device=device))], device=device)
            return

        mu0 = Z0.mean(dim=0)
        mu1 = Z1.mean(dim=0)

        def _cov(X):
            if X.shape[0] <= 1:
                return torch.zeros(D, D, device=device, dtype=Z.dtype)
            Xc = X - X.mean(dim=0, keepdim=True)
            return (Xc.T @ Xc) / (X.shape[0] - 1)

        S0 = _cov(Z0)
        S1 = _cov(Z1)
        denom = max(n0 + n1 - 2, 1)
        Sp = ((n0 - 1) * S0 + (n1 - 1) * S1) / denom
        Sp = Sp + eps_eye

        delta = (mu1 - mu0)  # [D]
        try:
            L = torch.linalg.cholesky(Sp)
            w = torch.cholesky_solve(delta.unsqueeze(1), L).squeeze(1)
        except RuntimeError:
            w = torch.linalg.solve(Sp, delta)

        prior0 = n0 / (n0 + n1)
        prior1 = n1 / (n0 + n1)
        b = -0.5 * (mu1 + mu0).matmul(w) + torch.log(torch.tensor(prior1 / prior0, device=device, dtype=Z.dtype))

        self.w = w
        self.b = b.view(1)

    # ------------------------------- Public API -------------------------------

    @torch.no_grad()
    def fit_from_loader(
        self,
        loader,
        entity_table: str,
        dst_table: str,
        device: torch.device,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
    ) -> float:
        """
        Collect (Phi, y) and solve the appropriate head.
        Returns:
          - regression: MAE on train set
          - binary:     BCE-with-logits on train set
        """
        Phi, y = self._collect_phi_y(
            loader, entity_table, dst_table, device, clamp_min, clamp_max
        )
        if self.task_kind == "regression":
            self._solve_ridge_regression(Phi, y, self.ridge_lambda)
            if Phi.numel() == 0:
                return 0.0
            Z = (Phi - self.mu) * self.inv_std if self.standardize else Phi
            y_hat = Z @ self.w + self.b
            return F.l1_loss(y_hat, y).item()  # report MAE
        else:
            self._solve_binary_lda(Phi, y, self.lda_shrinkage)
            if Phi.numel() == 0:
                return 0.0
            Z = (Phi - self.mu) * self.inv_std if self.standardize else Phi
            logits = Z @ self.w + self.b
            return F.binary_cross_entropy_with_logits(logits.view(-1), (y > 0.5).float()).item()

    @torch.no_grad()
    def forward(
        self,
        batch: HeteroData,
        entity_table: str,
        dst_table: str,
        read_out_table: str = "dst",
        **kwargs,
    ) -> torch.Tensor:
        """
        Keep the same signature as your base model so generate_representation(...)
        can call this seamlessly.
        Returns:
            [B,1] continuous (regression) or [B,1] logits (binary).
        """
        device = next(self.parameters()).device
        phi = self._embed_batch(batch, entity_table, dst_table, read_out_table, device=device)  # [B, D]
        if self.standardize and self.mu.numel() == phi.size(1):
            phi = (phi - self.mu) * self.inv_std

        out = phi @ self.w + self.b  # [B]
        return out.view(-1, 1)