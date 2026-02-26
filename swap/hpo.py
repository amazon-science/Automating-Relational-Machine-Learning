
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import Dict, List, Tuple, Any, Optional, Sequence, Union
from hyperopt import STATUS_OK
import optuna
from optuna.trial import TrialState
import os 
import inspect
from utils.hpo import (
    DevicePool, rng_to_seed, weights_to_repeated_choices, config_to_key,
    build_experiment3_catalog, load_heuristics_features,
    compute_landscape_metrics_for_trial, val_score_of_result, resolve_devices,
)
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from utils.type import seed_everything
from loguru import logger
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from swap.gym import (
    get_anchor_models_and_task_embedding, _task_sim_from_anchor_ranks,
    plot_combined_similarity_ranking_correlation,
)
from utils.database import SwapDB, create_swap_db
from utils.misc import load_yaml
from utils.information import (
    RDL_CONSIDERED_PARAMS, DFS_FTT_CONSIDERED_PARAMS, ALL_TASK_LIST,
    PRETRAINED_TASK_LIST, EVAL_TASK_LIST, rdl_column_name_mapping,
    ftt_column_name_mapping, task_name_to_metric_no_prefix,
)
from utils.retrieve_search_data import load_validated_runs




class TaskEmbeddingHPO:
    """
    Hyperparameter optimization guided by task embedding similarity.
    
    Uses task embeddings (homophily, heuristics, etc.) to find similar tasks
    in a performance database and create informed priors for hyperopt.
    """


    
    def __init__(
        self,
        csv_path: str = 'results/task_feature_summary.csv',
        pretrained_task_list=PRETRAINED_TASK_LIST,
        eval_task_list=EVAL_TASK_LIST,
        projection_function_g: bool = False,
        top_trials: int = 10,
        parallel_trials: int = 1,
        parallel_devices: Optional[Sequence[Union[str, int]]] = None,
        meta_classifier: bool = False,
        skip_autotransfer: bool = False,
    ):
        """
        Initialize the HPO system.

        Args:
            csv_path: Path to the task feature summary CSV file.
            skip_autotransfer: Ignored (kept for backward compatibility).
        """
        self.csv_path = csv_path
        self.parallel_trials = max(1, int(parallel_trials or 1))
        self.parallel_devices = resolve_devices(parallel_devices)
        self.use_meta_classifier = meta_classifier
        heuristics_df, autotransfer_df, simple_df = load_heuristics_features(csv_path=self.csv_path)
        task_order = {task: i for i, task in enumerate(ALL_TASK_LIST)}
        metric_to_task_type = {
            'roc_auc': 'classification',
            'mae': 'regression',
            'map': 'classification',
            'accuracy': 'classification',
        }

        # sort task order and annotate classification vs regression for each feature bank
        df_map = {
            'heuristics_df': heuristics_df,
            'autotransfer_df': autotransfer_df,
            'simple_df': simple_df,
        }
        for name, df in df_map.items():
            df['task_order'] = df['task_name'].map(task_order)
            df['task_type'] = (
                df['task_name']
                .map(task_name_to_metric_no_prefix)
                .map(lambda metric: metric_to_task_type.get(metric, 'classification'))
            )
            df_map[name] = df.sort_values('task_order').drop('task_order', axis=1).reset_index(drop=True)

        self.heuristics_df = df_map['heuristics_df']
        self.autotransfer_df = df_map['autotransfer_df']
        self.simple_df = df_map['simple_df']
            

        self.pretrained_task_list = pretrained_task_list
        self.eval_task_list = eval_task_list
        self.embedding_bank = None
        self.projection_function_g = projection_function_g
        # cache for pre-trained projection function g and aligned bank embeddings
        self.g_model = None
        self.g_bank_embeddings = None  # numpy array aligned to self.embedding_bank rows
        self.current_feature_banks = None
        self.top_trials = top_trials
        ## all trials is used to experiment 1
        ## it can't be used for experiment 2/3 since there will be leakage
        self.all_trials = load_validated_runs(entity=os.getenv("WANDB_ENTITY", "msuczk"), refresh=False)
        
        ## filter the trials
        self.all_trials['pre_sf'] = self.all_trials['pre_sf'].apply(lambda x: 'zero_learn' if x == 'src_dst' else x)
        self.all_trials = self.all_trials[self.all_trials['dfs_layer'] != 3]
        self.all_trials = self.all_trials[self.all_trials['task_type'] != 'recommendation']

        ## remove the column name test_rows from heuristics_df
        heuristics_df = heuristics_df.drop(columns=['test_rows'])

        # Ensure a baseline features table exists for later lookups.
        # Use validated_runs.csv by default if external aggregator is not configured.
        self.full_features = self.all_trials.copy()

        # Precompute similarity among ONLY pretrained tasks to avoid leakage
        bank_task_sim, _, _, _ = get_anchor_models_and_task_embedding(
            self.all_trials[self.all_trials['task_name'].isin(self.pretrained_task_list)]
        )
        self.bank_task_sim = bank_task_sim
        # Meta-predictor cache for Experiment 3
        self.meta_clf = None
        self.meta_scaler = None
        self.meta_numeric_cols = None
        self.meta_normalize = False
        self.meta_budget_aware = False
        self.meta_budget_feature_bank = 'homophily'
        self.allowed_dfs_models = ['ft_transformer', 'tabpfn']

        basic_config = load_yaml('./configs/default/task.yaml')
        self.landscape_db = create_swap_db(
            backend=basic_config.get("db_backend", "sqlite"),
            path_to_db=basic_config["db_location"],
            collection_name="loss_metrics",
            db_name=basic_config["db_name"],
        )
        self.meta_classifier_type = 'tabpfn'



    # ----------------------------
    # Meta-predictor (RDL vs DFS, labeling trick vs labeling trick,  dfs layer -> trivial)
    # ----------------------------
    def train_meta_predictor(self, need_normalization=True,
                              budget_aware=False, budgets=(3, 10, 30, 60), samples_per_budget=20,
                              budget_feature_bank='heuristic', min_trials_per_branch=3):
        """Fit the TabPFN-based meta predictor.

        Args:
            budget_aware: When True, learn from the budget-aware table produced by
                :meth:`build_budget_aware_predictor_dataset`.
            budgets: Budgets passed to the budget-aware table builder.
            samples_per_budget: Monte Carlo samples per budget.
            budget_feature_bank: Feature bank to attach to the budget-aware table.
        """
        from sklearn.preprocessing import StandardScaler

        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError("TabPFN is not installed. Install `tabpfn` to use the TabPFN meta classifier.") from exc

        scaler = StandardScaler()

        if budget_aware:
            dataset = self.build_budget_aware_predictor_dataset(
                budgets=budgets,
                samples_per_budget=samples_per_budget,
                feature_bank=budget_feature_bank,
                need_normalization=need_normalization,
                min_trials_per_branch=min_trials_per_branch,
            )
            dataset = dataset[dataset['preferred_branch'].isin(['rdl', 'dfs'])].reset_index(drop=True)
            if dataset.empty:
                raise ValueError("Budget-aware dataset is empty; cannot train meta predictor")

            drop_numeric = {
                'preferred_branch', 'samples_used', 'available_rdl_trials', 'available_dfs_trials',
                'rdl_win_probability', 'dfs_win_probability', 'prob_diff', 'ties'
            }
            numeric_cols = [
                col for col in dataset.columns
                if col not in drop_numeric and pd.api.types.is_numeric_dtype(dataset[col])
            ]
            if 'budget' not in numeric_cols and 'budget' in dataset:
                numeric_cols.append('budget')
            if not numeric_cols:
                raise ValueError("No numeric features available for budget-aware meta predictor")

            X = dataset[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            y = (dataset['preferred_branch'] == 'rdl').astype(int).to_numpy(dtype=int)
            self.meta_budget_aware = True
            self.meta_budget_feature_bank = budget_feature_bank
        else:
            feat_df = self.extract_full_features('homophily', need_normalization=need_normalization)
            feat_df = feat_df[feat_df['task_name'].isin(self.pretrained_task_list)].reset_index(drop=True)
            if feat_df.empty:
                raise ValueError("No homophily features available for PRETRAINED_TASK_LIST")

            perf = self._extract_task_performance(skip_nonft=True)
            tasks_in_feat = set(feat_df['task_name'].unique())
            tasks = sorted([t for t in perf.keys() if t in tasks_in_feat])
            if len(tasks) < 3:
                raise ValueError("Too few tasks to train meta-predictor")

            numeric_cols = [c for c in feat_df.columns if c not in ['dataset_name', 'task_name']]
            X_list, y_list = [], []
            for t in tasks:
                r = feat_df[feat_df['task_name'] == t].iloc[0]
                x = r[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                if perf[t]['task_type'] == 'regression':
                    label = 1 if perf[t]['rdl_best'] < perf[t]['dfs_best'] else 0
                else:
                    label = 1 if perf[t]['rdl_best'] > perf[t]['dfs_best'] else 0
                X_list.append(x)
                y_list.append(label)
            X = np.asarray(X_list, dtype=float)
            y = np.asarray(y_list, dtype=int)
            self.meta_budget_aware = False
            self.meta_budget_feature_bank = 'homophily'

        if X.shape[0] < 2:
            raise ValueError("Insufficient samples to train the meta predictor")

        X_train = scaler.fit_transform(X) if need_normalization else X
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clf = TabPFNClassifier(device=device)
        clf.fit(np.asarray(X_train, dtype=np.float32), np.asarray(y, dtype=np.int64))

        self.meta_classifier_type = 'tabpfn'
        self.meta_clf = clf
        self.meta_scaler = scaler if need_normalization else None
        self.meta_numeric_cols = numeric_cols
        self.meta_normalize = bool(need_normalization)

    def predict_family(self, database_name, task_name, threshold=0.8, budget=30,
                       budget_aware=False, budgets=(3, 10, 30, 60), samples_per_budget=20,
                       min_trials_per_branch=3, budget_feature_bank='homophily'):
        """Return {'p_rdl','p_dfs','decision'} where decision in {'rdl','dfs',None}."""
        self.train_meta_predictor(
            need_normalization=False,
            budget_aware=budget_aware,
            budgets=budgets,
            samples_per_budget=samples_per_budget,
            min_trials_per_branch=min_trials_per_branch,
            budget_feature_bank=budget_feature_bank,
        )

        if self.meta_budget_aware:
            feature_bank = self.meta_budget_feature_bank
        else:
            feature_bank = 'homophily'

        feat_df = self.extract_full_features(feature_bank, need_normalization=False)
        row = feat_df[(feat_df['dataset_name'] == database_name) & (feat_df['task_name'] == task_name)]
        if row.empty:
            return {'p_rdl': 0.5, 'p_dfs': 0.5, 'decision': None}

        base_row = row.iloc[0]
        values = []
        for col in self.meta_numeric_cols:
            if col == 'budget':
                values.append(float(budget))
            elif col in base_row.index:
                values.append(float(base_row[col]))
            else:
                values.append(0.0)
        x = np.asarray([values], dtype=float)
        if self.meta_normalize and (self.meta_scaler is not None):
            x = self.meta_scaler.transform(x)

        x_np = x.astype(np.float32)
        proba = np.asarray(self.meta_clf.predict_proba(x_np)[0], dtype=float)
        classes = getattr(self.meta_clf, 'classes_', np.array([0, 1]))

        if len(proba) == 1:
            p_rdl = float(proba[0])
            p_dfs = 1.0 - p_rdl
        else:
            if 1 in classes:
                idx1 = int(np.where(classes == 1)[0][0])
                p_rdl = float(proba[idx1])
                if 0 in classes:
                    idx0 = int(np.where(classes == 0)[0][0])
                    p_dfs = float(proba[idx0])
                else:
                    p_dfs = 1.0 - p_rdl
            else:
                p_rdl = float(proba[-1])
                p_dfs = float(proba[0]) if len(proba) > 1 else 1.0 - p_rdl
        decide = None
        if p_rdl >= threshold:
            decide = 'rdl'
        elif p_dfs >= threshold:
            decide = 'dfs'
        return {'p_rdl': p_rdl, 'p_dfs': p_dfs, 'decision': decide}

    def pretrain_projection_function_g(self, other_feats_df, bank_task_sim):
        """
        Pretrain projection function g using DataFrame input.
        
        Args:
            other_feats_df: DataFrame with columns ['dataset_name', 'task_name', ...features...]
            
        Returns:
            tuple: (trained_model, embeddings_tensor)
        """
        df = other_feats_df.copy()
        df['task_key'] = df['dataset_name'].astype(str) + '::' + df['task_name'].astype(str)
        df = df.set_index('task_key')

        common = df.index.intersection(bank_task_sim.index)
        if len(common) < 3:
            raise ValueError("Too few common tasks between features and sim matrix")

        df = df.loc[common]
        sim = bank_task_sim.loc[common, common].values   

        feature_cols = [c for c in df.columns if c not in ['dataset_name','task_name',]]
        X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        X = X / (X.norm(p=2, dim=1, keepdim=True) + 1e-12)

        input_dim = X.shape[1]
        model = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 16))
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
        rank_loss = nn.MarginRankingLoss(margin=0.1)

        N = X.shape[0]
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            Z = model(X)
            Z = Z / (Z.norm(p=2, dim=1, keepdim=True) + 1e-12)

            triples = []
            for _ in range(256):
                u, i, j = np.random.randint(0, N, 3)
                if u == i or u == j or i == j:
                    continue
                if sim[u, i] < sim[u, j]:
                    i, j = j, i
                triples.append((u, i, j))
            idx = np.array(triples, dtype=np.int64)
            ui = (Z[idx[:,0]] * Z[idx[:,1]]).sum(-1)
            uj = (Z[idx[:,0]] * Z[idx[:,2]]).sum(-1)
            loss = rank_loss(ui, uj, torch.ones_like(ui))
            loss.backward()
            optimizer.step()
            if epoch % 200 == 0:
                print(f"Loss: {loss.item():.4f}")

        with torch.no_grad():
            Z = model(X)
            Z = Z / (Z.norm(p=2, dim=1, keepdim=True) + 1e-12)

        emb = pd.DataFrame(Z.cpu().numpy(),
                        index=common,
                        columns=[f'embedding_{i}' for i in range(Z.shape[1])])
        return model.cpu(), emb
    
    def _tensor_to_dataframe(self, embedding_tensor, original_df):
        """
        Convert tensor embeddings back to DataFrame format for similarity calculation.

        Args:
            embedding_tensor: Tensor of shape (n_tasks, embedding_dim)
            original_df: Original DataFrame with dataset_name and task_name columns

        Returns:
            pd.DataFrame: DataFrame with dataset_name, task_name, and embedding features
        """
        # Convert tensor to numpy array
        embedding_array = embedding_tensor.numpy()
        # Create new DataFrame with embedding features
        embedding_cols = [f'embedding_{i}' for i in range(embedding_array.shape[1])]
        embedding_df = pd.DataFrame(embedding_array, columns=embedding_cols)
        # Add dataset_name and task_name from original DataFrame
        embedding_df['dataset_name'] = original_df['dataset_name'].values
        embedding_df['task_name'] = original_df['task_name'].values
        # Reorder columns to have dataset_name and task_name first
        embedding_df = embedding_df[['dataset_name', 'task_name'] + embedding_cols]
        return embedding_df

    def _compute_loo_similarity_with_g(self, combined_df, kendall_sim, rank_mat_sim):
        """Compute similarity matrix using LOO-trained g projections."""
        df = combined_df.copy()
        df['task_key'] = df['dataset_name'].astype(str) + '::' + df['task_name'].astype(str)
        df = df[df['task_key'].isin(kendall_sim.index)]
        if df.empty:
            logger.warning("No overlapping tasks between features and kendall_sim for LOO g training")
            return None

        feature_cols = [c for c in df.columns if c not in ['dataset_name', 'task_name', 'task_key']]
        if not feature_cols:
            logger.warning("No feature columns available for LOO g projection")
            return None

        loo_rows: Dict[str, pd.Series] = {}

        for task_key, target_row in tqdm(df.groupby('task_key'), desc="Computing LOO similarity with g"):
            train_df = df[df['task_key'] != task_key]
            train_keys = train_df['task_key'].tolist()
            if len(train_keys) < 3:
                logger.debug(f"Skipping LOO g training for {task_key}: insufficient training tasks")
                continue

            bank_sim = kendall_sim.loc[train_keys, train_keys]
            try:
                model, emb_df = self.pretrain_projection_function_g(
                    train_df.drop(columns=['task_key']),
                    bank_sim
                )
            except ValueError as exc:
                logger.debug(f"Failed LOO g training for {task_key}: {exc}")
                continue

            emb_df = emb_df.loc[train_keys]
            bank_embeddings = emb_df.values
            bank_embeddings = bank_embeddings / (np.linalg.norm(bank_embeddings, axis=1, keepdims=True) + 1e-12)

            target_feats = target_row[feature_cols].values.astype(np.float32)
            target_feats = target_feats / (np.linalg.norm(target_feats, axis=1, keepdims=True) + 1e-12)

            with torch.no_grad():
                target_embed = model(torch.from_numpy(target_feats))
                target_embed = target_embed / (target_embed.norm(p=2, dim=1, keepdim=True) + 1e-12)
            target_embed = target_embed.cpu().numpy()

            sims = (target_embed @ bank_embeddings.T).ravel()
            loo_rows[task_key] = pd.Series(sims, index=emb_df.index, dtype=float)

        if not loo_rows:
            logger.warning("LOO g training produced no similarity rows")
            return None

        aligned_tasks = rank_mat_sim.index
        loo_matrix = pd.DataFrame(np.nan, index=aligned_tasks, columns=aligned_tasks, dtype=float)

        for task_key, sims in loo_rows.items():
            loo_matrix.loc[task_key, task_key] = 1.0
            common_idx = sims.index.intersection(loo_matrix.columns)
            loo_matrix.loc[task_key, common_idx] = sims.loc[common_idx].values

        for i in aligned_tasks:
            for j in aligned_tasks:
                if i == j:
                    if pd.isna(loo_matrix.loc[i, j]):
                        loo_matrix.loc[i, j] = 1.0
                    continue
                vij = loo_matrix.loc[i, j]
                vji = loo_matrix.loc[j, i]
                if pd.isna(vij) and pd.isna(vji):
                    continue
                if pd.isna(vij):
                    loo_matrix.loc[i, j] = vji
                elif pd.isna(vji):
                    loo_matrix.loc[j, i] = vij
                else:
                    avg = 0.5 * (vij + vji)
                    loo_matrix.loc[i, j] = avg
                    loo_matrix.loc[j, i] = avg

        return loo_matrix

    def _get_similarity_matrices(self, feature_banks, kendall_sim, rank_mat_sim, need_normalization=False):
        """Helper method to get similarity matrices without plotting"""
        sim_matrices, sim_names = [], []
        
        # Concatenate all features from feature_banks
        combined_features_list = []
        for embedding_type in feature_banks:
            base_df = self.extract_full_features(embedding_type, need_normalization)
            base_df = base_df[base_df['task_name'].isin(self.all_trials['task_name'].unique())]
            if base_df is None or base_df.empty:
                continue
            
            # Extract only the feature columns (exclude dataset_name and task_name for now)
            feature_cols = [col for col in base_df.columns if col not in ['dataset_name', 'task_name', 'train_rows', 'test_rows','val_rows', 'task_type']]
            if feature_cols:
                # Add prefix to distinguish features from different banks
                feature_df = base_df[['dataset_name', 'task_name'] + feature_cols].copy()
                feature_df.columns = ['dataset_name', 'task_name'] + [f"{embedding_type}_{col}" for col in feature_cols]
                combined_features_list.append(feature_df)
        
        if not combined_features_list:
            logger.warning("No valid features found in feature_banks")
            return [], []
            
        # Merge all feature dataframes on dataset_name and task_name
        combined_df = combined_features_list[0]
        for feature_df in combined_features_list[1:]:
            combined_df = combined_df.merge(feature_df, on=['dataset_name', 'task_name'], how='outer')

        # Process the combined features with and without g projection
        for need_g in [True, False]:
            if need_g:
                sim_matrice = self._compute_loo_similarity_with_g(combined_df, kendall_sim, rank_mat_sim)
                if sim_matrice is None:
                    continue
            else:
                embedding_df = combined_df

            feature_banks_str = '+'.join(feature_banks)
            logger.info(f"Combined features: {feature_banks_str}, Need g: {need_g}")
            sim_names.append(f"combined_{feature_banks_str}" + (' + g' if need_g else ''))
            if need_g:
                sim_matrices.append(sim_matrice)
            else:
                sim_matrice = self.compute_embedding_similarity_for_ranking(embedding_df, rank_mat_sim, similarity_method='cosine')
                sim_matrices.append(sim_matrice)

        return sim_matrices, sim_names

    def graphgym_similarity_all_combinations(self, feature_combinations, anchor_mode_type='common'):
        """Run graphgym similarity for all feature bank combinations and plot them together"""
        # Collect all similarity matrices and names for combined plotting
        all_sim_matrices = []
        all_sim_names = []
        # Get the base rank similarity matrix once
        kendall_sim, stacked, rank_mat, _ = get_anchor_models_and_task_embedding(
            self.all_trials,
            anchor_mode_type=anchor_mode_type,
        )
        kendall_sim = kendall_sim.fillna(0)
        rank_mat_sim = _task_sim_from_anchor_ranks(rank_mat)
        for norm in [True,False]:
            norm_suffix = "_norm" if norm else "_no_norm"
            # Define all feature bank combinations
            for feature_banks in feature_combinations:
                # Get similarity matrices for this combination
                sim_matrices, sim_names = self._get_similarity_matrices(feature_banks, kendall_sim, rank_mat_sim, norm)
                # Add normalization suffix to names
                sim_names = [name + norm_suffix for name in sim_names]
                all_sim_matrices.extend(sim_matrices)
                all_sim_names.extend(sim_names)
        # Plot all results together
        result = plot_combined_similarity_ranking_correlation(rank_mat_sim, all_sim_matrices, all_sim_names, title=anchor_mode_type)

        # ---------------- Debug Similarity Neighbours -----------------
        debug_lines = []

        def _top_k_lines(name: str, sim_df: pd.DataFrame, reference_index: pd.Index) -> list[str]:
            if sim_df is None or sim_df.empty:
                return [f"=== {name}: empty matrix ==="]
            common = reference_index.intersection(sim_df.index)
            if common.empty:
                return [f"=== {name}: no overlapping tasks ==="]
            sub = sim_df.loc[common, common].copy()
            lines = [f"=== {name} ==="]
            for task in common:
                row = sub.loc[task].drop(labels=[task], errors='ignore')
                row = row.dropna().sort_values(ascending=False)
                top = row.head(3)
                if top.empty:
                    neighbour_str = "<no neighbours>"
                else:
                    neighbour_str = ", ".join(f"{idx}: {val:.4f}" for idx, val in top.items())
                lines.append(f"  {task}: {neighbour_str}")
            return lines

        task_index = kendall_sim.index
        debug_lines.extend(_top_k_lines("Ground Truth (kendall_sim)", kendall_sim, task_index))

        for name, sim in zip(all_sim_names, all_sim_matrices):
            debug_lines.extend(_top_k_lines(name, sim, task_index))

        debug_path = "similarity_debug"
        try:
            with open(debug_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(debug_lines))
        except OSError as exc:
            logger.warning(f"Failed to write similarity debug file '{debug_path}': {exc}")

        return result

    def sim_no_g(self, target_features, embedding_bank):
        """
        Return TRUE cosine similarity (higher = more similar).
        """
        x = target_features.select_dtypes(include=['number']).values
        bank = embedding_bank.select_dtypes(include=['number']).values
        x = torch.tensor(x, dtype=torch.float32)
        bank = torch.tensor(bank, dtype=torch.float32)
        # l2 normalize
        x = x / torch.norm(x, 2, 1, keepdim=True)
        bank = bank / torch.norm(bank, 2, 1, keepdim=True)
        sims = (x * bank).sum(-1).numpy()  # in [-1, 1]
        self.emb_sims = sims
        return sims
    
    def calculate_task_similarity_matrix(self, embedding_df, similarity_method='cosine'):
        """
        Calculate task-wise similarity matrix based on embedding dataframe.
        
        Args:
            embedding_df: DataFrame with columns ['dataset_name', 'task_name', ...features...]
            similarity_method: 'cosine' or 'euclidean'
            
        Returns:
            pd.DataFrame: NxN symmetric similarity matrix with task keys as index/columns
                          in format 'dataset_name::task_name'
        """        
        # Create task keys in the same format as rank_mat_sim
        embedding_df = embedding_df.copy()
        embedding_df['task_key'] = embedding_df['dataset_name'].astype(str) + '::' + embedding_df['task_name'].astype(str)
        
        # Extract numerical features (exclude dataset_name, task_name, task_key)
        feature_cols = [col for col in embedding_df.columns 
                       if col not in ['dataset_name', 'task_name', 'task_key']]
        
        # Get feature matrix, fill NaN with 0
        X = embedding_df[feature_cols].fillna(0).values

        # Standardize features
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        X_scaled = X
        
        # Calculate similarity matrix
        if similarity_method == 'cosine':
            # Cosine similarity (higher = more similar, range [-1, 1])
            sim_matrix = cosine_similarity(X_scaled)
        elif similarity_method == 'euclidean':
            # Euclidean distance (lower = more similar, convert to similarity)
            dist_matrix = euclidean_distances(X_scaled)
            # Convert distance to similarity: sim = 1 / (1 + distance)
            sim_matrix = 1 / (1 + dist_matrix)
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")
        
        # Create DataFrame with task keys as index and columns
        task_keys = embedding_df['task_key'].tolist()
        sim_df = pd.DataFrame(sim_matrix, index=task_keys, columns=task_keys)
        
        return sim_df
    
    def align_similarity_matrix(self, embedding_sim_df, rank_mat_sim):
        """
        Align embedding similarity matrix to match the exact order and format of rank_mat_sim.
        
        Args:
            embedding_sim_df: Similarity matrix from calculate_task_similarity_matrix
            rank_mat_sim: Reference matrix to match format/order
            
        Returns:
            pd.DataFrame: Aligned similarity matrix with same index/columns as rank_mat_sim
        """
        # Get the intersection of task keys
        embedding_tasks = set(embedding_sim_df.index)
        rank_tasks = set(rank_mat_sim.index)
        common_tasks = embedding_tasks.intersection(rank_tasks)
        
        if len(common_tasks) == 0:
            raise ValueError("No common tasks found between embedding and rank matrices")
        
        # Create aligned matrix with same order as rank_mat_sim
        aligned_sim_df = pd.DataFrame(index=rank_mat_sim.index, columns=rank_mat_sim.columns)
        
        # Fill in similarities for common tasks
        for task_i in rank_mat_sim.index:
            for task_j in rank_mat_sim.columns:
                if task_i in common_tasks and task_j in common_tasks:
                    aligned_sim_df.loc[task_i, task_j] = embedding_sim_df.loc[task_i, task_j]
                else:
                    # For tasks not in embedding, set to NaN or 0
                    aligned_sim_df.loc[task_i, task_j] = np.nan
        
        return aligned_sim_df
    
    def compute_embedding_similarity_for_ranking(self, embedding_df, rank_mat_sim, similarity_method='cosine'):
        """
        Complete pipeline to compute embedding-based similarity matrix aligned with rank_mat_sim.
        
        Args:
            embedding_df: DataFrame with columns ['db_name', 'task_name', ...features...]
            rank_mat_sim: Reference ranking similarity matrix to match format
            similarity_method: 'cosine' or 'euclidean'
            
        Returns:
            pd.DataFrame: Aligned similarity matrix ready for comparison with rank_mat_sim
        """
        # Step 1: Calculate similarity matrix from embeddings
        embedding_sim_df = self.calculate_task_similarity_matrix(embedding_df, similarity_method)
        
        # Step 2: Align with rank_mat_sim format and order
        aligned_sim_df = self.align_similarity_matrix(embedding_sim_df, rank_mat_sim)
        
        return aligned_sim_df

    def _extract_task_performance(self, skip_nonft = False):
        """
        Extract the best performance for each model type (RDL vs DFS) per task.
        
        Returns:
            dict: {task_name: {'rdl_best': float, 'dfs_best': float, 'task_type': str}}
        """
        task_performance = {}

        all_trials = self.all_trials.copy()

        if skip_nonft:
            all_trials = all_trials[~all_trials['torch_frame_model_cls'].isin(['lgbm', 'tabpfn'])]

        for task_name in all_trials['task_name'].unique():
            task_data = all_trials[all_trials['task_name'] == task_name]

            # Get task type (for determining metric direction)
            task_type = task_data['task_type'].iloc[0] if 'task_type' in task_data.columns else 'binary_classification'
            
            # Determine if higher is better (classification) or lower is better (regression)
            ascending = task_type == 'regression'
            
            # Extract RDL performance
            rdl_data = task_data[task_data['model_type'] == 'rdl']
            rdl_best = None
            if len(rdl_data) > 0 and 'test_metric' in rdl_data.columns:
                rdl_best = rdl_data.sort_values('val_metric', ascending=ascending)['test_metric'].iloc[0]
            
            # Extract DFS performance  
            dfs_data = task_data[(task_data['model_type'] == 'dfs')]
            dfs_best = None
            if len(dfs_data) > 0 and 'test_metric' in dfs_data.columns:
                dfs_best = dfs_data.sort_values('val_metric', ascending=ascending)['test_metric'].iloc[0]
            
            # Only include tasks where both models have results
            if rdl_best is not None and dfs_best is not None:
                task_performance[task_name] = {
                    'rdl_best': rdl_best,
                    'dfs_best': dfs_best,
                    'task_type': task_type
                }
        
        logger.info(f"Extracted performance for {len(task_performance)} tasks")
        return task_performance



    
    def _create_prediction_dataset(self, task_performance, features):
        """
        Build a supervised dataset for LOO: task-level features + label (RDL win vs DFS win).
        Returns:
            X:           (n_tasks, n_features) numeric array (not used directly in new flow)
            y:           (n_tasks,) labels (1=RDL wins, 0=DFS wins)
            task_names:  list[str]
            task_keys:   list[str] "dataset::task", used for strict alignment
        """
        feat_df = features.copy()
        if 'task_key' not in feat_df.columns:
            feat_df['task_key'] = feat_df['dataset_name'].astype(str) + '::' + feat_df['task_name'].astype(str)

        X_list, y_list, task_names, task_keys = [], [], [], []
        numeric_cols = [c for c in feat_df.columns if c not in ['dataset_name', 'task_name', 'task_key']]

        # Create task order mapping from ALL_TASK_LIST
        task_order_map = {task: i for i, task in enumerate(ALL_TASK_LIST)}
        
        # Sort task_performance items by ALL_TASK_LIST order
        sorted_tasks = sorted(task_performance.items(), key=lambda x: task_order_map.get(x[0], float('inf')))

        for tname, perf in sorted_tasks:
            row = feat_df.loc[feat_df['task_name'] == tname]
            if row.empty:
                logger.warning(f"No embedding row found for task {tname}")
                continue

            row = row.iloc[0]
            x = row[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

            # Label: classification => higher better; regression => lower better
            if perf['task_type'] == 'regression':
                winner = 1 if perf['rdl_best'] < perf['dfs_best'] else 0
            else:
                winner = 1 if perf['rdl_best'] > perf['dfs_best'] else 0

            X_list.append(x)
            y_list.append(winner)
            task_names.append(tname)
            task_keys.append(f"{row['dataset_name']}::{row['task_name']}")

        if not X_list:
            return None, None, None, None

        X = np.asarray(X_list, dtype=float)
        y = np.asarray(y_list, dtype=int)
        return X, y, task_names, task_keys


    def _fit_predict_meta_classifier(self, classifier, X_train, y_train, X_test):
        """Utility to fit a binary classifier and return prediction + probabilities."""
        classifier_name = classifier.__class__.__name__.lower()

        if 'tabpfn' in classifier_name:
            X_train_np = np.asarray(X_train, dtype=np.float32)
            X_test_np = np.asarray(X_test, dtype=np.float32)
            y_train_np = np.asarray(y_train, dtype=np.int64)
            classifier.fit(X_train_np, y_train_np)
            proba = np.asarray(classifier.predict_proba(X_test_np)[0], dtype=float)
            pred_raw = classifier.predict(X_test_np)[0]
        else:
            classifier.fit(X_train, y_train)
            proba = np.asarray(classifier.predict_proba(X_test)[0], dtype=float)
            pred_raw = classifier.predict(X_test)[0]

        classes = getattr(classifier, 'classes_', None)
        if proba.ndim == 0:
            proba = np.array([float(proba)], dtype=float)

        if len(proba) == 1:
            p_rdl = float(proba[0])
            p_dfs = 1.0 - p_rdl
        else:
            if classes is not None and 1 in classes:
                idx1 = int(np.where(classes == 1)[0][0])
                p_rdl = float(proba[idx1])
                if 0 in classes:
                    idx0 = int(np.where(classes == 0)[0][0])
                    p_dfs = float(proba[idx0])
                else:
                    p_dfs = 1.0 - p_rdl
            else:
                p_rdl = float(proba[-1])
                p_dfs = float(proba[0]) if len(proba) > 1 else 1.0 - p_rdl

        try:
            pred = int(pred_raw)
        except Exception:
            pred = int(np.argmax(proba)) if len(proba) > 1 else int(round(float(proba[0])))

        return pred, p_rdl, p_dfs

    def _perform_loo_classification(self, X, y, task_names, need_g, classifiers=None):
        """
        Perform leave-one-out cross-validation classification.
        
        Args:
            X: Feature matrix
            y: Binary labels
            task_names: List of task names
            
        Returns:
            dict: Results containing accuracy, predictions, and detailed results
        """
        from sklearn.metrics import accuracy_score, classification_report
        
        n_samples = len(X)
        predictions = []
        true_labels = []
        task_results = []
        
        if classifiers is None:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            classifiers = {
                'RandomForest': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': lambda: LogisticRegression(random_state=42, max_iter=1000)
            }
        
        results = {}
        
        for clf_name, classifier_class in classifiers.items():
            clf_predictions = []
            clf_true_labels = []
            clf_task_results = []
            
            for i in range(n_samples):
                # Reinitialize classifier for each LOO split
                classifier = classifier_class()
                
                # Create train/test split (leave one out)
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i, axis=0)
                X_test = X[i:i+1]
                y_test = y[i]

                # Note: g-projection requires strict alignment between features and similarity matrix.
                # In this LOO classifier, we fall back to raw features to avoid misalignment.
                X_train_g = X_train
                X_test = X_test
                
                # Train and predict
                pred, p_rdl, p_dfs = self._fit_predict_meta_classifier(
                    classifier, X_train_g, y_train, X_test
                )
                pred_proba = np.asarray([p_dfs, p_rdl], dtype=float)
                
                clf_predictions.append(pred)
                clf_true_labels.append(y_test)
                clf_task_results.append({
                    'task_name': task_names[i],
                    'true_label': y_test,
                    'predicted_label': pred,
                    'rdl_probability': float(pred_proba[1]),
                    'dfs_probability': float(pred_proba[0]),
                    'correct': pred == y_test
                })
            
            # Calculate accuracy
            accuracy = accuracy_score(clf_true_labels, clf_predictions)
            
            results[clf_name] = {
                'accuracy': accuracy,
                'predictions': clf_predictions,
                'true_labels': clf_true_labels,
                'task_results': clf_task_results,
                'classification_report': classification_report(clf_true_labels, clf_predictions, 
                                                             target_names=['DFS_Winner', 'RDL_Winner'], zero_division=0)
            }
            
            logger.info(f"{clf_name} LOO Accuracy: {accuracy:.3f}")
        
        # Add summary statistics
        summary = {
            'n_tasks': n_samples,
            'n_rdl_winners': int(np.sum(y)),
            'n_dfs_winners': int(len(y) - np.sum(y)),
        }
        best_name = None
        best_acc = -np.inf
        for clf_name in classifiers.keys():
            acc = results[clf_name]['accuracy']
            key = f"accuracy_{clf_name.lower()}"
            summary[key] = acc
            if acc > best_acc:
                best_acc = acc
                best_name = clf_name
        summary['best_classifier'] = best_name
        results['summary'] = summary
        
        return results
 

    def compute_task_similarity(self, target_features):
        if self.embedding_bank is None or len(self.embedding_bank) == 0:
            return np.array([])

        if not self.projection_function_g:
            return self.sim_no_g(target_features, self.embedding_bank)
        
        if self.projection_function_g and (self.bank_task_sim is None or len(self.bank_task_sim) == 0):
            raise ValueError("bank_task_sim is empty; call setup_embedding_bank_from_banks(...) first.")

        # Train g only on PRETRAINED_TASK_LIST-aligned bank once per bank setup
        if (self.g_model is None) or (self.g_bank_embeddings is None):
            model, bank_emb = self.pretrain_projection_function_g(self.embedding_bank, self.bank_task_sim)
            # Reorder bank_emb to align with self.embedding_bank rows
            tmp = bank_emb.reset_index().rename(columns={'index': 'task_key'})
            # Above split can be done with pandas; fallback to merge on task_key directly
            try:
                tmp[['dataset_name', 'task_name']] = tmp['task_key'].astype(str).str.split('::', n=1, expand=True)
                ordered = self.embedding_bank.copy()
                ordered['task_key'] = ordered['dataset_name'].astype(str) + '::' + ordered['task_name'].astype(str)
                ordered = ordered[['task_key']]
                bank_ordered = ordered.merge(tmp, on='task_key', how='left')
                B = bank_ordered[[c for c in bank_emb.columns if c.startswith('embedding_')]].values
            except Exception:
                # if any issue, fall back to raw order (indices should already match common)
                B = bank_emb.values
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
            self.g_model = model
            self.g_bank_embeddings = B
        else:
            model = self.g_model
            B = self.g_bank_embeddings
        
        logger.debug(f"target_features: {target_features}")

        feat_cols = [c for c in target_features.columns if c not in ['dataset_name','task_name']]
        x = target_features[feat_cols].values.astype(np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

        with torch.no_grad():
            z = model(torch.from_numpy(x))
            z = z / (z.norm(p=2, dim=1, keepdim=True) + 1e-12)
        z = z.cpu().numpy()

        sims = (z @ B.T).ravel()
        self.emb_sims = sims
        return sims

    def create_prior_distribution(self, target_features, config_space, top_k=2, weight_threshold=None, only_one = "", use_entire_bank=False):
        """
        Build priors for:
        - model type (rdl vs dfs)
        - per-parameter discrete distributions inside each branch
        NOTE: per-parameter priors are *conditional* and do NOT multiply model-type probs.
        """
        if self.embedding_bank is None:
            raise ValueError("Embedding bank not set up")

        if use_entire_bank:
            idx = np.arange(len(self.embedding_bank))
            if idx.size == 0:
                return {}
            sim_top = np.ones_like(idx, dtype=float)
        else:
            sims = self.compute_task_similarity(target_features)
            if sims.size == 0:
                return {}

            # Take top-k *most similar* (largest sim)
            idx = np.argsort(-sims)[:top_k]
            sim_top = sims[idx].astype(float)
        
        # Debug: Log the top similar tasks
        logger.debug(f"Top {top_k} similar tasks found:")
        for j, i in enumerate(idx):
            task_info = self.embedding_bank.iloc[i]
            logger.debug(f"  {j+1}. {task_info['dataset_name']}/{task_info['task_name']} (similarity: {sim_top[j]:.4f})")

        # Optional thresholding
        if weight_threshold is not None:
            mask = sim_top >= weight_threshold
            if not np.any(mask):
                logger.debug(f"No tasks meet similarity threshold {weight_threshold}")
                return {}
            idx = idx[mask]
            sim_top = sim_top[mask]
            logger.debug(f"After thresholding: {len(idx)} tasks remain")

        # Normalize weights (shift for non-negativity)
        if use_entire_bank:
            weights = np.ones_like(sim_top, dtype=float) / len(sim_top)
        else:
            sim_top = sim_top - sim_top.min()
            if sim_top.sum() <= 0:
                weights = np.ones_like(sim_top) / len(sim_top)
            else:
                weights = sim_top / sim_top.sum()

        # Accumulators
        model_type_items = []     # per similar task: {'rdl_prob', 'dfs_prob', 'weight'}
        rdl_bucket = []           # per similar task: {param: {value: prob}}
        dfs_bucket = []

        for w, i in zip(weights, idx):
            row = self.embedding_bank.iloc[i]
            database_name, task_name = row['dataset_name'], row['task_name']

            # pull trials
            df = self.full_features
            m = (df['dataset_name'] == database_name) & (df['task_name'] == task_name)
            trials_rdl = df[m & (df['model_type'] == 'rdl')].rename(columns=rdl_column_name_mapping)
            # Treat DFS trials or ft_transformer as DFS family if present
            # Treat DFS family (native 'dfs') and ft_transformer/tabpfn from torch_frame as DFS
            cond_ft = (df['torch_frame_model_cls'] == 'ft_transformer') if 'torch_frame_model_cls' in df.columns else False
            # cond_tabpfn = (df['torch_frame_model_cls'] == 'tabpfn') if 'torch_frame_model_cls' in df.columns else False

            trials_dfs = df[m & ((df['model_type'] == 'dfs') | cond_ft)].rename(columns=ftt_column_name_mapping)
            if self.allowed_dfs_models and 'torch_frame_model_cls' in trials_dfs.columns:
                trials_dfs = trials_dfs[trials_dfs['torch_frame_model_cls'].isin(self.allowed_dfs_models)]

            if len(trials_rdl) == 0 and len(trials_dfs) == 0:
                continue

            # Determine sorting direction
            # regression: lower is better -> ascending=True
            # classification: higher is better -> ascending=False
            task_type = (trials_rdl['task_type'].iloc[0] if len(trials_rdl) else
                        trials_dfs['task_type'].iloc[0] if len(trials_dfs) else
                        'binary_classification')
            ascending = True if 'regression' in str(task_type) else False

            if only_one == "":
                all_trials = pd.concat([trials_rdl, trials_dfs], ignore_index=True)
            elif only_one == "rdl":
                all_trials = trials_rdl
            elif only_one == "dfs":
                all_trials = trials_dfs
            else:
                raise ValueError(f"Invalid only_one value: {only_one}")

            if 'test_metric' not in all_trials.columns:
                continue

            top = all_trials.sort_values(by='test_metric', ascending=ascending).head(self.top_trials)

            # per-task model type probabilities from top trials
            n = len(top)
            if n == 0:
                continue
            p_dfs = top[top['model_type'] == 'dfs'].shape[0] / n
            p_rdl = top[top['model_type'] == 'rdl'].shape[0] / n
            model_type_items.append({'rdl_prob': p_rdl, 'dfs_prob': p_dfs, 'weight': w})

            # per-param priors within each branch (frequencies inside top)
            # --- RDL ---
            pri_rdl = {}
            top_rdl = top[top['model_type'] == 'rdl']
            if len(top_rdl) > 0:
                for key in RDL_CONSIDERED_PARAMS:
                    if key in top_rdl.columns:
                        vc = top_rdl[key].value_counts()
                        pri_rdl[key] = (vc / vc.sum()).to_dict()
            # --- DFS ---
            pri_dfs = {}
            top_dfs = top[top['model_type'] == 'dfs']
            if len(top_dfs) > 0:
                for key in DFS_FTT_CONSIDERED_PARAMS:
                    if key in top_dfs.columns:
                        vc = top_dfs[key].value_counts()
                        pri_dfs[key] = (vc / vc.sum()).to_dict()

            rdl_bucket.append(pri_rdl)
            dfs_bucket.append(pri_dfs)

        # Aggregate model-type probabilities (weighted by similarity only)
        rdl_prob = 0.0
        dfs_prob = 0.0
        for item in model_type_items:
            rdl_prob += item['weight'] * item['rdl_prob']
            dfs_prob += item['weight'] * item['dfs_prob']
        s = rdl_prob + dfs_prob
        if s > 0:
            rdl_prob /= s
            dfs_prob /= s
        else:
            rdl_prob = dfs_prob = 0.5

        # Respect hard selection by overriding model-type probabilities
        if only_one == 'rdl':
            rdl_prob, dfs_prob = 1.0, 0.0
        elif only_one == 'dfs':
            rdl_prob, dfs_prob = 0.0, 1.0

        priors = {
            'model_type_rdl_prob': float(rdl_prob),
            'model_type_dfs_prob': float(dfs_prob),
        }

        # Aggregate per-parameter priors across similar tasks (weighted by similarity only)
        def agg(bucket, prefix):
            if not bucket:
                return
            # collect all keys
            all_keys = set()
            for d in bucket:
                all_keys |= set(d.keys())
            for k in all_keys:
                # collect all possible values for this param
                vals = set()
                for d in bucket:
                    if k in d:
                        vals |= set(d[k].keys())
                vals = list(vals)
                probs = np.zeros(len(vals), dtype=float)
                for w, d in zip(weights, bucket):
                    if k not in d:
                        continue
                    for j, v in enumerate(vals):
                        probs[j] += w * float(d[k].get(v, 0.0))
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    priors[f'{prefix}_{k}'] = dict(zip(vals, probs.tolist()))

        agg(rdl_bucket, 'rdl')
        agg(dfs_bucket, 'dfs')
        return priors

    def extract_full_features(self, task_embedding_type = 'heuristic', need_normalization = False):
        """
        Extract task features based on embedding type or feature group key.

        Args:
                - New feature group keys (from exp42.py): 'AT_All', 'AT_RDL', 'AT_DFS',
                                'All_Model_Based', 'Real_Entity', 'Guess', 'Combine'
            need_normalization: Whether to normalize numerical features

        Returns:
            DataFrame with task features
        """
        # New feature group keys from exp42.py - delegate to get_task_features_by_group
        exp42_feature_groups = {'AT_All', 'AT_RDL', 'AT_DFS', 'All_Model_Based',
                                'Real_Entity', 'Guess', 'Combine'}

        if task_embedding_type in exp42_feature_groups:
            # Use the dedicated function for exp42 feature groups
            features = self.get_task_features_by_group(task_embedding_type, need_normalization=need_normalization)
            return features

        raise ValueError(f"Invalid task embedding type: {task_embedding_type}")
    
        # ---------- NEW: build a combined feature dataframe from multiple banks ----------
    def _build_combined_feature_df(self, feature_banks, need_normalization=True):
        """
        Merge multiple embedding/feature banks into a single dataframe
        with prefixed numeric columns.

        Returns:
            combined_df: columns ['dataset_name','task_name', <prefixed numeric features>]
        """
        combined_features_list = []
        for bank in feature_banks:
            base_df = self.extract_full_features(bank, need_normalization=need_normalization)
            if base_df is None or base_df.empty:
                continue
            # numeric columns only, keep identifiers
            num_cols = [c for c in base_df.columns if c not in ['dataset_name', 'task_name']]
            if not num_cols:
                continue
            tmp = base_df[['dataset_name', 'task_name'] + num_cols].copy()
            tmp.columns = ['dataset_name', 'task_name'] + [f"{bank}_{c}" for c in num_cols]
            combined_features_list.append(tmp)

        if not combined_features_list:
            raise ValueError(f"No valid features found for banks={feature_banks}")

        combined_df = combined_features_list[0]
        for tmp in combined_features_list[1:]:
            combined_df = combined_df.merge(tmp, on=['dataset_name','task_name'], how='outer')
        return combined_df

    def get_task_features_by_group(self, group_key: str, need_normalization: bool = False) -> pd.DataFrame:
        """
        Get task features for a specific feature group matching exp42.py design.

        Args:
            group_key: One of the feature group keys:
                - 'AT_All': All AutoTransfer features (RDL + DFS)
                - 'AT_RDL': AutoTransfer RDL features only
                - 'AT_DFS': AutoTransfer DFS features only
                - 'All_Model_Based': Model-based features (TabPFN, RandomSAGE, RandomNBFNet)
                - 'Real_Entity': Entity baseline features
                - 'Guess': Homophily + Autocorr + Size + Class Ratio features
                - 'Combine': Guess + Real_Entity (all model-free features)
            need_normalization: Whether to normalize the features

        Returns:
            DataFrame with columns: ['dataset_name', 'task_name', <feature_columns>]
        """
        # Get the full heuristics dataframe (contains all features)
        heuristics_df = self.heuristics_df.copy()
        autotransfer_df = self.autotransfer_df.copy()
        simple_df = self.simple_df.copy()

        # Define feature groups matching exp42.py
        homophily_adj_corr = ['h_adjs_corr_mean', 'h_adjs_corr_max', 'h_adjs_corr_min',
                               'h_adjs_corr_mode', 'h_adjs_corr_weighted_mean']

        autocorr_features = ['lag1_autocorr_corr', 'lag2_autocorr_corr']

        original_class_ratio = [
            'variance_same_class_ratio',
            'mean_same_class_ratio_ignore',
            'adjusted_mean_same_class_ratio',
            'sparsity_ratio',
            'mean_past_task_nodes'
        ]

        real_entity = ['entity_mean_val', 'entity_median_val', 'entity_mean_train', 'entity_median_train']

        # Model-based features
        tabpfn_features = ['tabpfn_1hop', 'tabpfn_2hop',
                          'rfr_randomsage_1', 'rfr_randomsage_2', 'rfr_randomsage_3',
                          'rfr_randomnbfnet_1', 'rfr_randomnbfnet_2', 'rfr_randomnbfnet_3']

        # AutoTransfer features (from autotransfer_df)
        at_rdl_features = [col for col in autotransfer_df.columns if col.startswith('at_rdl_')]
        at_dfs_features = [col for col in autotransfer_df.columns if col.startswith('at_dfs_')]

        # Size features (computed on the fly)
        if 'train_rows' in heuristics_df.columns and 'val_rows' in heuristics_df.columns:
            heuristics_df['log_total_rows'] = np.log1p(heuristics_df['train_rows'] + heuristics_df['val_rows'])
        size_features = ['log_total_rows']

        # Define combined groups
        guess = homophily_adj_corr + autocorr_features + size_features + original_class_ratio
        combine = guess + real_entity

        # Map group keys to feature lists and source dataframes
        group_mapping = {
            'AT_All': (at_rdl_features + at_dfs_features, autotransfer_df),
            'AT_RDL': (at_rdl_features, autotransfer_df),
            'AT_DFS': (at_dfs_features, autotransfer_df),
            'All_Model_Based': (tabpfn_features, heuristics_df),
            'Real_Entity': (real_entity, simple_df),
            'Guess': (guess, heuristics_df),
            # 'Combine': (combine, heuristics_df),
        }

        if group_key not in group_mapping:
            raise ValueError(
                f"Invalid group_key: {group_key}. "
                f"Must be one of: {list(group_mapping.keys())}"
            )

        feature_list, source_df = group_mapping[group_key]

        source_df.rename(columns={'db_name': 'dataset_name'}, inplace=True)

        # Filter to available features
        available_features = [f for f in feature_list if f in source_df.columns]

        if not available_features:
            raise ValueError(f"No features found for group '{group_key}' in the dataframe")

        # Extract the features
        id_cols = ['dataset_name', 'task_name']
        result_df = source_df[id_cols + available_features].copy()

        # Apply normalization if requested
        if need_normalization:
            for col in available_features:
                if col in result_df.columns:
                    mean_val = result_df[col].mean()
                    std_val = result_df[col].std()
                    result_df[col] = (result_df[col] - mean_val) / (std_val + 1e-8)

        return result_df.reset_index(drop=True)

    # ---------- NEW: set up embedding bank from one or more banks (+ optional g) ----------
    def setup_embedding_bank_from_banks(self, feature_banks, need_normalization=True, use_g=True):
        """
        Prepare self.embedding_bank and self.bank_task_sim on the *same* set of tasks.
        This prevents NaNs and misalignment when training/using the projection g.
        """
        # 1) Build combined features for all candidate tasks
        combined_df = self._build_combined_feature_df(feature_banks, need_normalization=need_normalization)

        # Keep only pretrained tasks (to avoid leakage), as before
        combined_df = combined_df[combined_df['task_name'].isin(self.pretrained_task_list)].reset_index(drop=True)

        # 2) Build task_key
        combined_df = combined_df.copy()
        combined_df['task_key'] = combined_df['dataset_name'].astype(str) + '::' + combined_df['task_name'].astype(str)

        # 3) Get trials restricted to the same keys (so Kendall sim is on the same tasks)
        trials = self.all_trials.copy()
        trials = trials.assign(task_key=trials['dataset_name'].astype(str) + '::' + trials['task_name'].astype(str))
        common_keys = sorted(set(combined_df['task_key']).intersection(trials['task_key']))

        if len(common_keys) < 3:
            raise ValueError(f"Too few common tasks to train g: found {len(common_keys)}")

        trials_aligned = trials[trials['task_key'].isin(common_keys)]
        kendall_sim, _, _, _ = get_anchor_models_and_task_embedding(trials_aligned)

        # 4) Align the bank and sim to exactly the same common_keys (same order)
        combined_df = combined_df[combined_df['task_key'].isin(common_keys)]
        combined_df = combined_df.set_index('task_key').loc[common_keys].reset_index()

        kendall_sim = kendall_sim.loc[common_keys, common_keys].fillna(0.0)

        # 5) Store aligned artifacts
        self.embedding_bank = combined_df.drop(columns=['task_key']).reset_index(drop=True)
        self.bank_task_sim = kendall_sim
        self.projection_function_g = bool(use_g)
        self.current_feature_banks = list(feature_banks)

        # Reset cached g now that the bank changed
        self.g_model = None
        self.g_bank_embeddings = None

    # ---------- NEW: a pure-uniform optimizer (Random/TPE) ----------
    def _restrict_space_for_branch(self, config_space, forced_branch, task_type=None):
        """Return a branch-specific catalog when the meta-predictor selects a family."""
        if forced_branch not in {'rdl', 'dfs'}:
            return config_space
        catalog = config_space or build_experiment3_catalog(
            task_type=task_type or 'binary_classification',
            allowed_dfs_models=self.allowed_dfs_models,
        )
        branch_space = catalog.get(forced_branch, {})
        return {forced_branch: deepcopy(branch_space)} if branch_space else catalog

    def _suggest_config_optuna(self, trial, catalog, prior_weights=None, prior_strength=0.7, forced_branch=None):
        if not catalog:
            raise ValueError("Sampling catalog is empty")

        branches = list(catalog.keys())
        if forced_branch in branches:
            branch = forced_branch
        else:
            branch_prior = None
            if prior_weights:
                branch_prior = {
                    br: prior_weights.get(f'model_type_{br}_prob', 0.0)
                    for br in branches
                }
            weighted_branches = weights_to_repeated_choices(branches, branch_prior, prior_strength)
            idx = trial.suggest_int('model_type_index', 0, len(weighted_branches) - 1)
            branch = weighted_branches[idx]

        config = {'model_type': branch}
        branch_space = catalog.get(branch, {})
        for param, candidates in branch_space.items():
            values = list(candidates)
            if not values:
                continue
            prior_map = None
            if prior_weights:
                key = f'{branch}_{param}'
                if key in prior_weights:
                    prior_map = {k: prior_weights[key][k] for k in prior_weights[key] if k in values}
                    if not prior_map:
                        prior_map = None
            weighted_values = weights_to_repeated_choices(values, prior_map, prior_strength)
            idx = trial.suggest_int(f'{branch}::{param}::index', 0, len(weighted_values) - 1)
            config[param] = weighted_values[idx]
        return config

    def _run_optuna_search(self, objective_fn, catalog, max_evals, sampler, prior_weights=None,
                           prior_strength=0.7, forced_branch=None, parallel_trials=None,
                           devices=None, pruner=None, pruner_step=None):
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
        seen_configs = set()
        trial_records = []
        best_record = None

        progress = None
        try:
            if tqdm is not None and max_evals and max_evals > 0:
                progress = tqdm(total=int(max_evals), desc="Optuna Trials", leave=True, position=0)
        except Exception:
            progress = None

        requested_trials = parallel_trials if parallel_trials is not None else self.parallel_trials
        requested_workers = max(1, int(requested_trials or 1))
        resolved_devices = self._resolve_devices(devices) if devices is not None else list(self.parallel_devices)
        device_count = len(resolved_devices)
        worker_count = requested_workers
        if device_count > 0:
            if requested_workers > device_count:
                logger.debug(
                    "Requested %d parallel trials but only %d devices detected; capping worker count.",
                    requested_workers,
                    device_count,
                )
            worker_count = min(requested_workers, device_count)

        use_device_pool = True
        if device_count == 1:
            worker_count = 1
            use_device_pool = False
        elif device_count == 0:
            use_device_pool = worker_count > 1
        else:
            use_device_pool = worker_count > 1

        if use_device_pool:
            if resolved_devices:
                pool_devices = resolved_devices[:worker_count]
            else:
                pool_devices = [None] * worker_count
            device_pool = DevicePool(pool_devices)
        else:
            device_pool = None

        signature = inspect.signature(objective_fn)
        parameters = signature.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        supports_override = 'device_override' in parameters or accepts_kwargs
        supports_device_kw = 'device' in parameters or accepts_kwargs

        def _invoke_objective(config, device_name=None):
            if device_name is not None:
                if supports_override:
                    try:
                        return objective_fn(config, device_override=device_name)
                    except TypeError:
                        if not accepts_kwargs and 'device_override' not in parameters:
                            raise
                if supports_device_kw:
                    try:
                        return objective_fn(config, device=device_name)
                    except TypeError:
                        if not accepts_kwargs and 'device' not in parameters:
                            raise
            return objective_fn(config)

        def _execute_trial(config):
            with device_pool.acquire() as device_name:
                result = _invoke_objective(config, device_name=device_name)
                if isinstance(result, dict) and device_name is not None:
                    result.setdefault('device', device_name)
                return result, device_name

        executed = 0
        attempts = 0
        max_attempts = max(100, max_evals * 20)

        if not use_device_pool:
            while executed < max_evals and attempts < max_attempts:
                trial = study.ask()
                attempts += 1
                config = self._suggest_config_optuna(
                    trial,
                    catalog,
                    prior_weights=prior_weights,
                    prior_strength=prior_strength,
                    forced_branch=forced_branch,
                )
                key = config_to_key(config)
                if key in seen_configs:
                    study.tell(trial, state=TrialState.PRUNED)
                    continue
                seen_configs.add(key)

                result = objective_fn(config)
                if isinstance(result, dict) and result.get('device') is not None:
                    device_name = result.get('device')
                elif device_count == 1:
                    device_name = resolved_devices[0]
                else:
                    device_name = None
                record = {
                    'trial_number': trial.number,
                    'config': deepcopy(config),
                    'result': deepcopy(result),
                    'skipped': bool(result.get('skipped', False)),
                    'device': device_name,
                }
                trial_records.append(record)

                if record['skipped']:
                    record['state'] = str(TrialState.PRUNED)
                    study.tell(trial, state=TrialState.PRUNED)
                    continue

                loss = result.get('loss')
                if loss is None or not np.isfinite(loss):
                    loss = float('inf')
                record['loss'] = float(loss)

                if pruner is not None:
                    step_val = pruner_step if pruner_step is not None else max_evals
                    trial.report(float(loss), step=step_val)
                    if trial.should_prune():
                        study.tell(trial, state=TrialState.PRUNED)
                        record['state'] = str(TrialState.PRUNED)
                        record['pruned'] = True
                        if progress is not None:
                            progress.refresh()
                        continue

                study.tell(trial, float(loss))
                record['state'] = str(TrialState.COMPLETE)
                executed += 1
                if progress is not None:
                    progress.update(1)

                if (best_record is None) or (record['loss'] < best_record.get('loss', float('inf'))):
                    best_record = record
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            futures = {}
            try:
                while (executed < max_evals and attempts < max_attempts) or futures:
                    while (
                        len(futures) < worker_count
                        and executed + len(futures) < max_evals
                        and attempts < max_attempts
                    ):
                        trial = study.ask()
                        attempts += 1
                        config = self._suggest_config_optuna(
                            trial,
                            catalog,
                            prior_weights=prior_weights,
                            prior_strength=prior_strength,
                            forced_branch=forced_branch,
                        )
                        key = config_to_key(config)
                        if key in seen_configs:
                            study.tell(trial, state=TrialState.PRUNED)
                            continue
                        seen_configs.add(key)
                        futures[executor.submit(_execute_trial, config)] = (trial, config, key)

                    if not futures:
                        break

                    done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        trial, config, key = futures.pop(future)
                        try:
                            result, device_name = future.result()
                        except Exception as exc:
                            logger.exception("Optuna trial %s failed during execution: %s", trial.number, exc)
                            result = {
                                'loss': 1.0,
                                'status': STATUS_OK,
                                'val_metric': {},
                                'test_metric': {},
                                'skipped': False,
                            }
                            device_name = None

                        record = {
                            'trial_number': trial.number,
                            'config': deepcopy(config),
                            'result': deepcopy(result),
                            'skipped': bool(result.get('skipped', False)),
                            'device': device_name,
                        }
                        trial_records.append(record)

                        if record['skipped']:
                            record['state'] = str(TrialState.PRUNED)
                            study.tell(trial, state=TrialState.PRUNED)
                            continue

                        loss = result.get('loss')
                        if loss is None or not np.isfinite(loss):
                            loss = float('inf')
                        record['loss'] = float(loss)

                        if pruner is not None:
                            step_val = pruner_step if pruner_step is not None else max_evals
                            trial.report(float(loss), step=step_val)
                            if trial.should_prune():
                                study.tell(trial, state=TrialState.PRUNED)
                                record['state'] = str(TrialState.PRUNED)
                                record['pruned'] = True
                                if progress is not None:
                                    progress.refresh()
                                continue

                        study.tell(trial, float(loss))
                        record['state'] = str(TrialState.COMPLETE)
                        executed += 1
                        if progress is not None:
                            progress.update(1)

                        if (best_record is None) or (record['loss'] < best_record.get('loss', float('inf'))):
                            best_record = record
            finally:
                executor.shutdown(wait=True)

        if progress is not None:
            progress.close()

        if executed < max_evals:
            logger.warning(
                "Optuna search executed %d/%d unique configurations (may be limited by catalog).",
                executed,
                max_evals,
            )

        best_val = best_record['result'].get('val_metric') if best_record else None
        best_test = best_record['result'].get('test_metric') if best_record else None
        all_results = [deepcopy(r['result']) for r in trial_records]

        return best_record, trial_records, best_val, best_test, all_results

    def _format_best_trial(self, record):
        if record is None:
            return None
        return {
            'misc': {'vals': deepcopy(record.get('config', {}))},
            'result': deepcopy(record.get('result', {})),
            'trial_number': record.get('trial_number'),
            'loss': record.get('loss'),
        }

    def _optimize_uniform(self, objective_fn, config_space, max_evals, algo='random', rstate=None,
                          forced_branch=None, task_type=None, parallel_trials=None, devices=None):
        """Uniform/TPE search over the provided catalog using Optuna samplers."""
        catalog = self._restrict_space_for_branch(config_space, forced_branch, task_type=task_type)
        seed = rng_to_seed(rstate)
        if algo == 'random':
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, warn_independent_sampling=False)

        best_record, trial_records, best_val, best_test, all_results = self._run_optuna_search(
            objective_fn,
            catalog,
            max_evals,
            sampler,
            forced_branch=forced_branch,
            parallel_trials=parallel_trials,
            devices=devices,
        )
        best_trial = self._format_best_trial(best_record)
        return best_trial, trial_records, best_val, best_test, all_results

    def build_budget_aware_predictor_dataset(self, budgets=(3, 10, 30, 60), samples_per_budget=20,
                                             feature_bank='heuristic', need_normalization=True,
                                             random_state=42, tasks=None, min_trials_per_branch=3):
        """Create a budget-aware training table for branch prediction.

        For each task and each budget, repeatedly sample `budget` trials from the
        RDL and DFS branches, estimate the probability that each branch yields the
        better model, and return a DataFrame suitable for supervised learning.

        Args:
            budgets: Iterable of evaluation budgets to consider.
            samples_per_budget: Number of Monte-Carlo resamples per budget.
            feature_bank: Name of the feature bank to attach to each row.
            need_normalization: Whether to normalize the feature bank prior to merging.
            random_state: Seed for reproducible resampling.
            tasks: Optional iterable of (dataset_name, task_name) pairs to restrict processing.
            min_trials_per_branch: Minimum number of historical trials required per branch.

        Returns:
            pandas.DataFrame with columns:
                - dataset_name, task_name, task_type
                - budget, samples_used, preferred_branch
                - rdl_win_probability, dfs_win_probability, prob_diff
                - available_rdl_trials, available_dfs_trials
                - feature columns from the requested feature bank
        """

        rng = np.random.RandomState(random_state)
        feature_df = self.extract_full_features(feature_bank, need_normalization=need_normalization)

        if tasks is not None:
            task_set = {(d, t) for d, t in tasks}
            feature_df = feature_df[feature_df.apply(lambda row: (row['dataset_name'], row['task_name']) in task_set, axis=1)]

        results = []
        lower_is_better_metrics = {'mae', 'rmse', 'mse', 'logloss', 'log_loss', 'map', 'mape'}

        for _, feat_row in feature_df.iterrows():
            dataset_name = feat_row['dataset_name']
            task_name = feat_row['task_name']

            task_trials = self.full_features[
                (self.full_features['dataset_name'] == dataset_name) &
                (self.full_features['task_name'] == task_name)
            ]
            if task_trials.empty:
                continue

            metric_name = str(task_trials['main_metric_name'].iloc[0]).lower()
            higher_is_better = metric_name not in lower_is_better_metrics
            task_type = task_trials['task_type'].iloc[0]

            branch_frames = {
                'rdl': task_trials[task_trials['model_type'] == 'rdl'],
                'dfs': task_trials[(task_trials['model_type'] == 'dfs') & (task_trials['dfs_layer'] >= 2)],
            }

            for budget in budgets:
                branch_best_scores = {}
                available = {}
                for branch, frame in branch_frames.items():
                    frame = frame.dropna(subset=['test_metric'])
                    available[branch] = len(frame)
                    if len(frame) < max(min_trials_per_branch, budget):
                        branch_best_scores[branch] = None
                        continue

                    branch_scores = []
                    for _ in range(samples_per_budget):
                        sample = frame.sample(n=budget, replace=False, random_state=rng.randint(0, 2**31 - 1))
                        numeric_scores = sample['test_metric'].astype(float)
                        if not higher_is_better:
                            numeric_scores = -numeric_scores
                        branch_scores.append(float(numeric_scores.max()))
                    branch_best_scores[branch] = branch_scores

                scores_rdl = branch_best_scores.get('rdl')
                scores_dfs = branch_best_scores.get('dfs')
                if not scores_rdl or not scores_dfs:
                    continue

                wins_rdl = 0
                wins_dfs = 0
                ties = 0
                for s_rdl, s_dfs in zip(scores_rdl, scores_dfs):
                    if s_rdl > s_dfs:
                        wins_rdl += 1
                    elif s_dfs > s_rdl:
                        wins_dfs += 1
                    else:
                        ties += 1

                samples_used = len(scores_rdl)
                if samples_used == 0:
                    continue

                prob_rdl = wins_rdl / samples_used
                prob_dfs = wins_dfs / samples_used

                if prob_rdl > prob_dfs:
                    preferred = 'rdl'
                elif prob_dfs > prob_rdl:
                    preferred = 'dfs'
                else:
                    preferred = 'tie'

                row = {
                    'dataset_name': dataset_name,
                    'task_name': task_name,
                    'task_type': task_type,
                    'budget': np.log2(int(budget)),
                    'samples_used': int(samples_used),
                    'preferred_branch': preferred,
                    'rdl_win_probability': prob_rdl,
                    'dfs_win_probability': prob_dfs,
                    'prob_diff': prob_rdl - prob_dfs,
                    'ties': ties,
                    'available_rdl_trials': available['rdl'],
                    'available_dfs_trials': available['dfs'],
                    'main_metric_name': metric_name,
                }

                for col in feature_df.columns:
                    if col not in {'dataset_name', 'task_name'}:
                        row[col] = feat_row[col]

                results.append(row)

        return pd.DataFrame(results)

    def _create_trial_record_from_optuna(self, trial):
        attrs = trial.user_attrs or {}
        record = {
            'trial_number': trial.number,
            'config': deepcopy(attrs.get('config')) if attrs.get('config') is not None else None,
            'result': deepcopy(attrs.get('result')) if attrs.get('result') is not None else None,
            'loss': float(trial.value) if (trial.value is not None and np.isfinite(trial.value)) else None,
            'skipped': bool(attrs.get('skipped', False)),
            'state': str(trial.state),
        }
        return record

    def _optimize_hyperband(self, objective_fn, config_space, max_evals, eta=3, rstate=None,
                             forced_branch=None, task_type=None, parallel_trials=None, devices=None):
        """Hyperband search using Optuna's HyperbandPruner."""
        catalog = self._restrict_space_for_branch(config_space, forced_branch, task_type=task_type)
        seed = rng_to_seed(rstate)
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, warn_independent_sampling=False)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(1, int(max_evals)),
            reduction_factor=int(max(eta, 2)),
        )
        best_record, trial_records, best_val, best_test, all_results = self._run_optuna_search(
            objective_fn,
            catalog,
            max_evals,
            sampler,
            forced_branch=forced_branch,
            parallel_trials=parallel_trials,
            devices=devices,
            pruner=pruner,
            pruner_step=max_evals,
        )
        best_trial = self._format_best_trial(best_record)
        return best_trial, trial_records, best_val, best_test, all_results
    
    def optimize_with_prior(self, objective_fn, config_space, target_task_features,
                            max_evals=50, top_k=3, prior_strength=0.7, rstate=None, algo='tpe',
                            forced_branch=None, skip_unknown_params=False, use_entire_bank=False,
                            parallel_trials=None, devices=None):
        """Prior-guided sampling using Optuna with simulated categorical priors."""
        only_one = forced_branch if (forced_branch in ['rdl', 'dfs']) else ""
        prior_weights = self.create_prior_distribution(
            target_task_features,
            config_space,
            top_k=top_k,
            only_one=only_one,
            use_entire_bank=use_entire_bank
        )

        catalog = self._restrict_space_for_branch(config_space, forced_branch)
        seed = rng_to_seed(rstate)
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, warn_independent_sampling=False)

        best_record, trial_records, best_val, best_test, all_results = self._run_optuna_search(
            objective_fn,
            catalog,
            max_evals,
            sampler,
            prior_weights=prior_weights,
            prior_strength=prior_strength,
            forced_branch=forced_branch,
            parallel_trials=parallel_trials,
            devices=devices,
        )
        best_trial = self._format_best_trial(best_record)
        return best_trial, trial_records, best_val, best_test, all_results


    def extract_target_combined_features(self, database_name, task_name, need_normalization=True):
        """
        Build a single-row combined feature dataframe for the target task using the same feature banks and preprocessing as the current embedding bank.
        """
        if not self.current_feature_banks:
            raise ValueError("Feature banks are not set. Call setup_embedding_bank_from_banks first.")
        
        full_combined = self._build_combined_feature_df(self.current_feature_banks, need_normalization=need_normalization)
        row = full_combined[(full_combined['dataset_name'] == database_name) & (full_combined['task_name'] == task_name)]
        
        if row.empty:
            raise ValueError(f"No combined features found for {database_name}::{task_name}")
        
        return row.reset_index(drop=True)
    # ---------- UPDATE: broaden optimize_for_task with all Exp-3 algorithms + 3 seeds ----------
    def optimize_for_task(self, objective_fn, config_space, database_name, task_name,
                          task_type='binary_classification',
                          max_evals=50, top_k=3, prior_strength=0.7,
                          task_embedding_type='homophily',
                          seeds_list=[42, 43, 44],
                          algorithm='random',
                          meta_threshold=0.5,
                          landscape_topk=10,
                          landscape_enable=True,
                          enable_meta_predictor=True,
                          enable_meta_for_landscape=True,
                          skip_unknown_params=False,
                          parallel_trials=None,
                          parallel_devices=None):
        """
        Extended for Experiment 3.
        - Normal baselines: 'random', 'tpe', 'hyperband'  (uniform, *no* priors)
        - Knowledge transfer: 'autotransfer', 'ours', 'ours_plus_autotransfer'
            * all with normalization and projection g enabled
        Seeds affect only the **hyperparameter choice** (sampler); inside `execution`
        the training seed is fixed to 42 (controllable results).
        parallel_trials / parallel_devices can override the instance defaults when running a task.
        """
        # Create a per-seed RNG for the **sampler only**
        all_seed_runs = []

        if config_space and all(isinstance(v, dict) for v in config_space.values()):
            base_catalog = deepcopy(config_space)
        else:
            base_catalog = build_experiment3_catalog(
                task_type=task_type,
                allowed_dfs_models=self.allowed_dfs_models,
            )
        # Record current algorithm and objective-time saving policy
        self.current_algorithm = algorithm
        self.save_model_during_objective = (
            algorithm in ['ours', 'ours_plus_autotransfer', 'ours+autotransfer', 'whole_prior']
            or landscape_enable
        )
        self.meta_predictor_enabled = enable_meta_predictor
        self.landscape_requested = landscape_enable and enable_meta_predictor


        # If using knowledge transfer, set up the bank once (pretrained-task only)
        if algorithm not in ['random', 'tpe', 'hyperband']:
            if algorithm == 'autotransfer':
                self.setup_embedding_bank_from_banks(['autotransfer'], need_normalization=True, use_g=True)
            elif algorithm == 'ours':
                self.setup_embedding_bank_from_banks(['homophily', 'heuristics_baseline', 'model_anchor_baseline'],
                                                     need_normalization=True, use_g=True)
            elif algorithm == 'whole_prior':
                ## we don't need to set up the bank for whole prior
                pass
            elif algorithm in ['ours_plus_autotransfer', 'ours+autotransfer']:
                self.setup_embedding_bank_from_banks(['homophily', 'heuristics_baseline', 'model_anchor_baseline', 'autotransfer'],
                                                     need_normalization=True, use_g=True)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        # Meta-predictor decision shared across seeds
        meta_decision = None
        preferred_family = None
        if enable_meta_predictor:
            try:
                meta_decision = self.predict_family(database_name, task_name, threshold=meta_threshold, budget=max_evals, budget_aware=True)
                preferred_family = meta_decision.get('decision')
            except Exception as exc:
                logger.warning(f"Meta predictor failed for {database_name}::{task_name}: {exc}")
                meta_decision = None
                preferred_family = None

        # Determine forced branch for samplers if applicable
        forced_branch_uniform = preferred_family if enable_meta_predictor else None

        for sd in seeds_list:
            # strictly for HPO suggestion randomness
            seed_everything(sd)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
            rstate = np.random.RandomState(sd)
            meta_payload = dict(meta_decision) if isinstance(meta_decision, dict) else None

            # ---- NORMAL HPO (uniform) ----
            if algorithm in ['random', 'tpe', 'hyperband']:
                search_catalog = deepcopy(base_catalog)
                if forced_branch_uniform in {'rdl', 'dfs'}:
                    search_catalog = self._restrict_space_for_branch(
                        search_catalog,
                        forced_branch_uniform,
                        task_type=task_type,
                    )
                if algorithm == 'random':
                    best_trial, trials, best_val, best_test, all_results = self._optimize_uniform(
                        objective_fn, search_catalog, max_evals, algo='random', rstate=rstate,
                        forced_branch=forced_branch_uniform, task_type=task_type,
                        parallel_trials=parallel_trials,
                        devices=parallel_devices,
                    )
                elif algorithm == 'tpe':
                    best_trial, trials, best_val, best_test, all_results = self._optimize_uniform(
                        objective_fn, search_catalog, max_evals, algo='tpe', rstate=rstate,
                        forced_branch=forced_branch_uniform, task_type=task_type,
                        parallel_trials=parallel_trials,
                        devices=parallel_devices,
                    )
                else:  # 'hyperband'
                    best_trial, trials, best_val, best_test, all_results = self._optimize_hyperband(
                        objective_fn, search_catalog, max_evals=max_evals, eta=3, rstate=rstate,
                        forced_branch=forced_branch_uniform, task_type=task_type,
                        parallel_trials=parallel_trials,
                        devices=parallel_devices,
                    )
                all_seed_runs.append({
                    'seed': sd,
                    'best': best_trial,
                    'trials': trials,
                    'best_val': best_val,
                    'best_test': best_test,
                    'all_results': all_results,
                    'algorithm': algorithm,
                    'meta_decision': meta_payload,
                    'preferred_family': preferred_family,
                    'meta_enabled': enable_meta_predictor,
                    'meta_for_landscape': enable_meta_for_landscape,
                    'landscape_enable': landscape_enable,
                    'landscape_topk': landscape_topk,
                    'forced_branch': forced_branch_uniform,
                'cached': False,
                })
                continue

            # ---- KNOWLEDGE-TRANSFER HPO (priors from embeddings) ----
            if algorithm in ['autotransfer', 'ours', 'ours_plus_autotransfer', 'ours+autotransfer', 'whole_prior']:
                # bank already set once above
                pass
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # features for the target task using the SAME banks as the embedding bank
            target_features = self.extract_target_combined_features(database_name, task_name, need_normalization=True)
            if target_features is None or target_features.empty:
                raise ValueError(f"No combined features for {database_name}::{task_name} using banks={self.current_feature_banks}")

            # run TPE with priors (sampler seeded by rstate)
            forced_branch = None
            if enable_meta_predictor and algorithm not in ['autotransfer']:
                forced_branch = preferred_family

            best_trial, trials, best_val, best_test, all_results = self.optimize_with_prior(
                objective_fn=objective_fn,
                config_space=base_catalog,
                target_task_features=target_features,
                max_evals=max_evals,
                top_k=top_k,
                prior_strength=prior_strength,
                rstate=rstate,
                algo='tpe',
                forced_branch=forced_branch,
                skip_unknown_params=skip_unknown_params,
                use_entire_bank=(algorithm == 'whole_prior'),
                parallel_trials=parallel_trials,
                devices=parallel_devices,
            )
            all_seed_runs.append({
                'seed': sd,
                'best': best_trial,
                'trials': trials,
                'best_val': best_val,
                'best_test': best_test,
                'all_results': all_results,
                'algorithm': algorithm,
                'meta_decision': meta_payload,
                'preferred_family': preferred_family,
                'meta_enabled': enable_meta_predictor,
                'meta_for_landscape': enable_meta_for_landscape,
                'landscape_enable': landscape_enable,
                'landscape_topk': landscape_topk,
                'forced_branch': forced_branch,
                'cached': False,
            })

        return all_seed_runs  # list of per-seed results
    
    def get_old_exps(self, model_config, landscape_db = None):
        """
        Find matching experiments from self.full_features based on model configuration.
        If landscape_db is provided, then it means we need to calculate landscape metrics. So we need this model_config to be present in landscape_db.
        Args:
            model_config: Dictionary containing model configuration parameters
            landscape_db: Mongodb database containing landscape metrics
        Returns:
            dict: Dictionary with 'val_metric' and 'test_metric' if matches found, None otherwise
        """
        if self.full_features is None or len(self.full_features) == 0:
            return None
            
        def _normalize_pre_sf(val):
            if val == 'src_dst':
                return 'zero_learn'
            return val

        def _build_required_params(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            model_type = params.get('model_type')
            if model_type == 'rdl':
                gnn = params.get('gnn_config', {})
                sampler = params.get('sampler_config', {})
                opt = params.get('optimizer_config', {})
                required = {
                    'batch_size': params.get('batch_size'),
                    'pre_sf': _normalize_pre_sf(gnn.get('pre_sf')),
                    'mpnn_type': gnn.get('mpnn_type'),
                    'hidden_channels': gnn.get('hidden_channels'),
                    'num_heads': gnn.get('num_heads'),
                    'dropout': gnn.get('dropout'),
                    'norm': gnn.get('norm'),
                    'aggregation': gnn.get('aggregation'),
                    'skip_connection': gnn.get('skip_connection'),
                    'temporal': sampler.get('temporal'),
                    'num_neighbors': sampler.get('num_neighbors'),
                    'num_layers': sampler.get('num_layers'),
                    'loader_type': sampler.get('loader_type'),
                    'lr': opt.get('lr'),
                    'weight_decay': opt.get('weight_decay'),
                    'gamma': opt.get('gamma'),
                    'scheduler': opt.get('scheduler'),
                }
            elif model_type == 'dfs':
                tab_nn = params.get('tab_nn_config', {})
                opt = params.get('optimizer_config', {})
                required = {
                    'torch_frame_model_cls': params.get('torch_frame_model_cls'),
                    'dfs_layer': params.get('dfs_layer'),
                    'batch_size': params.get('batch_size'),
                    'hidden_size': tab_nn.get('hidden_size'),
                    'dropout': tab_nn.get('dropout'),
                    'num_layers': tab_nn.get('num_layers'),
                    'attn_dropout': tab_nn.get('attn_dropout'),
                    'num_heads': tab_nn.get('num_heads'),
                    'norm': tab_nn.get('normalization'),
                    'lr': opt.get('lr'),
                    'weight_decay': opt.get('weight_decay'),
                    'gamma': opt.get('gamma'),
                    'scheduler': opt.get('scheduler'),
                }
            else:
                return None

            if any(v is None for v in required.values()):
                return None
            return required

        model_params = model_config.get('model_params', {})
        required_params = _build_required_params(model_params)
        if not required_params:
            return None

        data_config = model_config.get('dbs', [{}])[0] if model_config.get('dbs') else {}
        
        db_name = data_config.get('db_name', '')
        task_name = data_config.get('task_name', [''])[0] if data_config.get('task_name') else ''
        model_type = model_params.get('model_type', '')
        
        mask = (
            (self.full_features['dataset_name'] == db_name) &
            (self.full_features['task_name'] == task_name) &
            (self.full_features['model_type'] == model_type)
        )
        
        filtered_df = self.full_features[mask]
        
        if len(filtered_df) == 0:
            return None
            
        if model_type == 'rdl':
            column_map = rdl_column_name_mapping
        elif model_type == 'dfs':
            column_map = ftt_column_name_mapping
        else:
            return None

        # Build a single boolean mask for all parameter matches
        mask = pd.Series(True, index=filtered_df.index)
        for col, value in required_params.items():
            df_col = col
            if df_col not in filtered_df.columns:
                continue
            mask &= (filtered_df[df_col] == value)
        
        filtered_df = filtered_df[mask]
        if filtered_df.empty:
            return None

        match = filtered_df.iloc[0]
        return {
                'val_metric': match.get('val_metric', {}),
                'test_metric': match.get('test_metric', {}),
                'match_type': 'exact',
            'num_matches': len(filtered_df)
        }
