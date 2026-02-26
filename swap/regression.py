import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import re
import torch
from utils.analysis import summarize_best_trials
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def load_model_task_embedding(task_emb_path):
    ## go under task_emb_path
    ## all first with format rel-xxx-task_name-model-(free-gpu)
    ## find all ends with free-gpu
    ## extract their file contents, database name, task name and return
    files = []
    task_embeddings = {}
    for root, dirs, filenames in os.walk(task_emb_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    for filename in files:
        # Pattern to extract database and task name
        pattern_model_free = r'rel-([^-]+)-(.+?)-model-free-gpu'
        pattern_anchor_model = r'rel-([^-]+)-(.+?)-model-heuristics-full'
        # pattern_model = r'rel-([^-]+)-(.+?)-model(?!-free-gpu)'
        pattern_autotransfer_rdl = r'rel-([^-]+)-(.+?)-model-atrdl'
        pattern_autotransfer_dfs = r'rel-([^-]+)-(.+?)-model-atdfs'
        match_model_free = re.search(pattern_model_free, filename)
        # match_model = re.search(pattern_model, filename)
        match_autotransfer_rdl = re.search(pattern_autotransfer_rdl, filename)
        match_autotransfer_dfs = re.search(pattern_autotransfer_dfs, filename)
        match_anchor_model = re.search(pattern_anchor_model, filename)
        # if match_model_free:
        if match_autotransfer_rdl:
            database = f"rel-{match_autotransfer_rdl.group(1)}"  # Add "rel-" prefix back
            if database not in task_embeddings:
                task_embeddings[database] = {}
            task_name = match_autotransfer_rdl.group(2)
            if task_name not in task_embeddings[database]:
                task_embeddings[database][task_name] = {}
            print(f"Database: {database}")
            print(f"Task name: {task_name}")
            pt_file = torch.load(filename, map_location=torch.device('cpu'))
            task_embeddings[database][task_name]['model_atrdl'] = pt_file
        elif match_autotransfer_dfs:
            database = f"rel-{match_autotransfer_dfs.group(1)}"  # Add "rel-" prefix back
            task_name = match_autotransfer_dfs.group(2)
            if database not in task_embeddings:
                task_embeddings[database] = {}
            if task_name not in task_embeddings[database]:
                task_embeddings[database][task_name] = {}
            print(f"Database: {database}")
            print(f"Task name: {task_name}")
            pt_file = torch.load(filename, map_location=torch.device('cpu'))
            task_embeddings[database][task_name]['model_atdfs'] = pt_file
        elif match_model_free:
            database = f"rel-{match_model_free.group(1)}"  # Add "rel-" prefix back
            task_name = match_model_free.group(2)
            if database not in task_embeddings:
                task_embeddings[database] = {}
            if task_name not in task_embeddings[database]:
                task_embeddings[database][task_name] = {}
            print(f"Database: {database}")
            print(f"Task name: {task_name}")
            pt_file = torch.load(filename, map_location=torch.device('cpu'))
            task_embeddings[database][task_name]['model_free'] = pt_file
        elif match_anchor_model:
            database = f"rel-{match_anchor_model.group(1)}"  # Add "rel-" prefix back
            task_name = match_anchor_model.group(2)
            if database not in task_embeddings:
                task_embeddings[database] = {}
            if task_name not in task_embeddings[database]:
                task_embeddings[database][task_name] = {}
            print(f"Database: {database}")
            print(f"Task name: {task_name}")
            pt_file = torch.load(filename, map_location=torch.device('cpu'))
            task_embeddings[database][task_name]['model_anchor'] = pt_file
        else:
            continue

    return task_embeddings

def extract_homophily_related_features(task_embeddings):
    pandas_dataframe = []
    for database in task_embeddings:
        for task_name in task_embeddings[database]:
            if not 'model_free' in task_embeddings[database][task_name]:
                continue
            model_free = task_embeddings[database][task_name]['model_free']
            row = {}
            row['database'] = database
            row['task_name'] = task_name 
            row['h_edges_mean'] = np.array(model_free['h_edges']).mean()
            row['h_edges_std'] = np.array(model_free['h_edges']).std()
            row['h_edges_max'] = np.array(model_free['h_edges']).max()
            row['h_edges_min'] = np.array(model_free['h_edges']).min()
            row['h_edges_median'] = np.median(np.array(model_free['h_edges']))
            row['h_insensitives_mean'] = np.array(model_free['h_insensitives']).mean()
            row['h_insensitives_std'] = np.array(model_free['h_insensitives']).std()
            row['h_insensitives_max'] = np.array(model_free['h_insensitives']).max()
            row['h_insensitives_min'] = np.array(model_free['h_insensitives']).min()
            row['h_insensitives_median'] = np.median(np.array(model_free['h_insensitives']))
            row['h_adjs_mean'] = np.array(model_free['h_adjs']).mean()
            row['h_adjs_std'] = np.array(model_free['h_adjs']).std()
            row['h_adjs_max'] = np.array(model_free['h_adjs']).max()
            row['h_adjs_min'] = np.array(model_free['h_adjs']).min()
            row['h_adjs_median'] = np.median(np.array(model_free['h_adjs']))
            row['h_aggs_mean'] = np.array(model_free['h_aggs']).mean()
            row['h_aggs_std'] = np.array(model_free['h_aggs']).std()
            row['h_aggs_max'] = np.array(model_free['h_aggs']).max()
            row['h_aggs_min'] = np.array(model_free['h_aggs']).min()
            row['h_aggs_median'] = np.median(np.array(model_free['h_aggs']))
            pandas_dataframe.append(row)
    return pandas_dataframe

def extract_heuristics_features(task_embeddings):
    heuristics_features = []
    for database in task_embeddings:
        for task_name in task_embeddings[database]:
            if not 'model_anchor' in task_embeddings[database][task_name]:
                continue
            model_free = task_embeddings[database][task_name]['model_anchor']['heuristics']
            # model = task_embeddings[database][task_name]['model']
            row = {}
            row['database'] = database
            row['task_name'] = task_name 
            row['train_baseline'] = model_free['train_baseline']
            row['val_baseline'] = model_free['val_baseline']
            # wanted_metric = task_name_to_metric_no_prefix[task_name]
            for split in ['train', 'val']:
                row[f'global_zero_{split}'] = model_free[f'{split}_heuristics']['global_zero']
                row[f'global_mean_{split}'] = model_free[f'{split}_heuristics']['global_mean']
                row[f'global_median_{split}'] = model_free[f'{split}_heuristics']['global_median']
                row[f'entity_mean_{split}'] = model_free[f'{split}_heuristics']['entity_mean']
                row[f'entity_median_{split}'] = model_free[f'{split}_heuristics']['entity_median']
                row['majority'] = model_free[f'{split}_heuristics']['majority']
            heuristics_features.append(row)
    return heuristics_features

def extract_anchor_model_features(task_embeddings):
    anchor_model_features = []
    for database in task_embeddings:
        for task_name in task_embeddings[database]:
            if not 'model_anchor' in task_embeddings[database][task_name]:
                continue
            model_free = task_embeddings[database][task_name]['model_anchor']
            row = {}
            row['database'] = database
            row['task_name'] = task_name
            # row['train_baseline'] = model_free['train_baseline']
            # row['val_baseline'] = model_free['val_baseline']
            row['tabpfn_1hop'] = model_free['tabpfn']['1hop']
            row['tabpfn_2hop'] = model_free['tabpfn']['2hop']
            for i in range(1, 4):
                row[f'rfr_randomsage{i}'] = model_free['rfr_randomsage'][i - 1]
            for i in range(1, 4):
                row[f'rfr_randomnbfnet{i}'] = model_free['rfr_randomnbfnet'][i - 1]
            anchor_model_features.append(row)
    return anchor_model_features

def extract_autotransfer_features(task_embeddings):
    autotransfer_features = []
    for database in task_embeddings:
        for task_name in task_embeddings[database]:
            # model_free = task_embeddings[database][task_name]['model_free']
            if 'model_atrdl' not in task_embeddings[database][task_name]:
                continue
            model = task_embeddings[database][task_name]['model_atrdl']
            model_dfs = task_embeddings[database][task_name]['model_atdfs']
            row = {}
            row['dataset_name'] = database
            row['task_name'] = task_name 
            # row['oa_nt_at'] = model['oa_nt_at']
            # row['oa_th_at'] = model['oa_th_at']
            # row['oa_ft_at'] = model['oa_ft_at']
            for i in range(12):
                # row[f'ta_nt_at_{i}'] = model['ta_nt_at'][i]
                row[f'ta_th_at_{i}'] = model['ta_th_at'][i].item()
            for i in range(6):
                row[f'ta_th_at_dfs_{i}'] = model_dfs['ta_th_at_dfs_1'][i].item()
            for i in range(6, 12):
                row[f'ta_th_at_dfs_{i}'] = model_dfs['ta_th_at_dfs_2'][i - 6].item()
            autotransfer_features.append(row)
    return autotransfer_features

def extract_loss_barrier_features(task_embeddings):
    loss_barrier_features = []
    for database in task_embeddings:
        for task_name in task_embeddings[database]:
            # model_free = task_embeddings[database][task_name]['model_free']
            if 'model' not in task_embeddings[database][task_name]:
                continue
            model = task_embeddings[database][task_name]['model']
            row = {}
            row['database'] = database
            row['task_name'] = task_name 
            row['oa_nt_lb'] = model['oa_nt_lb']
            row['oa_th_lb'] = model['oa_th_lb']
            row['oa_ft_lb'] = model['oa_ft_lb']
            for i in range(4):
                row[f'ta_nt_lb_{i}'] = model['ta_nt_lb'][i]
                row[f'ta_th_lb_{i}'] = model['ta_th_lb'][i]
            loss_barrier_features.append(row)
    return loss_barrier_features

def extract_regression_targets(dataframe, target_selection = "val_gap"):
    # target 
    # 1. gap, gap between tree and gnn 
    # 2. validation gap, gap between validation and test for rdl 
    if target_selection == "rdl_dfs_gap":
        best_val = summarize_best_trials(dataframe, metric="val", tie_strategy="first", by_category=True) 
        gap = (
        best_val.pivot_table(index=["dataset_name","task_name","task_type"],
                    columns="method_category", values="test_metric", aggfunc="first")
        .reset_index()
    )
        gap["gap_raw"] = gap["rdl"] - gap["dfs"]
        gap["ratio"] = gap["gap_raw"] / gap[["rdl","dfs"]].max(axis=1)
        gap["advantage"] = gap["ratio"].where(gap["task_type"]!="regression", -gap["ratio"])
        return gap
    elif target_selection == 'val_gap':
        best_val = summarize_best_trials(dataframe, metric="val", tie_strategy="first", by_category=True)
        best_possible_test = summarize_best_trials(dataframe, metric="test", tie_strategy="first", by_category=True)
        ## only select rdl
        best_val = best_val[best_val['method_category'] == 'rdl']
        best_possible_test = best_possible_test[best_possible_test['method_category'] == 'rdl']
        
        # Merge the dataframes to align validation-selected and best-possible test results
        merged_data = best_val.merge(
            best_possible_test[['dataset_name', 'task_name', 'task_type', 'test_metric']], 
            on=['dataset_name', 'task_name', 'task_type'], 
            suffixes=('_val_selected', '_best_possible')
        )
        
        ## gap is the difference between best-possible test metric and validation-selected test metric
        merged_data["gap_raw"] = abs(merged_data["test_metric_best_possible"] - merged_data["test_metric_val_selected"])
        merged_data["ratio"] = merged_data["gap_raw"] / merged_data["test_metric_best_possible"]
        merged_data["advantage"] = merged_data['ratio']
        return merged_data



def plot_homophily_correlations(merged_df,
features = ['h_edges_mean', 'h_insensitives_mean', 'h_adjs_mean', 'h_aggs_mean'],
feature_labels = ['h_edges', 'h_insensitives', 'h_adjs', 'h_aggs'],
 save_path="homophily_correlations.png", figsize=(16, 12)):
    """
    Create a unified plot showing correlations between homophily features and advantage
    for both classification and regression tasks.
    
    Parameters:
    - merged_df: DataFrame containing the merged data
    - save_path: Path to save the combined figure
    - figsize: Figure size tuple
    """
    # Define the features to plot
    # features = ['h_edges_mean', 'h_insensitives_mean', 'h_adjs_mean', 'h_aggs_mean']
    # feature_labels = ['h_edges', 'h_insensitives', 'h_adjs', 'h_aggs']
    
    # Separate classification and regression data
    classification_data = merged_df[merged_df['task_type'] == 'binary_classification']
    regression_data = merged_df[merged_df['task_type'] == 'regression']
    
    # Create subplots: 4 rows (features) x 2 columns (classification, regression)
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle('Homophily Correlations with Advantage', fontsize=16, fontweight='bold')
    
    for i, (feature, label) in enumerate(zip(features, feature_labels)):
        # Classification subplot (left column)
        ax_class = axes[i, 0]
        if len(classification_data) > 0:
            X_class = classification_data[feature].values.reshape(-1, 1)
            y_class = classification_data['advantage'].values
            lr_class = LinearRegression()
            lr_class.fit(X_class, y_class)
            r2_class = r2_score(y_class, lr_class.predict(X_class))
            
            ax_class.scatter(classification_data[feature], classification_data['advantage'], alpha=0.6)
            ax_class.set_xlabel(feature)
            ax_class.set_ylabel("advantage")
            ax_class.set_title(f"Classification - {label} (R² = {r2_class:.4f})")
            ax_class.grid(True, alpha=0.3)
        else:
            ax_class.text(0.5, 0.5, 'No classification data', ha='center', va='center', transform=ax_class.transAxes)
            ax_class.set_title(f"Classification - {label}")
        
        # Regression subplot (right column)
        ax_reg = axes[i, 1]
        if len(regression_data) > 0:
            X_reg = regression_data[feature].values.reshape(-1, 1)
            y_reg = regression_data['advantage'].values
            lr_reg = LinearRegression()
            lr_reg.fit(X_reg, y_reg)
            r2_reg = r2_score(y_reg, lr_reg.predict(X_reg))
            
            ax_reg.scatter(regression_data[feature], regression_data['advantage'], alpha=0.6, color='orange')
            ax_reg.set_xlabel(feature)
            ax_reg.set_ylabel("advantage")
            ax_reg.set_title(f"Regression - {label} (R² = {r2_reg:.4f})")
            ax_reg.grid(True, alpha=0.3)
        else:
            ax_reg.text(0.5, 0.5, 'No regression data', ha='center', va='center', transform=ax_reg.transAxes)
            ax_reg.set_title(f"Regression - {label}")
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for the main title
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Homophily correlation plots saved to: {save_path}")
    
    # Clear the current figure
    plt.clf()
    
    return fig


def calculate_lr_metric(df, feature, target):
    """
    Calculate linear regression metrics for a given feature and target.
    
    Args:
        df: DataFrame containing the data
        feature: Name of the feature column to use as predictor
        target: Name of the target column to predict
    
    Returns:
        dict: Dictionary containing R², coefficient, intercept, and p-value
    """
    from scipy import stats
    
    # Remove rows with NaN values in either feature or target
    clean_df = df[[feature, target]].dropna()
    
    if len(clean_df) < 2:
        return {
            'r2': 0.0,
            'coefficient': 0.0,
            'intercept': 0.0,
            'p_value': 1.0,
            'n_samples': len(clean_df)
        }
    
    X = clean_df[feature].values.reshape(-1, 1)
    y = clean_df[target].values
    
    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Calculate R²
    y_pred = lr.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Calculate p-value using scipy.stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_df[feature], clean_df[target])
    
    return {
        'r2': r2,
        'coefficient': lr.coef_[0],
        'intercept': lr.intercept_,
        'p_value': p_value,
        'n_samples': len(clean_df)
    }

def plot_loss_barrier(res, ax=None):
    import numpy as np, matplotlib.pyplot as plt
    try:
        import torch
    except Exception:
        torch = None

    r = res[0] if isinstance(res, (list, tuple, np.ndarray)) or getattr(res, 'dtype', None) is object else res
    t = r['ts']
    t = t.detach().cpu().numpy() if (torch and hasattr(t, 'detach')) else np.asarray(t, float)
    L = np.asarray(r['losses'], float)
    l0, l1 = float(r.get('loss0', L[0])), float(r.get('loss1', L[-1]))
    base = (1 - t) * l0 + t * l1

    gap_conv = L - base
    i_star = int(np.argmax(gap_conv))
    barrier_conv = float(gap_conv[i_star])
    barrier_abs = float(np.max(L - max(l0, l1)))
    t_star = float(t[i_star])

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, L, marker='o', label='loss(t)')
    ax.plot(t, base, ls='--', label='linear baseline')
    ax.hlines(max(l0, l1), t.min(), t.max(), ls=':', label='max{L(0), L(1)}')
    if barrier_conv > 0:
        ax.vlines(t[i_star], base[i_star], L[i_star], ls=':')
    ax.set(xlabel='t', ylabel='loss', title='Loss barrier')
    ax.legend()
    return {"barrier_abs": barrier_abs, "barrier_conv": barrier_conv, "t_star": t_star}

def save_loss_barrier(res, path, ax=None, **savefig_kw):
    """Save a loss-barrier plot. Accepts any matplotlib.savefig kwargs."""
    import matplotlib.pyplot as plt
    stats = plot_loss_barrier(res, ax=ax)         # uses your function from above
    fig = ax.figure if ax is not None else plt.gcf()
    fig.savefig(path, bbox_inches="tight", **savefig_kw)
    return stats

def load_full_homophily_features(task_embeddings):
    homophily_related_features = extract_homophily_related_features(task_embeddings)
    homophily_df = pd.DataFrame(homophily_related_features)
    homophily_df = homophily_df.rename(columns={'database': 'dataset_name'})
    return homophily_df

def load_full_heuristics_features(task_embeddings):
    heuristics_features = extract_heuristics_features(task_embeddings)
    heuristics_df = pd.DataFrame(heuristics_features)
    heuristics_df = heuristics_df.rename(columns={'database': 'dataset_name'})
    return heuristics_df

def load_full_autotransfer_features(task_embeddings):
    autotransfer_features = extract_autotransfer_features(task_embeddings)
    autotransfer_df = pd.DataFrame(autotransfer_features)
    autotransfer_df = autotransfer_df.rename(columns={'database': 'dataset_name'})
    return autotransfer_df


if __name__ == "__main__":
    from utils.retrieve_search_data import WandBAnalyzer

    task_embeddings = load_model_task_embedding('./taskemb')
    homophily_related_features = extract_homophily_related_features(task_embeddings)
    heuristics_features = extract_heuristics_features(task_embeddings)
    autotransfer_features = extract_autotransfer_features(task_embeddings)
    loss_barrier_features = extract_loss_barrier_features(task_embeddings)
    analyzer = WandBAnalyzer(entity=os.getenv("WANDB_ENTITY", "msuczk"))
    analyzer.check_and_download_datasets()
    dataframe = analyzer.combine_and_generate_a_large_dataframe()
    target = extract_regression_targets(dataframe, target_selection="rdl_dfs_gap")
    target = target[target['advantage'].notna()]
    target2 = extract_regression_targets(dataframe, target_selection="val_gap")
    target2 = target2[target2['advantage'].notna()]

    homophily_df = pd.DataFrame(homophily_related_features)
    homophily_df = homophily_df.rename(columns={'database': 'dataset_name'})
    autotransfer_df = pd.DataFrame(autotransfer_features)
    loss_barrier_df = pd.DataFrame(loss_barrier_features)
    autotransfer_df = autotransfer_df.rename(columns={'database': 'dataset_name'})
    loss_barrier_df = loss_barrier_df.rename(columns={'database': 'dataset_name'})

    merged_df = target.merge(homophily_df, on=['dataset_name', 'task_name'], how='inner')
    merged_df2 = target2.merge(homophily_df, on=['dataset_name', 'task_name'], how='inner')
    merged_autotransfer_df = target.merge(autotransfer_df, on=['dataset_name', 'task_name'], how='inner')
    merged_autotransfer_df_val = target2.merge(autotransfer_df, on=['dataset_name', 'task_name'], how='inner')
    merged_loss_barrier_df = target.merge(loss_barrier_df, on=['dataset_name', 'task_name'], how='inner')
    merged_loss_barrier_df_val = target2.merge(loss_barrier_df, on=['dataset_name', 'task_name'], how='inner')

    # Plot homophily correlations
    for df, suffix in [(merged_df, ''), (merged_df2, '_val')]:
        plot_homophily_correlations(df, save_path=f"homophily_correlations{suffix}.png")
        plot_homophily_correlations(df,
            features=['h_edges_median', 'h_insensitives_median', 'h_adjs_median', 'h_aggs_median'],
            feature_labels=['h_edges', 'h_insensitives', 'h_adjs', 'h_aggs'],
            save_path=f"homophily_correlations{suffix}_median.png")
        plot_homophily_correlations(df,
            features=['h_edges_std', 'h_insensitives_std', 'h_adjs_std', 'h_aggs_std'],
            feature_labels=['h_edges', 'h_insensitives', 'h_adjs', 'h_aggs'],
            save_path=f"homophily_correlations{suffix}_std.png")

    # Plot autotransfer correlations
    at_cols = ['oa_nt_at', 'oa_th_at', 'oa_ft_at']
    for at_df, suffix in [(merged_autotransfer_df, ''), (merged_autotransfer_df_val, '_val')]:
        available_at_cols = [c for c in at_cols if c in at_df.columns]
        if not available_at_cols:
            continue
        at_df = at_df.astype({col: 'float64' for col in at_df.columns if '_at' in col})
        at_df[available_at_cols] = at_df[available_at_cols].fillna(at_df[available_at_cols].mean())
        at_df[available_at_cols] = at_df[available_at_cols].apply(lambda x: (x - x.mean()) / x.std())
        plot_homophily_correlations(at_df, features=available_at_cols, feature_labels=available_at_cols,
            save_path=f"homophily_correlations_autotransfer{suffix}.png")

    # Plot loss barrier for selected tasks
    for task_name in ['user-churn', 'site-success', 'driver-position', 'user-badge', 'item-sales']:
        task_data = merged_loss_barrier_df[merged_loss_barrier_df['task_name'] == task_name]
        if len(task_data) > 0:
            oa_th_lb = task_data['oa_th_lb'].values[0][0]
            save_loss_barrier(oa_th_lb, path=f"{task_name}_oa_nt_lb_loss_barrier.png")

    # Normalize and plot loss barrier correlations
    for lb_df, suffix in [(merged_loss_barrier_df, ''), (merged_loss_barrier_df_val, '_val')]:
        lb_df['oa_th_lb_barrier_conv'] = lb_df['oa_th_lb'].apply(lambda x: x[0]['barrier_conv'])
        mean_val = lb_df['oa_th_lb_barrier_conv'].mean()
        std_val = lb_df['oa_th_lb_barrier_conv'].std()
        lb_df['oa_th_lb_barrier_conv'] = (lb_df['oa_th_lb_barrier_conv'] - mean_val) / std_val
        plot_homophily_correlations(lb_df,
            features=['oa_th_lb_barrier_conv'], feature_labels=['oa_th_lb_barrier_conv'],
            save_path=f"homophily_correlations_loss_barrier{suffix}.png")


    


    








