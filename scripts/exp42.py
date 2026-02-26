import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import torch

# Check GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Try to import TabPFN
try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: TabPFN not available. Install with: pip install tabpfn")

import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('results/task_feature_summary.csv')

print("="*80)
print("COMPREHENSIVE LOO EVALUATION")
print("Including: Model-Free, AutoTransfer, Model-Based Features")
print("Tasks: Binary Classification (Winner) and Regression (Gap)")
print("="*80)

# ============================================================================
# STEP 1: CORRECT WINNER AND GAP CALCULATION
# ============================================================================

def determine_winner(row, selection='val'):
    """
    Correctly determine winner based on task type.
    Classification: higher is better
    Regression: lower is better
    """
    rdl_metric = row[f'rdl_test_metric_selected_by_{selection}']
    dfs_metric = row[f'dfs_test_metric_selected_by_{selection}']
    
    if 'classification' in row['task_type']:
        # Higher is better for classification
        return 1 if rdl_metric > dfs_metric else 0
    else:
        # Lower is better for regression
        return 1 if rdl_metric < dfs_metric else 0

def calculate_gap(row, selection='val'):
    """
    Calculate gap with correct sign based on task type.
    Positive gap means RDL is better.
    """
    rdl_metric = row[f'rdl_test_metric_selected_by_{selection}']
    dfs_metric = row[f'dfs_test_metric_selected_by_{selection}']
    
    if 'classification' in row['task_type']:
        return rdl_metric - dfs_metric  # Positive if RDL better
    else:
        return dfs_metric - rdl_metric  # Positive if RDL better (since lower is better)

# Create labels for both selection strategies
df['winner_by_val'] = df.apply(lambda x: determine_winner(x, 'val'), axis=1)
df['winner_by_test'] = df.apply(lambda x: determine_winner(x, 'test'), axis=1)
df['gap_by_val'] = df.apply(lambda x: calculate_gap(x, 'val'), axis=1)
df['gap_by_test'] = df.apply(lambda x: calculate_gap(x, 'test'), axis=1)

print("\n1. CORRECTED WINNER DISTRIBUTION:")
print("-" * 50)
print(f"Val-selected:  RDL wins {df['winner_by_val'].mean():.1%} (Gap mean: {df['gap_by_val'].mean():.3f})")
print(f"Test-selected: RDL wins {df['winner_by_test'].mean():.1%} (Gap mean: {df['gap_by_test'].mean():.3f})")
print(f"\nBy task type (test-selected):")
print(f"  Classification: RDL wins {df[df['task_type'].str.contains('classification')]['winner_by_test'].mean():.1%}")
print(f"  Regression:      RDL wins {df[df['task_type'] == 'regression']['winner_by_test'].mean():.1%}")

# ============================================================================
# STEP 2: DEFINE ALL FEATURE GROUPS
# ============================================================================

print("\n2. DEFINING FEATURE GROUPS:")
print("-" * 50)

# Homophily features
homophily_edge_corr = ['h_edges_corr_mean', 'h_edges_corr_max', 'h_edges_corr_min', 
                        'h_edges_corr_mode', 'h_edges_corr_weighted_mean']
homophily_adj_corr = ['h_adjs_corr_mean', 'h_adjs_corr_max', 'h_adjs_corr_min',
                       'h_adjs_corr_mode', 'h_adjs_corr_weighted_mean']



original_class_ratio = [
        'variance_same_class_ratio',
        'mean_same_class_ratio_ignore',
        'adjusted_mean_same_class_ratio',
        'sparsity_ratio',
        'mean_past_task_nodes'
    ]

real_entity = ['entity_mean_val', 'entity_median_val', 'entity_mean_train', 'entity_median_train']

recommended = ['entity_mean_val',
'entity_median_val',
       'variance_same_class_ratio',
        'mean_same_class_ratio_ignore',
        'adjusted_mean_same_class_ratio',
        'sparsity_ratio',
        'mean_past_task_nodes']

# Autocorrelation features
autocorr_features = ['lag1_autocorr_corr', 'lag2_autocorr_corr']


# Size features
df['log_total_rows'] = np.log1p(df['train_rows'] + df['val_rows'])
size_features = ['log_total_rows']

# 2. AUTOTRANSFER FEATURES
at_rdl_features = [col for col in df.columns if col.startswith('at_rdl_')]
at_dfs_features = [col for col in df.columns if col.startswith('at_dfs_')]
at_diff_features = []  # Create difference features
for i in range(min(len(at_rdl_features), len(at_dfs_features))):
    rdl_col = f'at_rdl_{i}'
    dfs_col = f'at_dfs_{i}'
    if rdl_col in df.columns and dfs_col in df.columns:
        diff_col = f'at_diff_{i}'
        df[diff_col] = df[rdl_col] - df[dfs_col]
        at_diff_features.append(diff_col)

# 3. MODEL-BASED FEATURES  
model_based_features = []

# TabPFN features
tabpfn_features = ['tabpfn_1hop', 'tabpfn_2hop', 'rfr_randomsage_1', 'rfr_randomsage_2', 'rfr_randomsage_3', 'rfr_randomnbfnet_1', 'rfr_randomnbfnet_2', 'rfr_randomnbfnet_3']
model_based_features = tabpfn_features

guess = homophily_adj_corr + autocorr_features + size_features +original_class_ratio

combine = guess + real_entity


# Check which features are actually available
all_feature_groups = {
    # Model-Free
    'AT_All': at_rdl_features + at_dfs_features,
    'All_Model_Based': model_based_features,
    'Real_Entity': real_entity,
    'Guess': guess,
    'Combine': combine,
}

# Filter to available features and print summary
available_groups = {}
for group_name, features in all_feature_groups.items():
    available = [f for f in features if f in df.columns]
    if available:
        available_groups[group_name] = available
        category = "Model-Free" if group_name in ['Entity', 'Entity_Log', 'Homophily_Edge_Corr', 
                                                   'Homophily_Adj_Corr', 'Class_Ratio', 
                                                   'Autocorr', 'Structural', 'Size'] else \
                   "AutoTransfer" if 'AT_' in group_name else "Model-Based"
        print(f"  [{category:12}] {group_name:20}: {len(available):2} features")

# ============================================================================
# STEP 3: EVALUATION FUNCTIONS
# ============================================================================

def loo_evaluate_classification(X, y, model_type='rf'):
    """
    Leave-One-Out cross-validation for binary classification
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    # Convert to DataFrame for easier NaN handling (matching llm.py approach)
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()
    
    # Fill NaN values with median (matching llm.py)
    for col in X_df.columns:
        X_df[col].fillna(X_df[col].median(), inplace=True)
    
    # Remove constant features
    non_constant = X_df.std() > 0
    X_df = X_df.loc[:, non_constant]
    
    if len(X_df.columns) == 0:
        return None, None
    
    # Handle y NaN values
    y_series = pd.Series(y)
    y_series.fillna(y_series.median(), inplace=True)
    
    if len(X_df) < 10:  # Need minimum samples
        return None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    loo = LeaveOneOut()
    y_pred = []
    y_prob = []
    
    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]
        
        if model_type == 'dt':
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
        elif model_type == 'rf':
            model =RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == 'gb':
            model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        elif model_type == 'nn':
            model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.2)
        elif model_type == 'tabpfn':
            if not TABPFN_AVAILABLE:
                return None, None
            model = TabPFNClassifier(device=DEVICE, N_ensemble_configurations=4)
        else:
            model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]
            y_pred.append(pred)
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_test)[0, 1]
            else:
                prob = pred
            y_prob.append(prob)
        except:
            y_pred.append(int(y_train.mean() > 0.5))
            y_prob.append(y_train.mean())
    
    accuracy = accuracy_score(y_series, y_pred)
    try:
        auc = roc_auc_score(y_series, y_prob)
    except:
        auc = 0.5
    
    return accuracy, auc

def loo_evaluate_regression(X, y, model_type='rf'):
    """
    Leave-One-Out cross-validation for regression (gap prediction)
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    # Convert to DataFrame for easier NaN handling (matching llm.py approach)
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()
    
    # Fill NaN values with median (matching llm.py)
    for col in X_df.columns:
        X_df[col].fillna(X_df[col].median(), inplace=True)
    
    # Remove constant features
    non_constant = X_df.std() > 0
    X_df = X_df.loc[:, non_constant]
    
    if len(X_df.columns) == 0:
        return None, None, None
    
    # Handle y NaN values
    y_series = pd.Series(y)
    y_series.fillna(y_series.median(), inplace=True)
    
    if len(X_df) < 10:  # Need minimum samples
        return None, None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    loo = LeaveOneOut()
    y_pred = []
    
    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]
        
        if model_type == 'linear':
            model = Ridge(alpha=1.0, random_state=42)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == 'gb':
            model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        elif model_type == 'nn':
            model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.2)
        elif model_type == 'tabpfn':
            if not TABPFN_AVAILABLE:
                return None, None, None
            model = TabPFNRegressor(device=DEVICE, N_ensemble_configurations=4)
        else:
            model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]
            y_pred.append(pred)
        except:
            y_pred.append(y_train.mean())
    
    mse = mean_squared_error(y_series, y_pred)
    mae = mean_absolute_error(y_series, y_pred)
    try:
        r2 = r2_score(y_series, y_pred)
    except:
        r2 = 0
    
    return mse, mae, r2

# ============================================================================
# STEP 4: TEST INDIVIDUAL FEATURE GROUPS
# ============================================================================

print("\n3. INDIVIDUAL FEATURE GROUP PERFORMANCE:")
print("="*120)

# Headers
print(f"{'Feature Group':<25} {'Category':<12} {'#F':<3} ", end="")
print(f"{'Val-Selected':^35} {'Test-Selected':^35}")
print(f"{'':^40} {'Winner Acc':^17} {'Gap MAE':^17} {'Winner Acc':^17} {'Gap MAE':^17}")
print("-"*120)

individual_results = []

for group_name, features in available_groups.items():
    if len(features) == 0:
        continue
        
    X = df[features].values
    
    # Determine category
    category = "Model-Free" if group_name in ['Entity', 'Entity_Log', 'Homophily_Edge_Corr', 
                                               'Homophily_Adj_Corr', 'Class_Ratio', 
                                               'Autocorr', 'Structural', 'Size'] else \
               "AutoTransfer" if 'AT_' in group_name else "Model-Based"
    
    # Test on VAL-SELECTED winners and gaps
    val_class_results = []
    val_reg_results = []

    for model_type in ['rf', 'dt', 'gb']:
        acc, auc = loo_evaluate_classification(X, df['winner_by_val'].values, model_type)
        if acc is not None:
            val_class_results.append(acc)

    for model_type in ['rf', 'linear', 'gb']:
        mse, mae, r2 = loo_evaluate_regression(X, df['gap_by_val'].values, model_type)
        if mae is not None:
            val_reg_results.append(mae)

    # Test on TEST-SELECTED winners and gaps
    test_class_results = []
    test_reg_results = []

    for model_type in ['rf', 'dt', 'gb']:
        acc, auc = loo_evaluate_classification(X, df['winner_by_test'].values, model_type)
        if acc is not None:
            test_class_results.append(acc)

    for model_type in ['rf', 'linear', 'gb']:
        mse, mae, r2 = loo_evaluate_regression(X, df['gap_by_test'].values, model_type)
        if mae is not None:
            test_reg_results.append(mae)
    
    # Get best results
    val_acc = max(val_class_results) if val_class_results else 0
    val_mae = min(val_reg_results) if val_reg_results else 999
    test_acc = max(test_class_results) if test_class_results else 0
    test_mae = min(test_reg_results) if test_reg_results else 999
    
    print(f"{group_name:<25} {category:<12} {len(features):<3} "
          f"{val_acc:>8.3f} / {1-val_acc:>5.3f}  {val_mae:>8.3f}        "
          f"{test_acc:>8.3f} / {1-test_acc:>5.3f}  {test_mae:>8.3f}")
    
    individual_results.append({
        'Group': group_name,
        'Category': category,
        'Num_Features': len(features),
        'Features': ', '.join(features),  # Save feature list as comma-separated string
        'Val_Winner_Acc': val_acc,
        'Val_Gap_MAE': val_mae,
        'Test_Winner_Acc': test_acc,
        'Test_Gap_MAE': test_mae
    })

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================

print("\n" + "="*120)
print("SAVING RESULTS")
print("="*120)

# Convert results to DataFrame
results_df = pd.DataFrame(individual_results)

# Sort by test accuracy (descending)
results_df = results_df.sort_values('Test_Winner_Acc', ascending=False)

# Save to CSV
output_file = 'results/exp42_feature_evaluation.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")
print(f"Total feature groups evaluated: {len(results_df)}")

# Also save the actual feature values for each task
print("\nSaving task-level feature data...")
feature_columns = []
for group_name, features in available_groups.items():
    feature_columns.extend(features)
feature_columns = list(set(feature_columns))  # Remove duplicates

# Create task features DataFrame with task identifiers
task_features_df = df[['dataset_name', 'task_name', 'task_type', 'winner_by_val', 'winner_by_test',
                        'gap_by_val', 'gap_by_test'] + [f for f in feature_columns if f in df.columns]]

task_output_file = 'results/exp42_task_features.csv'
task_features_df.to_csv(task_output_file, index=False)
print(f"Task-level features saved to: {task_output_file}")
print(f"Total tasks: {len(task_features_df)}")
print(f"Total features saved: {len([f for f in feature_columns if f in df.columns])}")

print("\n" + "="*120)
print("EVALUATION COMPLETE")
print("="*120)

# ============================================================================
# STEP 5: ABLATION STUDY - IMPACT OF TRAINING SET SIZE
# ============================================================================

print("\n" + "="*120)
print("ABLATION STUDY: IMPACT OF TRAINING SET SIZE")
print("Sampling different numbers of tasks to study generalization")
print("="*120)

def sample_balanced_tasks(df, n_tasks, seed):
    """
    Sample n_tasks with balanced classification/regression split (1:1 ratio).
    """
    np.random.seed(seed)

    # Separate classification and regression tasks
    class_tasks = df[df['task_type'].str.contains('classification')].copy()
    reg_tasks = df[df['task_type'] == 'regression'].copy()

    # Calculate how many of each type to sample
    n_class = n_tasks // 2
    n_reg = n_tasks - n_class

    # Sample with replacement if necessary
    if len(class_tasks) < n_class:
        sampled_class = class_tasks.sample(n=n_class, replace=True, random_state=seed)
    else:
        sampled_class = class_tasks.sample(n=n_class, replace=False, random_state=seed)

    if len(reg_tasks) < n_reg:
        sampled_reg = reg_tasks.sample(n=n_reg, replace=True, random_state=seed)
    else:
        sampled_reg = reg_tasks.sample(n=n_reg, replace=False, random_state=seed)

    return pd.concat([sampled_class, sampled_reg]).sample(frac=1, random_state=seed)

def ablation_evaluate(train_df, test_df, features, model_type='rf'):
    """
    Train on train_df and evaluate on test_df for winner prediction.
    Returns test accuracy.
    """
    if len(features) == 0:
        return None

    # Prepare training data
    X_train = train_df[features].values
    y_train = train_df['winner_by_test'].values

    # Prepare test data
    X_test = test_df[features].values
    y_test = test_df['winner_by_test'].values

    # Convert to DataFrame for NaN handling
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    # Fill NaN values with median from training set
    for col in X_train_df.columns:
        train_median = X_train_df[col].median()
        X_train_df[col].fillna(train_median, inplace=True)
        X_test_df[col].fillna(train_median, inplace=True)

    # Remove constant features
    non_constant = X_train_df.std() > 0
    X_train_df = X_train_df.loc[:, non_constant]
    X_test_df = X_test_df.loc[:, non_constant]

    if len(X_train_df.columns) == 0 or len(X_train_df) < 3:
        return None

    # Handle y NaN values
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    y_train_series.fillna(y_train_series.median(), inplace=True)
    y_test_series.fillna(y_test_series.median(), inplace=True)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # Train model
    if model_type == 'dt':
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    elif model_type == 'nn':
        model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42,
                             early_stopping=True, validation_fraction=0.2)
    elif model_type == 'tabpfn':
        if not TABPFN_AVAILABLE:
            return None
        model = TabPFNClassifier(device=DEVICE, N_ensemble_configurations=4)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    try:
        model.fit(X_train_scaled, y_train_series)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_series, y_pred)
        return accuracy
    except:
        return None

# Run ablation study
ablation_results = []
task_sizes = [8, 12, 16]  # Number of tasks to sample
n_trials = 5  # Number of random samplings
model_types = ['rf', 'dt', 'gb', 'nn']  # Models to evaluate

print("\nRunning ablation study...")
print(f"Task sizes: {task_sizes}")
print(f"Trials per size: {n_trials}")
print(f"Models: {model_types}")
print(f"Feature groups: {list(available_groups.keys())}")
print()

for group_name, features in available_groups.items():
    if len(features) == 0:
        continue

    print(f"Evaluating {group_name} ({len(features)} features)...")

    for n_tasks in task_sizes:
        for model_type in model_types:
            trial_accuracies = []

            for trial in range(n_trials):
                # Sample training tasks (balanced class/reg)
                train_sample = sample_balanced_tasks(df, n_tasks, seed=42 + trial)

                # Use remaining tasks as test set
                test_sample = df[~df.index.isin(train_sample.index)]

                if len(test_sample) < 2:
                    continue

                # Evaluate
                acc = ablation_evaluate(train_sample, test_sample, features, model_type)

                if acc is not None:
                    trial_accuracies.append(acc)

            if trial_accuracies:
                ablation_results.append({
                    'Group': group_name,
                    'Num_Features': len(features),
                    'Train_Size': n_tasks,
                    'Model': model_type,
                    'Mean_Accuracy': np.mean(trial_accuracies),
                    'Std_Accuracy': np.std(trial_accuracies),
                    'Min_Accuracy': np.min(trial_accuracies),
                    'Max_Accuracy': np.max(trial_accuracies),
                    'Num_Trials': len(trial_accuracies)
                })

# Convert to DataFrame and save
ablation_df = pd.DataFrame(ablation_results)

if not ablation_df.empty:
    # Sort by mean accuracy
    ablation_df = ablation_df.sort_values(['Train_Size', 'Mean_Accuracy'], ascending=[True, False])

    # Save results
    ablation_output_file = 'results/exp42_ablation_study.csv'
    ablation_df.to_csv(ablation_output_file, index=False)
    print(f"\nAblation study results saved to: {ablation_output_file}")

    # Print summary statistics
    print("\n" + "="*120)
    print("ABLATION STUDY SUMMARY")
    print("="*120)

    for n_tasks in task_sizes:
        print(f"\n--- Training Size: {n_tasks} tasks ---")
        subset = ablation_df[ablation_df['Train_Size'] == n_tasks]

        # Best performance per model type
        print(f"\nBest performance by model type:")
        for model in model_types:
            model_subset = subset[subset['Model'] == model]
            if not model_subset.empty:
                best_row = model_subset.nlargest(1, 'Mean_Accuracy').iloc[0]
                print(f"  {model:10s}: {best_row['Mean_Accuracy']:.3f} ± {best_row['Std_Accuracy']:.3f} "
                      f"({best_row['Group']})")

        # Best overall
        if not subset.empty:
            best_overall = subset.nlargest(1, 'Mean_Accuracy').iloc[0]
            print(f"\n  Best Overall: {best_overall['Mean_Accuracy']:.3f} ± {best_overall['Std_Accuracy']:.3f}")
            print(f"    Group: {best_overall['Group']}")
            print(f"    Model: {best_overall['Model']}")
            print(f"    Features: {best_overall['Num_Features']}")

    # Create visualization
    print("\n" + "="*120)
    print("Creating ablation study visualization...")
    print("="*120)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ablation Study: Impact of Training Set Size', fontsize=16, fontweight='bold')

    # Plot 1: Performance vs Training Size (by feature group)
    ax1 = axes[0, 0]
    for group in ablation_df['Group'].unique():
        group_data = ablation_df[ablation_df['Group'] == group]
        group_mean = group_data.groupby('Train_Size')['Mean_Accuracy'].mean()
        ax1.plot(group_mean.index, group_mean.values, marker='o', label=group, linewidth=2)
    ax1.set_xlabel('Training Set Size (number of tasks)')
    ax1.set_ylabel('Mean Accuracy')
    ax1.set_title('Performance vs Training Size by Feature Group')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(task_sizes)

    # Plot 2: Performance vs Training Size (by model type)
    ax2 = axes[0, 1]
    for model in model_types:
        model_data = ablation_df[ablation_df['Model'] == model]
        model_mean = model_data.groupby('Train_Size')['Mean_Accuracy'].mean()
        ax2.plot(model_mean.index, model_mean.values, marker='s', label=model, linewidth=2)
    ax2.set_xlabel('Training Set Size (number of tasks)')
    ax2.set_ylabel('Mean Accuracy')
    ax2.set_title('Performance vs Training Size by Model Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(task_sizes)

    # Plot 3: Variance vs Training Size
    ax3 = axes[1, 0]
    for group in ablation_df['Group'].unique():
        group_data = ablation_df[ablation_df['Group'] == group]
        group_std = group_data.groupby('Train_Size')['Std_Accuracy'].mean()
        ax3.plot(group_std.index, group_std.values, marker='o', label=group, linewidth=2)
    ax3.set_xlabel('Training Set Size (number of tasks)')
    ax3.set_ylabel('Std Deviation of Accuracy')
    ax3.set_title('Stability vs Training Size (Lower is Better)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(task_sizes)

    # Plot 4: Heatmap of best configurations
    ax4 = axes[1, 1]
    pivot_data = ablation_df.pivot_table(
        values='Mean_Accuracy',
        index='Group',
        columns='Train_Size',
        aggfunc='max'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax4,
                cbar_kws={'label': 'Max Accuracy'}, vmin=0.5, vmax=1.0)
    ax4.set_title('Best Accuracy by Group and Training Size')
    ax4.set_xlabel('Training Set Size')
    ax4.set_ylabel('Feature Group')

    plt.tight_layout()
    ablation_plot_file = 'results/exp42_ablation_study.png'
    plt.savefig(ablation_plot_file, dpi=150, bbox_inches='tight')
    print(f"Ablation study plot saved to: {ablation_plot_file}")

    # Additional analysis: Learning curves
    print("\n" + "="*120)
    print("LEARNING CURVE ANALYSIS")
    print("="*120)

    for group in ablation_df['Group'].unique():
        group_data = ablation_df[ablation_df['Group'] == group]
        group_by_size = group_data.groupby('Train_Size')['Mean_Accuracy'].agg(['mean', 'max'])

        print(f"\n{group}:")
        for size in task_sizes:
            if size in group_by_size.index:
                mean_acc = group_by_size.loc[size, 'mean']
                max_acc = group_by_size.loc[size, 'max']
                print(f"  {size:2d} tasks: mean={mean_acc:.3f}, max={max_acc:.3f}")

        # Calculate improvement rate
        if len(group_by_size) >= 2:
            improvement = (group_by_size['mean'].iloc[-1] - group_by_size['mean'].iloc[0]) / group_by_size['mean'].iloc[0] * 100
            print(f"  Improvement: {improvement:+.1f}% from {task_sizes[0]} to {task_sizes[-1]} tasks")

else:
    print("\nNo ablation results generated (insufficient data)")

print("\n" + "="*120)
print("ALL EXPERIMENTS COMPLETE")
print("="*120)
