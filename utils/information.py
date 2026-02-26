from copy import deepcopy

task_name_to_metric = {
    'driver-top3': 'val_roc_auc',
    'driver-dnf': 'val_roc_auc',
    'driver-position': 'val_mae',
    'user-item-purchase': 'val_map',
    'user-churn': 'val_roc_auc',
    'item-sales': 'val_mae',
    'study-outcome': 'val_roc_auc',
    'study-adverse': 'val_mae',
    'site-success': 'val_mae',
    'condition-sponsor-run': 'val_map',
    'site-sponsor-run': 'val_map',
    'user-engagement': 'val_roc_auc',
    'user-badge': 'val_roc_auc',
    'post-votes': 'val_mae',
    'user-post-comment': 'val_map',
    'post-post-related': 'val_map',
    'user-repeat': 'val_roc_auc',
    'user-ignore': 'val_roc_auc',
    'user-attendance': 'val_mae',
    'user-visits': 'val_roc_auc',
    'user-clicks': 'val_roc_auc',
    'ad-ctr': 'val_mae',
    'user-ad-visit': 'val_map',
    'repeater': 'val_roc_auc',
    'results-position': 'val_mae',
    'event_interest-interested': 'val_roc_auc',
    'searchstream-click': 'val_roc_auc',
    'searchinfo-isuserloggedon': 'val_roc_auc',
    'badges-class': 'val_roc_auc',
    'domain-ip': 'val_roc_auc',
    'prepay': 'val_accuracy',
    'charge': 'val_accuracy',
    'transaction-fraud': 'val_roc_auc',
    'driver-wins': 'val_mae',
    'constructor-scores-points': 'val_roc_auc',
    'driver-position-change': 'val_mae',
    'customer-spending': 'val_mae',
    'customer-average-price': 'val_mae',
    'user-comment-count': 'val_mae',
}

task_name_to_metric_no_prefix = {
    'driver-top3': 'roc_auc',
    'driver-dnf': 'roc_auc',
    'driver-position': 'mae',
    'user-item-purchase': 'map',
    'user-churn': 'roc_auc',
    'item-sales': 'mae',
    'study-outcome': 'roc_auc',
    'study-adverse': 'mae',
    'site-success': 'mae',
    'condition-sponsor-run': 'map',
    'site-sponsor-run': 'map',
    'user-engagement': 'roc_auc',
    'user-badge': 'roc_auc',
    'post-votes': 'mae',
    'user-post-comment': 'map',
    'post-post-related': 'map',
    'user-repeat': 'roc_auc',
    'user-ignore': 'roc_auc',
    'user-attendance': 'mae',
    'user-visits': 'roc_auc',
    'user-clicks': 'roc_auc',
    'ad-ctr': 'mae',
    'user-ad-visit': 'map',
    'repeater': 'roc_auc',
    'results-position': 'mae',
    'event_interest-iterested': 'roc_auc',
    'searchstream-click': 'roc_auc',
    'searchinfo-isuserloggedon': 'roc_auc',
    'badges-class': 'accuracy',
    'domain-ip': 'roc_auc',
    'prepay': 'accuracy',
    'charge': 'accuracy',
    'transaction-fraud': 'roc_auc',
    'driver-wins': 'mae',
    'constructor-scores-points': 'roc_auc',
    'driver-position-change': 'mae',
    'customer-spending': 'mae',
    'customer-average-price': 'mae',
    'user-comment-count': 'mae',
    'destination': 'roc_auc',
    'paper-citation': 'roc_auc',
    'author-publication': 'mae',
    'item-churn': 'roc_auc',
    'user-ltv': 'mae',
    'item-ltv': 'mae',
}

NEEDED_TASKS = ['driver-top3', 'driver-dnf', 'driver-position', 'user-item-purchase', 'user-churn', 'item-sales', 'study-outcome', 'study-adverse', 'site-success', 'condition-sponsor-run', 'site-sponsor-run', 'user-engagement', 'user-badge', 'post-votes', 'user-post-comment', 'post-post-related', 'user-repeat', 'user-ignore', 'user-attendance', 'user-visits', 'user-clicks', 'ad-ctr', 'user-ad-visit', 'repeater', 'transaction-fraud',  'constructor-scores-points', 'driver-position-change', 'user-comment-count', 'paper-citation', 'author-publication']

ABALATION_TASKS = ['driver-top3', 'driver-dnf', 'driver-position', 'user-churn', 'item-sales', 'study-outcome', 'study-adverse', 'site-success', 'user-engagement', 'user-badge', 'post-votes', 'user-repeat', 'user-ignore', 'user-attendance', 'user-visits', 'user-clicks', 'ad-ctr']

task_name_to_metric_prefix_rdl = deepcopy(task_name_to_metric_no_prefix)
for key, value in task_name_to_metric_prefix_rdl.items():
    if value == 'map':
        task_name_to_metric_prefix_rdl[key] = 'link_prediction_map'



task_name_to_task_type = {
    'driver-top3': 'binary_classification',
    'driver-dnf': 'binary_classification',
    'driver-position': 'regression',
    'user-item-purchase': 'recommendation',
    'user-churn': 'binary_classification',
    'item-sales': 'regression',
    'post-votes': 'regression',
    'study-outcome': 'binary_classification',
    'study-adverse': 'regression',
    'site-success': 'regression',
    'condition-sponsor-run': 'recommendation',
    'site-sponsor-run': 'recommendation',
    'user-engagement': 'binary_classification',
    'user-badge': 'binary_classification',
    'user-post-comment': 'recommendation',
    'post-post-related': 'recommendation',
    'user-repeat': 'binary_classification',
    'user-ignore': 'binary_classification',
    'user-attendance': 'regression',
    'user-visits': 'binary_classification',
    'user-clicks': 'binary_classification',
    'ad-ctr': 'regression',
    'user-ad-visit': 'recommendation',
    'repeater': 'auto_binary_classification',
    'results-position': 'auto_regression',
    'event_interest-iterested': 'auto_binary_classification',
    'searchstream-click': 'auto_binary_classification',
    'searchinfo-isuserloggedon': 'auto_binary_classification',
    'badges-class': 'auto_binary_classification',
    'domain-ip': 'auto_binary_classification',
    'prepay': 'auto_multi_classification',
    'charge': 'auto_multi_classification',
    'transaction-fraud': 'auto_binary_classification',
    'driver-wins': 'regression',
    'constructor-scores-points': 'binary_classification',
    'driver-position-change': 'regression',
    'customer-spending': 'regression',
    'customer-average-price': 'regression',
    'user-comment-count': 'regression',
    'destination': 'auto_binary_classification',
    'paper-citation': 'binary_classification',
    'author-publication': 'regression',
    'item-churn': 'binary_classification',
    'user-ltv': 'regression',
    'item-ltv': 'regression',
}

REGRESSION_CAN_DO_THREE = ['driver-position', 'study-adverse', 'site-success', 'driver-wins', 'driver-position-change', 'item-sales', 'user-attendance']

CLASSIFICATION_CAN_DO_THREE = ['study-outcome', 'repeater', 'driver-dnf', 'user-clicks', 'user-visits', 'driver-top3']

ALL_CAN_DO_THREE = REGRESSION_CAN_DO_THREE + CLASSIFICATION_CAN_DO_THREE

SUMMARY_TASKS = {
    'rel-avs': ['repeater'],
    'rel-ieeecis': ['transaction-fraud'],
    'rel-f1': ['driver-dnf', 'driver-top3', 'driver-position', 'constructor-scores-points', 'driver-position-change'],
    'rel-trial': ['study-adverse', 'site-success', 'study-outcome'],
    'rel-stack': ['user-engagement', 'user-badge', 'post-votes', 'user-comment-count'],
    'rel-avito': ['user-visits', 'user-clicks', 'ad-ctr'],
    'rel-hm': ['user-churn', 'item-sales'],
    'rel-event': ['user-repeat', 'user-ignore', 'user-attendance'],
    'rel-arxiv': ['paper-citation', 'author-publication'],
}

SUMMARY_DATASETS = list(SUMMARY_TASKS.keys())
SUMMARY_TASK_LIST = [task for tasks in SUMMARY_TASKS.values() for task in tasks]


relbench_lgbm_cared_columns = ['dfs_layer']

rdl_cared_columns = ['lr', 'gamma', 'weight_decay', 'num_layers', 'num_neighbors', 'norm', 'pre_sf', 'post_sf', 'dropout', 'aggregation', 'hidden_channels', 'mpnn_type', 'channels', 'batch_size', 'scheduler', 
'temporal', 'num_heads', 'skip_connection', 'loss_fn', 'loader_type']

relbench_tabpfn_cared_columns = ['dfs_layer', 'temporal', 'use_auto_tabpfn']

relbench_ft_transformer_cared_columns = ['lr', 'gamma', 'weight_decay', 'num_layers', 'dropout', 'hidden_channels', 'batch_size', 'scheduler', 
                      'dfs_layer', 'max_steps_per_epoch',
                      'num_heads', 'attn_dropout', 'norm', 'loss_fn', 'torch_frame_model_cls']


## make order matches with this
ALL_TASK_LIST = [
    'user-churn',
    'study-outcome',
    'repeater',
    'item-sales',
    'user-clicks',
    'user-visits',
    'ad-ctr',
    'site-success',
    'study-adverse',
    'user-attendance',
    'user-repeat',
    'user-ignore',
    'user-badge',
    'user-engagement',
    'post-votes',
    'driver-dnf',
    'driver-top3',
    'driver-position',
    'transaction-fraud',
    'constructor-scores-points',
    'driver-position-change',
    'customer-spending',
    'customer-average-price',
    'user-comment-count',
    'destination',
    'paper-citation',
    'author-publication'
]

CLASSIFICATION_TASK_LIST = [
    'user-churn',
    'study-outcome',
    'repeater',
    'driver-dnf',
    'user-clicks',
    'user-badge',
    'user-engagement',
    'transaction-fraud', 
    'user-ignore',
    'user-repeat',
    'user-visits',
    'driver-top3',
    'paper-citation',
    'constructor-scores-points',
]

PRETRAINED_TASK_LIST = [
    'user-churn',
    'study-outcome',
    'repeater',
    'item-sales',
    'user-clicks',
    'user-visits',
    'ad-ctr',
    'site-success',
    'study-adverse',
    'user-attendance',
    'user-repeat',
    'user-ignore',
    'user-badge', 
    'user-engagement',
    'post-votes',
    'driver-dnf',
    'transaction-fraud',
    'constructor-scores-points',
    'driver-position-change',
    'user-comment-count',
    'author-publication'
]

EVAL_TASK_LIST = [
    ('rel-f1', 'driver-top3'),
    ('rel-f1', 'driver-position'),
    ('rel-hm', 'user-churn'),
    ('rel-f1', 'driver-dnf'),
]

RDL_CONSIDERED_PARAMS = [
    'batch_size', 'gnn.pre_sf', 'gnn.post_sf', 'gnn.hidden_channels', 'gnn.num_heads', 'gnn.dropout', 'gnn.norm', 'gnn.aggregation', 'gnn.skip_connection', 'gnn.mpnn_type', 'sampler_config.num_layers', 'sampler_config.num_neighbors', 'optimizer_config.lr', 'optimizer_config.weight_decay', 'optimizer_config.gamma'
]

DFS_FTT_CONSIDERED_PARAMS = [
    'dfs_layer', 'tab_nn_config.hidden_size', 'tab_nn_config.dropout', 'tab_nn_config.num_layers', 'tab_nn_config.attn_dropout', 'tab_nn_config.num_heads', 'tab_nn_config.normalization', 'optimizer_config.lr', 'optimizer_config.weight_decay', 'optimizer_config.gamma'
]

rdl_column_name_mapping = {
    'batch_size': 'batch_size',
    'pre_sf': 'gnn.pre_sf',
    'post_sf': 'gnn.post_sf',
    'hidden_channels': 'gnn.hidden_channels',
    'num_heads': 'gnn.num_heads',
    'dropout': 'gnn.dropout',
    'norm': 'gnn.norm',
    'lr': 'optimizer_config.lr',
    'gamma': 'optimizer_config.gamma',
    'weight_decay': 'optimizer_config.weight_decay',
    'num_layers': 'sampler_config.num_layers',
    'num_neighbors': 'sampler_config.num_neighbors',
    'mpnn_type': 'gnn.mpnn_type',
    'aggregation': 'gnn.aggregation',
    'skip_connection': 'gnn.skip_connection',
    'temporal': 'sampler_config.temporal',
    'loader_type': 'sampler_config.loader_type',
    'scheduler': 'optimizer_config.scheduler',
}

ftt_column_name_mapping = {
    'dfs_layer': 'dfs_layer',
    'hidden_size': 'tab_nn_config.hidden_size',
    'dropout': 'tab_nn_config.dropout',
    'num_layers': 'tab_nn_config.num_layers',
    'attn_dropout': 'tab_nn_config.attn_dropout',
    'num_heads': 'tab_nn_config.num_heads',
    'norm': 'tab_nn_config.normalization',
    'lr': 'optimizer_config.lr',
    'gamma': 'optimizer_config.gamma',
    'weight_decay': 'optimizer_config.weight_decay',
    'scheduler': 'optimizer_config.scheduler',
}
