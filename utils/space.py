import random

RDL_SEARCH_SPACE = {
    "full_entities": {
        'pre_sf': ['none', 'zero_learn'],
        'mpnn_type': ['relgnn', 'sage', 'hgt', 'pna'],
        'post_sf': {'link': ['none', 'shallow', 'contextgnn'], 'node': ['none']}
    },
    "model_config": {
        "encoder_num_layers": [4], 
        "torch_frame_model_cls": ['resnet'], 
        "batch_size": [128, 256],
        "gnn_config": {
            "loss_fn": {
                "binary_classification": ["bce"], 
                "regression": ["mae"], 
                "recommendation": ['bpr'],
                "multiclass_classification": ["ce"]
            },
            "hidden_channels": [64, 128, 256], 
            "num_heads": [1, 4],
            "dropout": [0.0, 0.0, 0.0, 0.5], 
            "norm": ["layernorm", "batchnorm", 'none'], 
            "aggregation": ["mean", "sum"], 
            "jk": [False],
            "skip_connection": [True, False, False]
        }, 
        "sampler_config": {
            "temporal": ["uniform", "last"], 
            "num_neighbors": [32, 64, 128],
            "num_layers": [1, 2, 3, 4],
            "loader_type": ["node", "edge"]
        }, 
        "optimizer_config": {
            ## 1e-2 is for trial task, but don't make frequency too high
            "lr": [1e-3, 1e-4, 1e-2, 1e-3, 1e-4], 
            "weight_decay": [0.0, 1e-5],  
            "scheduler": ["expontential"], 
            "gamma": [.8, .9, 1.]
        }
    }
}

DFS_SEARCH_SPACE = {
    "full_entities": {
        'dfs_layer': [1, 2],
        'text_to_pca': [True]
    },
    # "pre_defined_search_time": {
    #     ## these are 50 times
    #     "resnet": 50, 
    #     "ft_transformer": 50, 
    # },
    "model_type": ["tabpfn", "resnet", "ft_transformer", "lgbm"],
    "pre_feature": ["cn", "none", "hash"],
    "batch_size": [64, 128, 256],
    "negative_sampling_ratio": [5, 10, 20],
    "model_config": {
        "hidden_size": [128, 256, 512],
        "dropout": [0.0, 0.5, 0.3],
        "num_layers": [1, 2, 3, 4],
        "attn_dropout": [0.0, 0.5],
        "num_heads": [1, 4, 8],
        "normalization": ["layernorm", "batchnorm", 'none'],
        "loss_fn": {
            "binary_classification": ["bce"],
            "regression": ["mae", 'mse'],
            "ranking": ['bce', 'bpr'],
            "multiclass_classification": ["ce"]
        }
    }, 
    "auto_config": {
        "max_train_samples": [None],
        "training_sample_selection_strategy": ['graph', 'random', 'stratified'],
        "num_trials": [30],
        "temporal": ["delete_both", "delete_add", "delete_none"] 
    }, 
    "optimizer_config": {
        "lr": [1e-3, 1e-4, 5e-4, 5e-3], 
        "weight_decay": [0.0, 1e-5, 5e-4],  
        "scheduler": ["expontential"], 
        "gamma": [1., 0.8, 0.9]
    }
} 


def search_rdl(task_type: str, pre_sf: str, mpnn_type: str, post_sf: str, minimum_layers = 0):
    """
    Performs a random search over the RDL_SEARCH_SPACE to generate a single,
    valid hyperparameter configuration.

    This function respects the hierarchical and conditional nature of the search space.
    For example, the choice for 'mpnn_design_choice' is dependent on the value
    sampled for 'mpnn_type'. Similarly, the 'loss_fn' is chosen based on a
    randomly selected task type ('classification', 'regression', or 'ranking').
    don't search for full_entities

    Returns:
        dict: A dictionary containing a complete, randomly sampled model configuration
              that maintains the original structure of the search space.
    """
    global RDL_SEARCH_SPACE

    # Get references to the specific configuration spaces
    model_space = RDL_SEARCH_SPACE["model_config"]
    gnn_space = model_space["gnn_config"].copy()
    sampler_space = model_space["sampler_config"].copy()
    if minimum_layers > 0:
        sampler_space["num_layers"] = [i for i in sampler_space["num_layers"] if i >= minimum_layers]
    optimizer_space = model_space["optimizer_config"]

    # --- 1. Sample GNN Configuration ---
    sampled_gnn_config = {}

    # Sample standard parameters
    for key in ["hidden_channels", "num_heads", "dropout", "norm", "aggregation", "jk", "skip_connection"]:
        sampled_gnn_config[key] = random.choice(gnn_space[key])
    sampled_gnn_config['pre_sf'] = pre_sf
    sampled_gnn_config['mpnn_type'] = mpnn_type
    sampled_gnn_config['post_sf'] = post_sf
    # Handle conditional parameter: mpnn_type -> mpnn_design_choice
    # selected_mpnn_type = random.choice(gnn_space["mpnn_type"])
    # sampled_gnn_config["mpnn_type"] = selected_mpnn_type
    
    # # Based on the selected mpnn_type, choose from the corresponding design choices
    # design_choices = gnn_space["mpnn_design_choice"][selected_mpnn_type]
    # sampled_gnn_config["mpnn_design_choice"] = random.choice(design_choices)

    
    # Now, select a loss function valid for that task
    loss_fn_choices = gnn_space["loss_fn"][task_type]
    sampled_gnn_config["loss_fn"] = random.choice(loss_fn_choices)


    # --- 2. Sample Sampler Configuration ---
    sampled_sampler_config = {}
    for key in sampler_space:
        sampled_sampler_config[key] = random.choice(sampler_space[key])
    
    ## if recommendation, both node and link is ok, other wise only node is ok
    if task_type == "recommendation" and post_sf == 'none':
        sampled_sampler_config['loader_type'] = 'link'
    else:
        sampled_sampler_config['loader_type'] = 'node'
    

    # --- 3. Sample Optimizer Configuration ---
    sampled_optimizer_config = {}
    for key in optimizer_space:
        sampled_optimizer_config[key] = random.choice(optimizer_space[key])

    random_batch_size = random.choice(model_space["batch_size"])
    share_same_time = True
    negative_sampling_ratio = 1
    # --- 4. Assemble the Final Configuration ---
    final_config = {
        # Select from fixed top-level parameters
        "encoder_num_layers": random.choice(model_space["encoder_num_layers"]),
        # "channels": sampled_gnn_config["hidden_channels"],
        "torch_frame_model_cls": random.choice(model_space["torch_frame_model_cls"]),
        "batch_size": random_batch_size,
        "eval_batch_size": random_batch_size * 2,
        # Assign the sampled nested configurations
        "gnn_config": sampled_gnn_config,
        "sampler_config": sampled_sampler_config,
        "optimizer_config": sampled_optimizer_config,
        "num_negative_samples": negative_sampling_ratio,
        "share_same_time": share_same_time
    }

    return final_config

def search_dfs(task_type: str, selected_model: str = 'lightgbm'):
    global DFS_SEARCH_SPACE
    # Sample from predefined model types
    # selected_model = 'lightgbm'
    # selected_pre_feature = random.choice(DFS_SEARCH_SPACE["pre_feature"])
    
    base_config = {
        "torch_frame_model_cls": selected_model,
        # "pre_feature": selected_pre_feature,
        # "dfs_layer": random.choice(DFS_SEARCH_SPACE["full_entities"]["dfs_layer"]),
        "batch_size": random.choice(DFS_SEARCH_SPACE["batch_size"]),
        "num_trials": 20,
    }

    if selected_model in ['resnet', 'ft_transformer']:
        # Sample optimizer config
        optimizer_config = {
        "lr": random.choice(DFS_SEARCH_SPACE["optimizer_config"]["lr"]),
        "weight_decay": random.choice(DFS_SEARCH_SPACE["optimizer_config"]["weight_decay"]),
        "scheduler": random.choice(DFS_SEARCH_SPACE["optimizer_config"]["scheduler"]),
        "gamma": random.choice(DFS_SEARCH_SPACE["optimizer_config"]["gamma"])
        }
        base_config["optimizer_config"] = optimizer_config

    # Configure model-specific parameters
    if selected_model in ['resnet', 'ft_transformer']:
        tab_nn_config = {
            "hidden_size": random.choice(DFS_SEARCH_SPACE["model_config"]["hidden_size"]),
            "dropout": random.choice(DFS_SEARCH_SPACE["model_config"]["dropout"]),
            "num_layers": random.choice(DFS_SEARCH_SPACE["model_config"]["num_layers"]),
            "attn_dropout": random.choice(DFS_SEARCH_SPACE["model_config"]["attn_dropout"]),
            "num_heads": random.choice(DFS_SEARCH_SPACE["model_config"]["num_heads"]),
            "normalization": random.choice(DFS_SEARCH_SPACE["model_config"]["normalization"])
        }
        base_config["tab_nn_config"] = tab_nn_config
    
    if task_type == 'binary_classification':
        base_config["loss_fn"] = random.choice(DFS_SEARCH_SPACE["model_config"]["loss_fn"]["binary_classification"])
    elif task_type == 'regression':
        base_config["loss_fn"] = random.choice(DFS_SEARCH_SPACE["model_config"]["loss_fn"]["regression"])
    elif task_type == 'retrieval':
        base_config["loss_fn"] = random.choice(DFS_SEARCH_SPACE["model_config"]["loss_fn"]["ranking"])
    elif task_type == 'multiclass_classification':
        base_config["loss_fn"] = random.choice(DFS_SEARCH_SPACE["model_config"]["loss_fn"]["multiclass_classification"])
    else:
        raise ValueError(f"Invalid task type: {task_type}")
    
    return base_config

def search_ng(task_type: str):
    dfs = search_dfs(task_type)
    dfs.pop('pre_feature', None)
    dfs.pop('dfs_layer', None)
    return dfs

def search(model_type: str = "rdl", task_type: str = "classification", pre_sf: str = 'src_dst', 
    mpnn_type: str = 'relgnn', post_sf: str = 'none', minimum_layers = 0):
    if model_type == "rdl":
        res = search_rdl(task_type, pre_sf, mpnn_type, post_sf, minimum_layers)
        res['model_type'] = 'rdl'
    elif model_type == "fs":
        res = search_dfs(task_type)
        res['model_type'] = 'fs'
    elif model_type == "ng":
        res = search_ng(task_type)
        res['model_type'] = 'ng'
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return res


