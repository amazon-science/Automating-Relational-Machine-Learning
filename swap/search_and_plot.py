import pandas as pd
from swap.hpo_exec import execution
from models.automl.simplelandscape import GNNLossLandscape, FTTLossLandscape
from utils.misc import load_yaml
import torch
import os
from data.mtaskdataset import prepare_transductive_rdb_datasets_save_memory, prepare_transductive_rdb_datasets
from loguru import logger
from copy import deepcopy
from utils.database import create_swap_db
from swap.execution import get_auto_model, configure_rdl
from models.dfs import generate_dfs_data_pre_post
from torch_frame.data import DataLoader
from dbinfer.solutions.rel_tabnn_solutions import check_task_type, convert_all_splits

def load_model_performance_bank(file_path = './validated_runs.csv'):
    model_performance_bank = pd.read_csv(file_path)
    return model_performance_bank

def load_top_models(model_performance_bank, split = 'val', task_name = 'driver-dnf', model_type = 'rdl'):
    top_models = model_performance_bank[model_performance_bank['task_name'] == task_name]
    top_models = top_models[top_models['model_type'] == model_type]
    task_type = top_models['task_type'].iloc[0]
    top_models = top_models.sort_values(by=f'{split}_metric', ascending=False if 'classification' in task_type else True)
    top_models = top_models.head(1)
    return top_models

def wrap_searched_configurations_to_rdl_configs(search_configs):
    template = './configs/model/rdl_test.yaml'
    template_dict = load_yaml(template)
    template_dict['model_type'] = 'rdl'
    template_dict['batch_size'] = int(search_configs['batch_size'].item())
    template_dict['eval_batch_size'] = int(search_configs['batch_size'].item() * 2)
    for key, value in template_dict['gnn_config'].items():
        if key in search_configs:
            template_dict['gnn_config'][key] = search_configs[key].item()
            if key == 'hidden_channels':
                template_dict['gnn_config'][key] = int(search_configs[key].item())
            if key == 'num_heads':
                template_dict['gnn_config'][key] = int(search_configs[key].item())
    template_dict['sampler_config']['num_neighbors'] = int(search_configs['num_neighbors'].item())
    template_dict['sampler_config']['num_layers'] = int(search_configs['num_layers'].item())
    template_dict['sampler_config']['loader_type'] = search_configs['loader_type'].item()
    template_dict['sampler_config']['temporal'] = search_configs['temporal'].item()
    template_dict['optimizer_config']['lr'] = search_configs['lr'].item()
    template_dict['optimizer_config']['weight_decay'] = search_configs['weight_decay'].item()
    template_dict['optimizer_config']['gamma'] = search_configs['gamma'].item()
    return template_dict


def wrap_searched_configurations_to_dfs_configs(search_configs):
    template = './configs/model/ftt.yaml'
    template_dict = load_yaml(template)
    template_dict['model_type'] = 'dfs'
    template_dict['batch_size'] = search_configs['batch_size'].item()
    template_dict['eval_batch_size'] = search_configs['batch_size'].item() * 2
    template_dict['dfs_layer'] = int(search_configs['dfs_layer'].item())
    ## configure loss function
    template_dict['loss_fn'] = {
        'binary_classification': 'bce',
        'multiclass_classification': 'ce',
        'regression': 'mse',
        'auto_binary_classification': 'bce',
        'auto_multiclass_classification': 'ce',
        'auto_regression': 'mae'
    }[search_configs['task_type'].item()] if search_configs['loss_fn'].item() not in ['bce', 'ce', 'mse', 'mae'] else search_configs['loss_fn'].item()
    for key, value in template_dict['tab_nn_config'].items():
        if key in search_configs:
            template_dict['tab_nn_config'][key] = search_configs[key].item()
            if key == 'hidden_size':
                template_dict['tab_nn_config'][key] = int(search_configs[key].item())
            if key == 'num_layers':
                template_dict['tab_nn_config'][key] = int(search_configs[key].item())
            if key == 'num_heads':
                template_dict['tab_nn_config'][key] = int(search_configs[key].item())
    template_dict['optimizer_config']['lr'] = search_configs['lr'].item()
    template_dict['optimizer_config']['weight_decay'] = search_configs['weight_decay'].item()
    template_dict['optimizer_config']['gamma'] = search_configs['gamma'].item()
    return template_dict

def create_model(model_config, database):
    model = get_auto_model(model_config['model_params']["model_type"])
    model_init = {**model_config['model_params']}
    model_init['col_stats_dict'] = database.get_col_stats_dict()
    task_info = database.tasks[0]
    ## convert task type to string for easier manipulation
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
    return model


def plot_loss_landscape_and_calculate_metrics(search_configs, model_weight_name, model_type='rdl', 
                                            dataset=None, device='cuda', task_type=None, num_classes=None, task_name=None,
                                            train_loader=None, val_loader=None, test_loader=None,
                                            train_dataset=None, scaler_path=None, clamp_min=None, clamp_max=None, plot_name=None):
    """
    Create and return a loss landscape object for the specified model type.
    
    Args:
        search_configs: Model configuration dictionary
        model_weight_name: Path to the saved model weights
        model_type: Type of model ('rdl' for GNN, 'dfs' for FT-Transformer)
        dataset: MultiTabularDataset object (required for GNN models)
        device: Device to run on
        task_type: Task type string (required for FT-Transformer)
        num_classes: Number of classes (required for FT-Transformer)
        train_loader: Training data loader (required for FT-Transformer)
        val_loader: Validation data loader (required for FT-Transformer)
        test_loader: Test data loader (required for FT-Transformer)
        train_dataset: Training dataset object (required for FT-Transformer model recreation)
        scaler_path: Path to scaler (optional for FT-Transformer regression)
        clamp_min: Minimum clamp value (optional for FT-Transformer regression)
        clamp_max: Maximum clamp value (optional for FT-Transformer regression)
    
    Returns:
        Initialized loss landscape object
    """
    if model_type == 'rdl':
        # For GNN models, we need to load the model and use the dataset
        if dataset is None:
            raise ValueError("dataset is required for RDL (GNN) models")
        
        # Load the trained model
        model = create_model(search_configs, dataset)
        try:
            model.load_state_dict(torch.load(model_weight_name, map_location=device))
        except RuntimeError as exc:
            logger.warning(f"Failed to load RDL weights for landscape metrics: {exc}")
            return None
        model = model.to(device)
        model.eval()
        
        landscape = GNNLossLandscape(
            model=model,
            dataset=dataset,
            device=device,
            model_config=search_configs
        )

        loss_grid, alphas, betas, center_metrics = landscape.plot(
            grid_points=5,
            figure_name=f"rdl_loss_landscape_{dataset.database.name}_{task_name}_{plot_name}.png",
            stage="val",
            use_amp=False
        )

        return center_metrics
        
    elif model_type == 'dfs':
        # For FT-Transformer models, we need the data loaders and task info
        if any(x is None for x in [train_loader, val_loader, test_loader, task_type, num_classes, train_dataset]):
            raise ValueError("train_loader, val_loader, test_loader, task_type, num_classes, and train_dataset are required for DFS (FT-Transformer) models")
        
        # Load the trained FT-Transformer model
        from dbinfer.solutions.rel_tabnn_solutions import create_ft_transformer_model
        search_configs['model_params']['channels'] = search_configs['model_params']['tab_nn_config']['hidden_size']
        search_configs['model_params']['num_layers'] = search_configs['model_params']['tab_nn_config']['num_layers']
        model = create_ft_transformer_model(train_dataset, search_configs['model_params'], task_type, num_classes)
        try:
            model.load_state_dict(torch.load(model_weight_name, map_location=device))
        except RuntimeError as exc:
            logger.warning(f"Failed to load DFS weights for landscape metrics: {exc}")
            return None
        model = model.to(device)
        model.eval()

        search_configs['loss_fn'] = {
            'binary_classification': 'bce',
            'multiclass_classification': 'ce',
            'regression': 'mae'
        }[task_type]
        
        landscape = FTTLossLandscape(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            task_type=task_type,
            num_classes=num_classes,
            model_config=search_configs,
            scaler_path=scaler_path,
            clamp_min=clamp_min,
            clamp_max=clamp_max
        )

        loss_grid, alphas, betas, center_metrics = landscape.plot(
            grid_points=20,
            figure_name=f"dfs_loss_landscape_{dataset.database.name if dataset is not None else 'unknown'}_{task_name}_{plot_name}.png",
            stage="val"
        )

        return center_metrics
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be 'rdl' or 'dfs'")
    

def load_torch_frame_loader(mtaskdataset, model_config, text_to_pca, text_to_pca_dim):
    autog_in_name = ""
    model_config['model_params']['text_to_pca'] = text_to_pca
    model_config['model_params']['text_to_pca_dim'] = text_to_pca_dim 
    scaler_path_name = f"scaler_{mtaskdataset.database.name}_{mtaskdataset.tasks[0].name}_{model_config['model_params']['dfs_layer']}"
    rdb_dataset = generate_dfs_data_pre_post("datasets/dfs", mtaskdataset, max_depth=model_config['model_params']['dfs_layer'], 
                                                 load_scalers_from=scaler_path_name, lte = mtaskdataset.tasks[0].auto, text_to_pca=text_to_pca, text_to_pca_dim=text_to_pca_dim)
    task_meta = mtaskdataset.get_task_meta(0, 1)
    clamp_min = task_meta.get('clamp_min', None)
    clamp_max = task_meta.get('clamp_max', None)

    scaler_path = rdb_dataset.scaler_path
    val_sample_batch = model_config.get('val_sample_size', 2000)
    max_train_samples = model_config['model_params'].get('auto_config', {}).get('max_train_samples')
    if isinstance(max_train_samples, str):
        max_train_samples = None if max_train_samples.lower() in {'none', ''} else max_train_samples
    sample_size = None
    if max_train_samples is not None:
        try:
            sample_size = int(max_train_samples)
        except (TypeError, ValueError):
            sample_size = None
    task = rdb_dataset.tasks[0]  # Assuming single task for now
    task_type, num_classes = check_task_type(task)
    is_classification = task_type == 'classification'

    if (task_type == 'classification' or task_type == 'binary_classification') and num_classes > 2:
        task_type = 'multiclass_classification'
    elif task_type == 'binary_classification' and num_classes == 2:
        task_type = 'binary_classification'
    elif task_type == 'classification' and num_classes == 2:
        task_type = 'binary_classification'
    
    remove_add_feature = True
    remove_cat_temporal_information = True

    # Convert datasets to torch_frame format
    train_ds, val_ds, test_ds = convert_all_splits(
        dbb_dataset=rdb_dataset,
        task_type=task_type,
        task_name=mtaskdataset.tasks[0].name,
        transform="none",
        cache_dir="./datasets",
        remove_add_feature=remove_add_feature,
        remove_cat_temporal_information=remove_cat_temporal_information, 
        sample_size = sample_size,
        scaler_path=scaler_path,
        val_sample_batch=val_sample_batch
    )    
    # Create data loaders
    train_loader = DataLoader(train_ds.tensor_frame, batch_size=model_config['model_params']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds.tensor_frame, batch_size=model_config['model_params'].get('eval_batch_size', model_config['model_params']['batch_size'] * 2))
    test_loader = DataLoader(test_ds.tensor_frame, batch_size=model_config['model_params'].get('eval_batch_size', model_config['model_params']['batch_size'] * 2))

    return {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader, 'task_type': task_type, 'num_classes': num_classes, 'model_config': model_config, 'scaler_path': scaler_path, 'clamp_min': clamp_min, 'clamp_max': clamp_max, 'train_dataset': train_ds}


def main():
    basic_config = load_yaml('./configs/default/task.yaml')
    data_config = load_yaml('./configs/pyg/hpo/all-nodes-test.yaml')
    result_db = create_swap_db(
        backend=basic_config.get("db_backend", "sqlite"),
        path_to_db=basic_config["db_location"],
        collection_name="loss_metrics",
        db_name=basic_config["db_name"],
    )
    model_performance_bank = load_model_performance_bank()
    databases = prepare_transductive_rdb_datasets_save_memory(data_config, basic_config)
    for database in databases:
        for task_id, task in enumerate(database.tasks):
            data_config = {
                "dbs": [
                    {
                        "db_name": database.database.name,
                        "task_name": [task.name],
                        "path": ""
                    }
                ]
            }
            full_mtaskdataset = prepare_transductive_rdb_datasets(data_config, basic_config)[0]
            top_validation_rdl_config = load_top_models(model_performance_bank, split = 'val', task_name = task.name, model_type = 'rdl')
            dfs_bank = model_performance_bank[(model_performance_bank['model_type'] == 'dfs') & (model_performance_bank['torch_frame_model_cls'] == 'ft_transformer') & (model_performance_bank['dfs_layer'] <= 2)]
            top_validation_dfs_config = load_top_models(dfs_bank, split = 'val', task_name = task.name, model_type = 'dfs')
            top_validation_rdl_args = wrap_searched_configurations_to_rdl_configs(top_validation_rdl_config)
            top_validation_dfs_args = wrap_searched_configurations_to_dfs_configs(top_validation_dfs_config)
            top_test_rdl_config = load_top_models(model_performance_bank, split = 'test', task_name = task.name, model_type = 'rdl')
            top_test_dfs_config = load_top_models(dfs_bank, split = 'test', task_name = task.name, model_type = 'dfs')
            top_test_rdl_args = wrap_searched_configurations_to_rdl_configs(top_test_rdl_config)
            top_test_dfs_args = wrap_searched_configurations_to_dfs_configs(top_test_dfs_config)

            ## rdl val
            if result_db.search_loss_metrics(task_name=task.name, type_f='rdl_val_selected'):
                top_validation_rdl_center_metrics = result_db.search_loss_metrics(task_name=task.name, type_f='rdl_val_selected')
            else:
                if not os.path.exists(f"modelcache/model_weight_{database.database.name}_{task.name}_rdl_val_selected.pt"):
                    res = execution(top_validation_rdl_args, basic_config, data_config, 
                    True, "", "", result_db, device = 'cuda', noexit = True, text_to_pca = True, text_to_pca_dim = 3)
                    top_validation_rdl_model_weight_name = res['model_weight_name']
                    ## rename model_weight_name 
                    os.rename(top_validation_rdl_model_weight_name, f"modelcache/model_weight_{database.database.name}_{task.name}_rdl_val_selected.pt")
                    top_validation_rdl_model_weight_name = f"modelcache/model_weight_{database.database.name}_{task.name}_rdl_val_selected.pt"
                else:
                    top_validation_rdl_model_weight_name = f"modelcache/model_weight_{database.database.name}_{task.name}_rdl_val_selected.pt"
                general_config = deepcopy(basic_config)
                general_config['model_params'] = top_validation_rdl_args
                top_validation_rdl_center_metrics = plot_loss_landscape_and_calculate_metrics(general_config, top_validation_rdl_model_weight_name, model_type='rdl', dataset=database, device='cuda', task_name=task.name, plot_name='val_selected')
                top_validation_rdl_center_metrics['task_name'] = task.name
                top_validation_rdl_center_metrics['type'] = 'rdl_val_selected'
                result_db.save_loss_metrics(top_validation_rdl_center_metrics)


            ## dfs val
            if result_db.search_loss_metrics(task_name=task.name, type_f='dfs_val_selected'):
                top_validation_dfs_center_metrics = result_db.search_loss_metrics(task_name=task.name, type_f='dfs_val_selected')
            else:
                if not os.path.exists(f"modelcache/model_weight_{database.database.name}_{task.name}_dfs_val_selected.pt"):
                    res = execution(top_validation_dfs_args, basic_config, data_config, 
                True, "", "", result_db, device = 'cuda', noexit = True, text_to_pca = True, text_to_pca_dim = 3)
                    top_validation_dfs_model_weight_name = res['model_weight_name']
                    os.rename(top_validation_dfs_model_weight_name, f"modelcache/model_weight_{database.database.name}_{task.name}_dfs_val_selected.pt")
                    top_validation_dfs_model_weight_name = f"modelcache/model_weight_{database.database.name}_{task.name}_dfs_val_selected.pt"
                else:
                    top_validation_dfs_model_weight_name = f"modelcache/model_weight_{database.database.name}_{task.name}_dfs_val_selected.pt"
                general_config = deepcopy(basic_config)
                general_config['model_params'] = top_validation_dfs_args
                torch_frame_meta_info = load_torch_frame_loader(full_mtaskdataset, general_config, True, 3)
                top_validation_dfs_center_metrics = plot_loss_landscape_and_calculate_metrics(general_config, top_validation_dfs_model_weight_name, model_type='dfs', dataset=database, device='cuda', task_type=torch_frame_meta_info['task_type'], num_classes=torch_frame_meta_info['num_classes'], task_name=task.name, train_loader=torch_frame_meta_info['train_loader'], val_loader=torch_frame_meta_info['val_loader'], test_loader=torch_frame_meta_info['test_loader'], train_dataset=torch_frame_meta_info['train_dataset'], scaler_path=torch_frame_meta_info['scaler_path'], clamp_min=torch_frame_meta_info['clamp_min'], clamp_max=torch_frame_meta_info['clamp_max'], plot_name='val_selected_dfs')
                top_validation_dfs_center_metrics['task_name'] = task.name
                top_validation_dfs_center_metrics['type'] = 'dfs_val_selected'
                result_db.save_loss_metrics(top_validation_dfs_center_metrics)


            ## rdl_test
            if result_db.search_loss_metrics(task_name=task.name, type_f='rdl_test_selected'):
                top_test_rdl_center_metrics = result_db.search_loss_metrics(task_name=task.name, type_f='rdl_test_selected')
            else:
                if not os.path.exists(f"modelcache/model_weight_{database.database.name}_{task.name}_rdl_test_selected.pt"):
                    res = execution(top_test_rdl_args, basic_config, data_config, 
                True, "", "", result_db, device = 'cuda', noexit = True, text_to_pca = True, text_to_pca_dim = 3)
                    top_test_rdl_model_weight_name = res['model_weight_name']
                    os.rename(top_test_rdl_model_weight_name, f"modelcache/model_weight_{database.database.name}_{task.name}_rdl_test_selected.pt")
                    top_test_rdl_model_weight_name = f"modelcache/model_weight_{database.database.name}_{task.name}_rdl_test_selected.pt"
                else:
                    top_test_rdl_model_weight_name = f"modelcache/model_weight_{database.database.name}_{task.name}_rdl_test_selected.pt"
                general_config = deepcopy(basic_config)
                general_config['model_params'] = top_test_rdl_args
                top_test_rdl_center_metrics = plot_loss_landscape_and_calculate_metrics(general_config, top_test_rdl_model_weight_name, model_type='rdl', dataset=database, device='cuda', task_name=task.name, plot_name='test_selected')
                top_test_rdl_center_metrics['task_name'] = task.name
                top_test_rdl_center_metrics['type'] = 'rdl_test_selected'
                result_db.save_loss_metrics(top_test_rdl_center_metrics)
            
            ## dfs_test
            if result_db.search_loss_metrics(task_name=task.name, type_f='dfs_test_selected'):
                top_test_dfs_center_metrics = result_db.search_loss_metrics(task_name=task.name, type_f='dfs_test_selected')
            else:
                if not os.path.exists(f"modelcache/model_weight_{database.database.name}_{task.name}_dfs_test_selected.pt"):
                    res = execution(top_test_dfs_args, basic_config, data_config, 
                True, "", "", result_db, device = 'cuda', noexit = True, text_to_pca = True, text_to_pca_dim = 3)
                    top_test_dfs_model_weight_name = res['model_weight_name']
                    os.rename(top_test_dfs_model_weight_name, f"modelcache/model_weight_{database.database.name}_{task.name}_dfs_test_selected.pt")
                    top_test_dfs_model_weight_name = f"modelcache/model_weight_{database.database.name}_{task.name}_dfs_test_selected.pt"
                else:
                    top_test_dfs_model_weight_name = f"modelcache/model_weight_{database.database.name}_{task.name}_dfs_test_selected.pt"
                general_config = deepcopy(basic_config)
                general_config['model_params'] = top_test_dfs_args
                torch_frame_meta_info = load_torch_frame_loader(full_mtaskdataset, general_config, True, 3)
                top_test_dfs_center_metrics = plot_loss_landscape_and_calculate_metrics(general_config, top_test_dfs_model_weight_name, model_type='dfs', dataset=database, device='cuda', task_type=torch_frame_meta_info['task_type'], num_classes=torch_frame_meta_info['num_classes'], task_name=task.name, train_loader=torch_frame_meta_info['train_loader'], val_loader=torch_frame_meta_info['val_loader'], test_loader=torch_frame_meta_info['test_loader'], train_dataset=torch_frame_meta_info['train_dataset'], scaler_path=torch_frame_meta_info['scaler_path'], clamp_min=torch_frame_meta_info['clamp_min'], clamp_max=torch_frame_meta_info['clamp_max'], plot_name='test_selected_dfs')
                top_test_dfs_center_metrics['task_name'] = task.name
                top_test_dfs_center_metrics['type'] = 'dfs_test_selected'
                result_db.save_loss_metrics(top_test_dfs_center_metrics)

            

if __name__ == "__main__":
    import wandb
    from utils.type import seed_everything
    wandb.init(mode="disabled")
    seed_everything(42)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    main()
