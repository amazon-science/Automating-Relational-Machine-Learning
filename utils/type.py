from pathlib import Path
import json 
from torch_frame import stype
from relbench.modeling.utils import get_stype_proposal  
import ast
import torch_frame
from relbench.datasets import on_disks
import random
import os
import numpy as np
import torch

def read_txt_dict(file_path):
    """Read dictionary from a text file."""
    with open(file_path, 'r') as file:
        content = file.read()
    return ast.literal_eval(content)

def get_stype_dict(dataset, cache_dir, dataset_name, dfs_mode = False):
    """
        Get the stype dict from the dataset
    """
    stypes_cache_path = Path(f"{cache_dir}/{dataset_name}/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db(), on_disk=dataset.name in on_disks, dfs_mode=dfs_mode)
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)
    return col_to_stype_dict


def convert_llm_output_to_stype_proposal(llm_output_path):
    type_dict = read_txt_dict(llm_output_path)
    translation_dict = {
        'category': torch_frame.categorical,
        'timestamp': torch_frame.timestamp,
        'text': torch_frame.text_embedded,
        'float': torch_frame.numerical,
        'multi_category': torch_frame.categorical,
        'datetime': torch_frame.timestamp,
    }
    infered_type_dict = {}
    for table_name, table in type_dict.items():
        infered_type_dict[table_name] = {}
        for col_name, col_info in table.items():
            inferred_type = col_info[0]
            translated_type = translation_dict[inferred_type]
            infered_type_dict[table_name][col_name] = translated_type
    return infered_type_dict


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False