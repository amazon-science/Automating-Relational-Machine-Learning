import torch
from typing import List

def get_auto_model(model_type: str):
    if model_type.lower() == "rdl":
        from .nn.autognn import AutoFullPredictor
        return AutoFullPredictor
    else:
        raise ValueError(f"Model {model_type} not found")


def freeze_everything_but(model: torch.nn.Module, target_names: List[str]):
    """
    Freezes (requires_grad=False) all parameters of the sub‐module
    whose full module name equals target_name.
    """
    for name, module in model.named_modules():
        if name not in target_names:
            for p in module.parameters():
                p.requires_grad = False


def unfreeze_everything(model: torch.nn.Module):
    for name, module in model.named_modules():
        for p in module.parameters():
            p.requires_grad = True
        # print(f"Unfroze module '{name}'")