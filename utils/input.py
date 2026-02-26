from typing import Dict, Any
import os
import yaml
try:
    import wandb
except ImportError:
    wandb = None

def write_dict_to_yaml(data_dict: Dict[str, Any], file_path: str) -> None:
    """
    Writes a Python dictionary to a YAML file.

    Ensures the directory for the file path exists before writing.

    Args:
        data_dict: The dictionary containing the data to be written.
        file_path: The complete path (including filename) for the output YAML file.

    Raises:
        IOError: If there's an error writing to the file.
        yaml.YAMLError: If there's an error during YAML serialization.
        Exception: For any other unexpected errors.

    Returns:
        None
    """
    try:
        # Extract the directory path from the full file path
        directory = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        # The 'exist_ok=True' argument prevents an error if the directory already exists
        if directory: # Ensure directory is not empty (e.g., if file is in current dir)
             os.makedirs(directory, exist_ok=True)

        # Open the file in write mode ('w') with UTF-8 encoding
        # Using 'with' ensures the file is properly closed even if errors occur
        with open(file_path, 'w', encoding='utf-8') as yaml_file:
            # Dump the dictionary data into the YAML file
            # default_flow_style=False makes the output more readable (block style)
            # sort_keys=False preserves the original order of keys in the dictionary
            yaml.dump(data_dict, yaml_file, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"Successfully wrote dictionary to YAML file: {file_path}")

    except IOError as e:
        print(f"Error: Could not write to file {file_path}. Reason: {e}")
        raise  # Re-raise the exception if specific handling is needed upstream
    except yaml.YAMLError as e:
        print(f"Error: Failed to serialize dictionary to YAML. Reason: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def initialize_wandb(project_name, experiment_name, config=None, resume=False):
    """
    Initializes Weights & Biases (wandb) for experiment tracking.

    Args:
        project_name (str): The name of the wandb project.
        experiment_name (str): The name of the specific experiment run.
        config (dict, optional): A dictionary containing experiment configurations. Defaults to None.
        resume (bool, optional): Whether to resume a previous run. Defaults to False.

    Returns:
        wandb.Run: The wandb run object.
    """
    run = wandb.init(
        project=project_name,
        name=experiment_name,
        config=config,
        resume=resume,
    )
    return run


def merge_namespaces_with_opts(config, args):
    """
    Merges attributes from an argparse.Namespace into a dictionary,
    including handling the special 'opts' format and nested config changes.

    Args:
        config (dict): The original configuration dictionary.
        args (argparse.Namespace): The argparse Namespace to merge into config.

    Returns:
        dict: The updated configuration dictionary.
    """
    full = config.copy()  # Create a copy to avoid modifying the original

    # Handle 'opts' first to process flags without values
    if hasattr(args, 'opts') and args.opts:
        opts_dict = {}
        i = 0
        while i < len(args.opts):
            if i + 1 < len(args.opts) and not args.opts[i + 1].startswith('--'):
                opts_dict[args.opts[i].lstrip('--')] = args.opts[i + 1]
                i += 2
            else:
                opts_dict[args.opts[i].lstrip('--')] = True
                i += 1

        for opt_key, opt_value in opts_dict.items():
            if isinstance(opt_value, str) and opt_value.isdigit():
                opt_value = int(opt_value)

            # Handle nested keys like "optim_params.lr"
            if '.' in opt_key:
                parts = opt_key.split('.')
                temp_config = full
                for part in parts[:-1]:
                    if part in temp_config:
                        temp_config = temp_config[part]
                    else:
                        break # if any part of the path is not found, skip
                else:
                    temp_config[parts[-1]] = opt_value
            else:
                full[opt_key] = opt_value

    # Merge standard attributes, overriding with values from 'opts' if present
    for key, value in vars(args).items():
        if key != "opts" and value is not None: # only merge if value is not none.
            # Handle nested keys like "optim_params.lr"
            if '.' in key:
                parts = key.split('.')
                temp_config = full
                for part in parts[:-1]:
                    if part in temp_config:
                        temp_config = temp_config[part]
                    else:
                        break # if any part of the path is not found, skip
                else:
                    temp_config[parts[-1]] = value
            else:
                full[key] = value

    return full