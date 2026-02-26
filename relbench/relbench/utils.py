import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile
import pandas as pd
import pooch
import numpy as np
from typing import List
import os.path as osp
import gdown
import torch
from torch_sparse import SparseTensor

def decompress_gz_file(input_path: str, output_path: str):
    import gzip
    import shutil

    # Open the gz file in binary read mode
    with gzip.open(input_path, "rb") as f_in:
        # Open the output file in binary write mode
        with open(output_path, "wb") as f_out:
            # Copy the decompressed data from the gz file to the output file
            shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed file saved as: {output_path}")


def unzip_processor(fname: Union[str, Path], action: str, pooch: pooch.Pooch) -> Path:
    zip_path = Path(fname)
    unzip_path = zip_path.parent / zip_path.stem
    if action != "fetch":
        shutil.unpack_archive(zip_path, unzip_path)
    else:  # fetch
        try:  # sanity check if all files are fully extracted comparing size
            for f in ZipFile(zip_path).infolist():
                if not f.is_dir():
                    fsize = os.path.getsize(os.path.join(unzip_path, f.filename))
                    assert f.file_size == fsize
        except Exception:  # otherwise do full unpack
            shutil.unpack_archive(zip_path, unzip_path)

    return unzip_path


def clean_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    r"""Clean the time column of a pandas dataframe.
    Args:
        df (pd.DataFrame): The pandas dataframe to clean the timecolumn for.
        col (str): The time column name.

    Returns:
        (pd.DataFrame): The pandas dataframe with the cleaned time column.
    """
    df[col] = pd.to_datetime(df[col], errors="coerce")

    # Count the number of rows before removing invalid dates
    total_before = len(df)

    # Remove rows where timestamp is NaT (indicating parsing failure)
    df = df.dropna(subset=[col])

    # Count the number of rows after removing invalid dates
    total_after = len(df)

    # Calculate the percentage of rows removed
    percentage_removed = ((total_before - total_after) / total_before) * 100

    # Print the percentage of comments removed
    print(
        f"Percentage of rows removed due to invalid dates: "
        f"{percentage_removed:.2f}%"
    )
    return df


def train_val_test_split_by_temporal(df, time_col, train_ratio, val_ratio):
    """
    Split a dataframe temporally with timestamp indicated by ``time_col``
    according to train-validation-test ratio

    .. code::

       train_ratio : val_ratio : (1 - train_ratio - val_ratio)
    """
    df = df.sort_values(time_col)
    time = pd.to_datetime(df[time_col])
    train_time_cutoff, val_time_cutoff = np.quantile(
        np.unique(time),
        [train_ratio, train_ratio + val_ratio]
    )
    return train_time_cutoff, val_time_cutoff


def check_raw_path_existence():
    if os.environ.get("RAW_DATA_PATH") is None:
        return ""
    elif not os.path.exists(os.environ.get("RAW_DATA_PATH")):
        return ""
    else:
        return os.environ.get("RAW_DATA_PATH")
    

def convert_multiple_df_columns_to_list_embedding(df: pd.DataFrame, columns: list[str]) -> List[List]:
    """
    Convert multiple numerical columns of a DataFrame into a list of lists.
    
    Args:
        df: Input DataFrame
        columns: List of column names to convert
        
    Returns:
        List of lists where each inner list contains the values from the specified columns for each row
    """
    # Use vectorized operations instead of iterrows for much better performance
    return df[columns].values.tolist()




def download_dataset_from_drive(url, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    file_id = url.split('/')[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # output_path = os.path.join(output_directory, 'dataset.zip')
    try:
        output_path = gdown.download(download_url, output_directory + osp.sep, quiet=False)
    except:
        print('It looks like Gdown encounters errors, or Google drive exhibits download '
              'number limits during downloading. However, You still can download the file '
              'from a web browser by using this link:\n\n'
              f'{url}\n\n'
              'Then unzip this file (only if it is a Zip file), and manually put all the content to '
              f'{output_directory}.')
        exit(0)

    
    # Unzip the dataset if necessary
    if output_path.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_directory)
        os.remove(output_path)  # Remove the zip file after extraction


def download_dataset_from_hub(repo, file, output_directory):
    from huggingface_hub import hf_hub_download
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    output_path = hf_hub_download(repo_id=repo, filename=file, repo_type="dataset")
    
    # Unzip the dataset if necessary
    if output_path.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_directory)
    else:
        return output_path
    

def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

def get_sparse_tensor(edge_index, num_nodes=None, num_src_nodes=None, num_dst_nodes=None, return_e_id=False):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if return_e_id:
            value = torch.arange(adj_t.nnz())
            adj_t = adj_t.set_value(value, layout='coo')
        return adj_t


    if (num_nodes is None) and (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes = int(edge_index.max()) + 1
        num_dst_nodes = num_src_nodes
    elif (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes, num_dst_nodes = num_nodes, num_nodes


    value = torch.arange(edge_index.size(1)) if return_e_id else None
    return SparseTensor(row=edge_index[0], col=edge_index[1],
                        value=value,
                        sparse_sizes=(num_src_nodes, num_dst_nodes)).t()