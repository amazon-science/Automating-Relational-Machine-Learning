import json
import pkgutil
from functools import lru_cache
from typing import List
import pooch
from relbench.base import Dataset
from relbench.datasets import avito, event, f1, fakef1, hm, stack, trial, ieeecis, arxiv
import os
from pathlib import Path

# Default cache directory: <project_root>/cache_data
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
DEFAULT_CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache_data")

from relbench.datasets import avs

dataset_registry = {}

hashes_str = pkgutil.get_data(__name__, "hashes.json")
hashes = json.loads(hashes_str)

test_timecutoffs_str = pkgutil.get_data(__name__, "test_timecutoffs.json")
test_timecutoffs = json.loads(test_timecutoffs_str)

DOWNLOAD_REGISTRY = pooch.create(
    path=os.environ.get("XDG_CACHE_HOME") or DEFAULT_CACHE_DIR,
    base_url="https://relbench.stanford.edu/download/",
    registry=hashes,
)


def register_dataset(
    name: str,
    cls: Dataset,
    *args,
    **kwargs,
) -> None:
    r"""Register an instantiation of a :class:`Dataset` subclass with the given name.

    Args:
        name: The name of the dataset.
        cls: The class of the dataset.
        args: The arguments to instantiate the dataset.
        kwargs: The keyword arguments to instantiate the dataset.

    The name is used to enable caching and downloading functionalities.
    `cache_dir` is added to kwargs by default. If you want to override it, you
    can pass `cache_dir` as a keyword argument in `kwargs`.
    """

    cache_dir = f"{os.environ.get('XDG_CACHE_HOME') or DEFAULT_CACHE_DIR}/{name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    dataset_registry[name] = (cls, args, kwargs)


def get_dataset_names() -> List[str]:
    r"""Return a list of names of the registered datasets."""
    return list(dataset_registry.keys())


def get_test_timecutoff(name: str) -> str:
    r"""Return the test timecutoff for a given dataset by name.
    
    Args:
        name: The name of the dataset.
        
    Returns:
        str: The test timecutoff timestamp as a string.
    """
    return test_timecutoffs.get(name)


def download_dataset(name: str) -> None:
    r"""Download dataset from RelBench server into its cache directory.

    The downloaded database will be automatically picked up by the dataset object, when
    `dataset.get_db()` is called.
    """

    DOWNLOAD_REGISTRY.fetch(
        f"{name}/db.zip",
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True,
    )


@lru_cache(maxsize=None)
def get_dataset(name: str, download=True) -> Dataset:
    r"""Return a dataset object by name.

    Args:
        name: The name of the dataset.
        download: If True, download the dataset from the RelBench server.

    Returns:
        Dataset: The dataset object.

    If `download` is True, the database comprising the dataset will be
    downloaded into the cache from the RelBench server. If you use
    `download=False` the first time, the database will be processed from the
    raw files of the original source.

    Once the database is cached, either because of download or processing from
    raw files, the cache will be used. `download=True` will verify that the
    cached database matches the RelBench version even in this case.
    """

    if download and name not in on_disks:
        download_dataset(name)
    cls, args, kwargs = dataset_registry[name]
    dataset = cls(*args, **kwargs)
    return dataset


register_dataset("rel-avito", avito.AvitoDataset)
register_dataset("rel-event", event.EventDataset)
register_dataset("rel-f1", f1.F1Dataset)
register_dataset("rel-fakef1", fakef1.F1Dataset)
register_dataset("rel-hm", hm.HMDataset)
register_dataset("rel-stack", stack.StackDataset)
register_dataset("rel-trial", trial.TrialDataset)
register_dataset("rel-ieeecis", ieeecis.IEEECISDataset)
register_dataset("rel-avs", avs.AVSDataset)
register_dataset("rel-arxiv", arxiv.ArxivDataset)
on_disks = ['rel-ieeecis', 'rel-avs', 'rel-fakef1']

