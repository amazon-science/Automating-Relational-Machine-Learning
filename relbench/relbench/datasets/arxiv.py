import pandas as pd

from relbench.base import Database, Dataset, Table


class ArxivDataset(Dataset):
    val_timestamp = pd.Timestamp("2022-01-01")
    test_timestamp = pd.Timestamp("2023-01-01")

    def make_db(self) -> Database:
        raise RuntimeError(
            "Raw data processing is not supported for rel-arxiv. "
            "Please use the preprocessed version by calling "
            "get_dataset('rel-arxiv', download=True) to download "
            "directly from the RelBench server."
        )
