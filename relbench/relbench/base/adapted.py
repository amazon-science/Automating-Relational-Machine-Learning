import time
from functools import lru_cache
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from relbench.base.database import Database
from relbench.base.dataset import Dataset
import os 
os.environ.setdefault("RAW_DATA_PATH", os.environ.get("XDG_CACHE_HOME", "cache_data"))

class TaskAdaptedDataset(Dataset):
    r"""
    Dataset with task-specific train/val/test temporal splits.
    """
    @lru_cache(maxsize=None)
    def get_db(self, upto_test_timestamp=True, test_timestamp = None, mask_cols = "") -> Database:
        r"""Return the database object.

        The returned database object is cached in memory.

        Args:
            upto_test_timestamp: If True, only return rows upto test_timestamp.

        Returns:
            Database: The database object.

        `upto_test_timestamp` is True by default to prevent test leakage.
        """

        db_path = f"{self.cache_dir}/db"
        if self.cache_dir and Path(db_path).exists() and any(Path(db_path).iterdir()):
            print(f"Loading Database object from {db_path}...")
            tic = time.time()
            db = Database.load(db_path)
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

        else:
            print("Making Database object from scratch...")
            print(
                "(You can also use `get_dataset(..., download=True)` "
                "for datasets prepared by the RelBench team.)"
            )
            tic = time.time()
            db = self.make_db()
            db.reindex_pkeys_and_fkeys()
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

            if self.cache_dir:
                print(f"Caching Database object to {db_path}...")
                tic = time.time()
                db.save(db_path)
                toc = time.time()
                print(f"Done in {toc - tic:.2f} seconds.")

        if upto_test_timestamp:
            if test_timestamp is None:
                db = db.upto(self.test_timestamp)
            else:
                db = db.upto(test_timestamp)

        self.validate_and_correct_db(db)

        if self.target_col:
            # Get the modified db with the target column removed
            db = self.get_modified_db(db)

        return db

    def make_db(self) -> Database:
        r"""Make the database object from scratch, i.e. using raw data sources.

        To be implemented by subclass.
        """
        raise NotImplementedError
