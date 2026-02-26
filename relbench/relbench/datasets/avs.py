import os
import numpy as np
import pandas as pd
import polars as pl
import pooch
import subprocess
import sys
import zipfile
from relbench.base import Database, Dataset, Table
from relbench.utils import unzip_processor
import os
from relbench.base import TaskAdaptedDataset
from relbench.utils import check_raw_path_existence, train_val_test_split_by_temporal, convert_multiple_df_columns_to_list_embedding

def download_avs_dataset(dataset_path: str = "cache_data"):
    """
    Download AVS dataset using 4dbinfer
    
    Args:
        dataset_path: Base path where the dataset should be stored
    """
    
    # Create the raw data directory
    raw_path = os.path.join(dataset_path)
    os.makedirs(raw_path, exist_ok=True)
    
    print(f"Downloading AVS dataset to {raw_path}")
    
    try:
        cmd = ["pixi", "run", "python3", "-m", "dbinfer.main", "avs"]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, env={**os.environ, "DBB_DATASET_HOME": raw_path}, capture_output=True, text=True, check=True)
        print("Download completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

class AVSDataset(TaskAdaptedDataset):
    val_timestamp = pd.Timestamp('2013-04-20 00:00:00')
    test_timestamp = pd.Timestamp("2013-04-25 00:00:00")

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        expected_path = os.path.join(os.environ.get("XDG_CACHE_HOME", "cache_data"), "avs")
        if not os.path.exists(os.path.join(expected_path, "avs", "metadata.yaml")):
            # For Seznam, we expect the data to be downloaded via 4dbinfer
            # The path should be set to the raw data directory
            path = os.environ.get("XDG_CACHE_HOME", "cache_data")
            download_avs_dataset(path)
        else:
            path = check_raw_path_existence()

        downloaded_path = os.path.join(os.environ.get("XDG_CACHE_HOME", "cache_data"), "avs")
        transaction_df = pd.read_parquet(os.path.join(downloaded_path, "data", "transactions.pqt"))
        offers_df = pd.read_parquet(os.path.join(downloaded_path, "data", "offers.pqt"))
        history_df = pd.read_parquet(os.path.join(downloaded_path, "data", "history.pqt"))
        offers_unique_pqt = offers_df['category'].unique()
        transaction_df = transaction_df[transaction_df['category'].isin(offers_unique_pqt)]

        # Create dimension tables from transaction data
        tables = {}

        # Create Company dimension table
        company_df = pd.concat([transaction_df[['company']], offers_df[['company']]]).drop_duplicates().reset_index(drop=True)
        company_df['id'] = company_df.index.astype(str)
        company_df = company_df[['id', 'company']].rename(columns={'company': 'name'})

        # Create Brand dimension table  
        brand_df = pd.concat([transaction_df[['brand']], offers_df[['brand']]]).drop_duplicates().reset_index(drop=True)
        brand_df['id'] = brand_df.index.astype(str)
        brand_df = brand_df[['id', 'brand']].rename(columns={'brand': 'name'})

        # Create Category dimension table
        category_df = pd.concat([transaction_df[['category']], offers_df[['category']]]).drop_duplicates().reset_index(drop=True)
        category_df['id'] = category_df.index.astype(str)
        category_df = category_df[['id', 'category']].rename(columns={'category': 'name'})

        # Create Chain dimension table from unique chain values
        chain_df = pd.concat([transaction_df[['chain']], history_df[['chain']]]).drop_duplicates().reset_index(drop=True)
        chain_df['id'] = chain_df.index.astype(str)
        chain_df = chain_df[['id', 'chain']].rename(columns={'chain': 'name'})

        # Create Customer dimension table from unique original id values (the duplicate ids)
        customer_ids_from_transaction = transaction_df[['id']].rename(columns={'id': 'customer_id'})
        customer_ids_from_history = history_df[['id']].rename(columns={'id': 'customer_id'})
        customer_df = pd.concat([customer_ids_from_transaction, customer_ids_from_history]).drop_duplicates().reset_index(drop=True)
        customer_df['id'] = customer_df.index.astype(str)  # New primary key for customer table

        # Create lookup dictionaries for foreign key mapping
        company_lookup = dict(zip(company_df['name'], company_df['id']))
        brand_lookup = dict(zip(brand_df['name'], brand_df['id']))
        category_lookup = dict(zip(category_df['name'], category_df['id']))
        chain_lookup = dict(zip(chain_df['name'], chain_df['id']))
        customer_lookup = dict(zip(customer_df['customer_id'], customer_df['id']))  # Map original ids to customer table ids

        # Preserve original id columns before mapping
        #transaction_df['original_id'] = transaction_df['id']
        #history_df['original_id'] = history_df['id']
        
        # Create new primary key columns for transaction and history
        transaction_df['transaction_id'] = range(len(transaction_df))
        history_df['history_id'] = range(len(history_df))

        # Update transaction table with foreign keys
        transaction_df['company'] = transaction_df['company'].map(company_lookup)
        transaction_df['brand'] = transaction_df['brand'].map(brand_lookup)
        transaction_df['category'] = transaction_df['category'].map(category_lookup)
        transaction_df['chain'] = transaction_df['chain'].map(chain_lookup)
        transaction_df['customer'] = transaction_df['id'].map(customer_lookup)  # Map original id to customer table

        # Update offers table with foreign keys
        offers_df['company'] = offers_df['company'].map(company_lookup)
        offers_df['brand'] = offers_df['brand'].map(brand_lookup)
        offers_df['category'] = offers_df['category'].map(category_lookup)

        # Update history table with foreign key
        history_df['id'] = history_df['id'].map(customer_lookup)  # Map original id to customer table
        history_df['chain'] = history_df['chain'].map(chain_lookup)
        
        # Convert time columns to datetime format for temporal queries
        if 'date' in transaction_df.columns:
            transaction_df['date'] = pd.to_datetime(transaction_df['date'], errors='coerce')
            print(f"Transaction date column converted to datetime. Range: {transaction_df['date'].min()} to {transaction_df['date'].max()}")
        
        if 'offerdate' in history_df.columns:
            history_df['offerdate'] = pd.to_datetime(history_df['offerdate'], errors='coerce')
            print(f"History offerdate column converted to datetime. Range: {history_df['offerdate'].min()} to {history_df['offerdate'].max()}")
        
        # Check for any null datetime values
        if 'date' in transaction_df.columns and transaction_df['date'].isna().any():
            print(f"Warning: {transaction_df['date'].isna().sum()} null values in transaction date column")
        
        if 'offerdate' in history_df.columns and history_df['offerdate'].isna().any():
            print(f"Warning: {history_df['offerdate'].isna().sum()} null values in history offerdate column")

        # Create dimension tables
        tables["company"] = Table(
            df=company_df,
            fkey_col_to_pkey_table={},
            pkey_col="id",
            time_col=None,
        )

        tables["brand"] = Table(
            df=brand_df,
            fkey_col_to_pkey_table={},
            pkey_col="id",
            time_col=None,
        )

        tables["category"] = Table(
            df=category_df,
            fkey_col_to_pkey_table={},
            pkey_col="id",
            time_col=None,
        )

        tables["chain"] = Table(
            df=chain_df,
            fkey_col_to_pkey_table={},
            pkey_col="id",
            time_col=None,
        )

        tables["customer"] = Table(
            df=customer_df,
            fkey_col_to_pkey_table={},
            pkey_col="id",
            time_col=None,
        )

        # Transaction table with foreign key relationships
        tables["transaction"] = Table(
            df=transaction_df,
            fkey_col_to_pkey_table={
                "company": "company",
                "brand": "brand", 
                "category": "category",
                "customer": "customer", 
                "chain": "chain",
            },
            pkey_col="transaction_id",
            time_col="date",
        )

        # Offer table with foreign key relationships  
        tables["offer"] = Table(
            df=offers_df,
            fkey_col_to_pkey_table={
                "company": "company",
                "brand": "brand",
                "category": "category",
            },
            pkey_col="offer",
            time_col=None,
        )

        # History table with foreign key relationships
        tables["history"] = Table(
            df=history_df,
            fkey_col_to_pkey_table={
                # "customer": "customer",
                "offer": "offer",
                "chain": "chain",
                "id": "customer"
            },
            pkey_col="history_id",
            time_col="offerdate",
        )
        return Database(tables)

