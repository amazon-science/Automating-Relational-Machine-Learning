import os
import numpy as np
import pandas as pd
import pooch
import subprocess
import sys
import zipfile
from relbench.base import Database, Dataset, Table
from relbench.utils import unzip_processor
from relbench.base import TaskAdaptedDataset
from relbench.utils import check_raw_path_existence, train_val_test_split_by_temporal, convert_multiple_df_columns_to_list_embedding


def download_ieeecis_dataset(dataset_path: str = "cache_data"):
    """
    Download IEEE CIS Fraud Detection dataset from Kaggle.
    
    Args:
        dataset_path: Base path where the dataset should be stored
    """
    
    # Create the raw data directory
    raw_path = os.path.join(dataset_path, "ieeecis", "raw")
    os.makedirs(raw_path, exist_ok=True)
    
    print(f"Downloading IEEE CIS dataset to {raw_path}")
    
    try:
        # Download the dataset using kaggle CLI
        cmd = [
            "kaggle", "competitions", "download", 
            "-c", "ieee-fraud-detection", 
            "--path", raw_path
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Download completed successfully!")
        print(result.stdout)
        
        # Extract the zip file
        zip_file = os.path.join(raw_path, "ieee-fraud-detection.zip")
        if os.path.exists(zip_file):
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(raw_path)
            print("Extraction completed!")
            
            # Remove the zip file to save space
            os.remove(zip_file)
            print("Removed zip file to save space.")
        else:
            print(f"Warning: Expected zip file {zip_file} not found.")
            
        # List the downloaded files
        print("\nDownloaded files:")
        for file in os.listdir(raw_path):
            file_path = os.path.join(raw_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                print(f"  {file} ({size:.2f} MB)")
                
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print(f"Error output: {e.stderr}")
        print("\nMake sure you have:")
        print("1. Kaggle API credentials set up (kaggle.json in ~/.kaggle/)")
        print("2. Kaggle CLI installed (pip install kaggle)")
        print("3. Accepted the competition rules on Kaggle website")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

class IEEECISDataset(TaskAdaptedDataset):
    val_timestamp = pd.Timestamp('1970-04-12 05:23:18')
    test_timestamp = pd.Timestamp('1970-05-22 02:55:00')

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        expected_path = os.path.join(os.environ.get("XDG_CACHE_HOME", "cache_data"), "ieeecis")
        if not os.path.exists(os.path.join(expected_path, "raw", "train_transaction.csv")):
            # For IEEE CIS, we expect the data to be downloaded via Kaggle
            # The path should be set to the raw data directory
            path = os.environ.get("XDG_CACHE_HOME", "cache_data")
            download_ieeecis_dataset(path)
            path = os.path.join(path, "ieeecis", "raw")
        else:
            path = os.path.join(expected_path, "raw")

        # Read the transaction and identity data
        transaction_df = pd.read_csv(os.path.join(path, "train_transaction.csv"))
        identity_df = pd.read_csv(os.path.join(path, "train_identity.csv"))

        # Process transaction data
        transaction_df['index'] = transaction_df.index
        transaction_df['TransactionDT'] = pd.to_datetime(transaction_df['TransactionDT'], unit='s')
        transaction_df['TransactionAmt'] = np.log1p(transaction_df['TransactionAmt'])
        
        # Create tables
        tables = {}

        # Transaction table
        tables["transactions"] = Table(
            df=transaction_df,
            fkey_col_to_pkey_table={},
            pkey_col="TransactionID",
            time_col="TransactionDT",
        )

        # Identity table
        tables["identity"] = Table(
            df=identity_df,
            fkey_col_to_pkey_table={
                "TransactionID": "transactions",
            },
            pkey_col=None,
            time_col=None,
        )

        # tables["identity"].df["id_features"] = convert_multiple_df_columns_to_list_embedding(
        #     tables["identity"].df,
        #     [f"id_{i:02d}" for i in range(1, 12)]
        # )

        # tables["identity"].df.drop(
        #     [f"id_{i:02d}" for i in range(1, 12)],
        #     inplace=True,
        #     axis=1
        # )

        tables['identity'].df.rename(columns={"id_33": "resolution"}, inplace=True)

        device_df = tables["identity"].df[["DeviceType", "DeviceInfo", "resolution"]].drop_duplicates().reset_index(drop=True)
        device_df["DeviceID"] = np.arange(len(device_df))

        device_mapping = pd.merge(
            tables["identity"].df[["DeviceType", "DeviceInfo", "resolution"]],
            device_df,
            on=["DeviceType", "DeviceInfo", "resolution"],
            how="left"
        )
        tables["identity"].fkey_col_to_pkey_table["DeviceID"] = "device"
        tables["identity"].df["DeviceID"] = device_mapping["DeviceID"]

        tables["identity"].df.drop(
            ["DeviceType", "DeviceInfo", "resolution"],
            inplace=True,
            axis=1
        )

        tables["device"] = Table(
            df=device_df,
            fkey_col_to_pkey_table={},
            pkey_col="DeviceID",
            time_col=None,
        )



        email_domains_df = transaction_df[["P_emaildomain", "R_emaildomain"]].drop_duplicates().reset_index(drop=True)
        pkey = np.arange(len(email_domains_df))
        email_domains_df["DomainID"] = pkey

        # get a mapping from P, R pair to DomainID, and then create a foreign key 
        # from the transaction table to the email_domains table
        domain_mapping = pd.merge(
            transaction_df[["TransactionID", "P_emaildomain", "R_emaildomain"]],
            email_domains_df,
            on=["P_emaildomain", "R_emaildomain"],
            how="left"
        )
        tables["transactions"].fkey_col_to_pkey_table["DomainID"] = "email_domains"
        tables["transactions"].df["DomainID"] = domain_mapping["DomainID"]
        tables['transactions'].df.drop(
            ["P_emaildomain", "R_emaildomain"],
            inplace=True,
            axis=1
        )
        
        tables["email_domains"] = Table(
            df=email_domains_df,
            fkey_col_to_pkey_table={},
            pkey_col="DomainID",
            time_col=None,
        )
        

        # Match status table (combining all match features)
        match_columns = [f"M{i}" for i in range(1, 10)]
        match_data = transaction_df[match_columns].drop_duplicates().reset_index(drop=True)
        match_data = match_data.reset_index().rename(columns={"index": "MatchStatusID"})
        match_data["MatchStatusID"] = np.arange(len(match_data))

        # need a foreign key in the transaction table to the match_status table
        match_mapping = pd.merge(
            transaction_df[["TransactionID", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]],
            match_data,
            on=match_columns,
            how="left"
        )
        tables["transactions"].fkey_col_to_pkey_table["MatchStatusID"] = "match_status"
        tables["transactions"].df["MatchStatusID"] = match_mapping["MatchStatusID"]
        # Drop match columns from transaction table
        tables['transactions'].df.drop(
            match_columns,
            inplace=True,
            axis=1
        )
        
        tables["match_status"] = Table(
            df=match_data,
            fkey_col_to_pkey_table={},
            pkey_col="MatchStatusID",
            time_col=None,
        )

        return Database(tables)

