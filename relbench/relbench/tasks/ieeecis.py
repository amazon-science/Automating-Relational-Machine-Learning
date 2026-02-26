"""
IEEE-CIS Fraud Detection Tasks

This module contains tasks for the IEEE-CIS fraud detection dataset.
The main task is to predict whether a transaction is fraudulent or not.
"""

from relbench.base import AutoCompleteTask, TaskType

# Fraud detection task - predict whether a transaction is fraudulent
class FraudDetectionTask(AutoCompleteTask):
    r"""Predict whether a transaction is fraudulent (isFraud = 1) or legitimate (isFraud = 0).
    
    This is a binary classification task where the model needs to predict the fraud status
    of transactions based on transaction features and identity information.
    """
    
    task_type = TaskType.BINARY_CLASSIFICATION
    entity_table = "Transaction"
    target_col = "isFraud"
    remove_columns = [
    ]
