"""
This module contains utility functions for data preprocessing and feature engineering.

It includes functions for hashing objects, creating configuration dictionaries, and engineering features for a DataFrame. The feature engineering function uses the `FeatureEngineer` class from the `src.data_preprocess.feature_engineering` module.

Functions:
    get_hash(obj) -> str: 
        Returns the hash value of an object as a string.
    create_config_dict(model_name, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate, raw_data_path, interim_data_path, processed_data_path, logging_level, logging_format) -> dict: 
        Creates a configuration dictionary with the provided parameters.

Example:
    To use the feature engineering function, you can pass a pandas DataFrame:
    
    >>> df = pd.DataFrame(data)
    >>> engineered_df = engineer_features(df)

Note:
    The module expects a specific structure for the input dataframe as required by the FeatureEngineer class for the standardization of features.
"""

import hashlib
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preprocess.feature_engineering import FeatureEngineer
from src.data_preprocess.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


def get_hash(obj):
    """
    Returns the hash value of an object as a string.

    Parameters
    ----------
    obj : object
        The object to hash.

    Returns
    -------
    str
        The hash value of the object as a string.
    """
    return hashlib.sha256(str(obj).encode()).hexdigest()


def create_config_dict(model_name, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate, raw_data_path=None, interim_data_path=None, processed_data_path=None, logging_level='INFO', logging_format='%(asctime)s - %(levelname)s - %(message)s'):
    config = {
        'model_name': model_name,
        'model': {
            'rnn': {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'output_size': output_size,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        },
        'logging': {
            'level': logging_level,
            'format': logging_format
        }
    }
    config['data'] = {}
    config['data'] = {
        'raw_data_path': raw_data_path,
        'interim_data_path': interim_data_path,
        'processed_data_path': processed_data_path
    }
    return config
