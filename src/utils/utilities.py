from datetime import datetime

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import logging

from src.data_preprocess.feature_engineering import FeatureEngineer
from src.data_preprocess.preprocessing import DataPreprocessor


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








############
# NEW FUNCTIONS
############

def engineer_features(df):
    """
    Performs feature engineering steps on the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns
    -------
    pandas.DataFrame
        The feature-engineered DataFrame.
    """
    feature_engineer = FeatureEngineer(df=df)
    feature_engineer.standardize_features(
        [
            "Fdis",
            "FdisF",
            "FdisL",
            "Wdis",
            "WdisF",
            "WdisL",
            "Fangle",
            "Wangle",
            "F2Wdis",
            "F2Wdis_rate",
            "F2Wangle",
            "W2Fangle",
            "ANTdis",
            "F2W_blob_dis",
            "bp_F_delta",
            "bp_W_delta",
            "ap_F_delta",
            "ap_W_delta",
            "ant_W_delta",
        ]
    )  # Standardize the selected features
    return feature_engineer.df
