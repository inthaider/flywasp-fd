import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def handle_infinity_and_na_numpy(*arrays):
    """
    Replaces infinite and NaN values in multiple NumPy arrays with forward/backward filled values.
    Returns new arrays with the replaced values.
    """
    handled_arrays = []
    
    try:
        logging.info("Handling infinite and NaN values for multiple NumPy arrays...")

        for arr in arrays:
            # Make a copy of the original array
            new_arr = np.copy(arr)

            # Replace infinite values with NaN
            new_arr[np.isinf(new_arr)] = np.nan

            # Forward fill NaN values along each time series (assuming time series are along axis 1)
            for i in range(new_arr.shape[0]):
                mask = np.isnan(new_arr[i])
                new_arr[i][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), new_arr[i][~mask])

            # Backward fill any remaining NaN values
            for i in range(new_arr.shape[0]):
                mask = np.isnan(new_arr[i])
                if np.any(mask):
                    new_arr[i][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), new_arr[i][~mask], left=new_arr[i][~mask][-1], right=new_arr[i][~mask][-1])

            handled_arrays.append(new_arr)

        return tuple(handled_arrays)

    except Exception as e:
        logging.error(f"Error handling infinite and NaN values in NumPy arrays: {e}")
        raise


def load_train_test_data(data_dir='data/processed/rnn_input/'):
    """
    Loads the train and test datasets for the RNN model from 4 .pkl files.

    Parameters
    ----------
    data_dir : str, optional
        The path to the directory containing the .pkl files, by default 'data/processed/rnn_input/'

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The train-test splits as (X_train, Y_train, X_test, Y_test).
    """
    try:
        # Load the train and test datasets from .pkl files
        X_train_file = Path(data_dir) / "X_train.pkl"
        Y_train_file = Path(data_dir) / "Y_train.pkl"
        X_test_file = Path(data_dir) / "X_test.pkl"
        Y_test_file = Path(data_dir) / "Y_test.pkl"

        X_train = pickle.load(open(X_train_file, "rb"))
        Y_train = pickle.load(open(Y_train_file, "rb"))
        X_test = pickle.load(open(X_test_file, "rb"))
        Y_test = pickle.load(open(Y_test_file, "rb"))

        return X_train, Y_train, X_test, Y_test
    except Exception as e:
        logging.error(f"Error loading train and test datasets: {e}")
        raise


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


def create_sequences(data, sequence_length=3):
    n = len(data) - sequence_length
    x = np.zeros((n, sequence_length, data.shape[1]))
    y = np.zeros(n)
    for i in range(n):
        x[i] = data[i:i + sequence_length]
        # Assuming the target column is the last one
        y[i] = data[i + sequence_length, -1]
    return x, y


def prepare_train_test_sequences(df, sequence_length=3, split_ratio=2/3):
    logging.info("Preparing training and testing sequences...")

    # Initial checks and setup
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer.")
    if not isinstance(split_ratio, float):
        raise TypeError("split_ratio must be a float.")
    if not (0 < split_ratio < 1):
        raise ValueError("split_ratio must be between 0 and 1.")

    # Convert the DataFrame to a NumPy array for faster slicing
    df_values = df.values

    # Create empty lists to collect the sequences
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    # Calculate sizes in advance for pre-allocation
    unique_files = df['file'].unique()
    total_sequences = sum(
        len(df[df['file'] == file]) - sequence_length for file in unique_files)
    train_size = int(total_sequences * split_ratio)
    test_size = total_sequences - train_size

    # Pre-allocate numpy arrays
    input_dim = df.shape[1] - 2  # -2 because we're dropping 'Frame' and 'file'
    X_train = np.zeros((train_size, sequence_length, input_dim))
    Y_train = np.zeros(train_size)
    X_test = np.zeros((test_size, sequence_length, input_dim))
    Y_test = np.zeros(test_size)

    train_idx, test_idx = 0, 0

    i = 0
    for file in unique_files:
        logging.info(f"===================")
        logging.info(f"Fly-wasp pair # {i}")
        logging.info(f"Processing file {file}...")
        file_data = df[df['file'] == file].drop(
            ['Frame', 'file'], axis=1).values

        # Create sequences for each file
        x, y = create_sequences(file_data, sequence_length=sequence_length)

        # Calculate the split index for this file
        n = len(x)
        file_train_size = int(n * split_ratio)

        # Add the sequences to the pre-allocated arrays
        X_train[train_idx:train_idx + file_train_size] = x[:file_train_size]
        Y_train[train_idx:train_idx + file_train_size] = y[:file_train_size]
        X_test[test_idx:test_idx + n - file_train_size] = x[file_train_size:]
        Y_test[test_idx:test_idx + n - file_train_size] = y[file_train_size:]

        # Update the indices for the next iteration
        train_idx += file_train_size
        test_idx += n - file_train_size

        i = i+1

    logging.info(
        f"Prepared {len(X_train)} training sequences and {len(X_test)} testing sequences.")
    return X_train, Y_train, X_test, Y_test


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
