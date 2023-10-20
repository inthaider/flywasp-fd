import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def handle_infinity_and_na_numpy(*arrays):
    """
    Replaces infinite and NaN values in multiple NumPy arrays of RNN input X (not Y!) with forward/backward filled values.
    This function modifies the RNN input X arrays in-place.

    If a feature in a given batch (i.e., a specific sample or observation) and time step is NaN,
    the function will look for the nearest non-NaN value in the same feature but different time step 
    within the same batch. If all time steps for that feature in that batch are NaN, the function 
    will look for the nearest non-NaN value in the same feature and time step but in a different batch.

    Parameters
    ----------
    *arrays : array_like
        One or more NumPy arrays to be processed.

    Raises
    ------
    Exception
        If an error occurs while handling infinite and NaN values.

    Notes
    -----
    - The function assumes that the first axis (axis=0) corresponds to different batches or samples, 
      the second axis (axis=1) corresponds to different time steps in each sample, and the third axis (axis=2) 
      corresponds to different features.
    - The function modifies the input arrays in-place. Make sure to keep copies if the original data 
      is needed later.
    - Logging is used for debug and info messages. Make sure to configure your logging level accordingly.
    """

    try:
        logging.info(
            "Starting to handle infinite and NaN values for multiple NumPy arrays...")

        for idx, arr in enumerate(arrays):
            logging.info(
                f"Starting to process array {idx + 1} of {len(arrays)}...")

            # Replace infinite values with NaN
            logging.info("Replacing infinite values with NaN...")
            arr[np.isinf(arr)] = np.nan
            logging.info("Infinite values replaced with NaN.")

            # Forward fill NaN values along each time series (axis=1)
            logging.info("Starting forward fill for NaN values...")
            mask = np.isnan(arr)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[2]):
                    logging.debug(
                        f"Forward filling NaN values in batch {i}, feature {j}...")
                    if np.all(mask[i, :, j]):
                        logging.warning(
                            f"All values are NaN in batch {i}, feature {j}. Looking into other batches for filling...")
                        for k in range(i, arr.shape[0]):
                            if not np.all(np.isnan(arr[k, :, j])):
                                arr[i, :, j] = arr[k, :, j]
                                logging.info(
                                    f"Forward filled using batch {k} for batch {i}, feature {j}.")
                                break
                    else:
                        valid_idx = np.flatnonzero(~mask[i, :, j])
                        arr[i, mask[i, :, j], j] = np.interp(np.flatnonzero(
                            mask[i, :, j]), valid_idx, arr[i, ~mask[i, :, j], j])
                        logging.debug(
                            f"Successfully forward filled NaN values in batch {i}, feature {j}.")

            logging.info("Forward filling completed.")

            # Backward fill any remaining NaN values
            logging.info("Starting backward fill for remaining NaN values...")
            mask = np.isnan(arr)
            for i in range(arr.shape[0] - 1, -1, -1):
                for j in range(arr.shape[2]):
                    logging.debug(
                        f"Backward filling NaN values in batch {i}, feature {j}...")
                    if np.all(mask[i, :, j]):
                        logging.warning(
                            f"All values are NaN in batch {i}, feature {j}. Looking into other batches for filling...")
                        for k in range(i, -1, -1):
                            if not np.all(np.isnan(arr[k, :, j])):
                                arr[i, :, j] = arr[k, :, j]
                                logging.info(
                                    f"Backward filled using batch {k} for batch {i}, feature {j}.")
                                break
                    else:
                        valid_idx = np.flatnonzero(~mask[i, :, j])
                        arr[i, mask[i, :, j], j] = np.interp(np.flatnonzero(
                            mask[i, :, j]), valid_idx, arr[i, ~mask[i, :, j], j], left=arr[i, ~mask[i, :, j]][-1], right=arr[i, ~mask[i, :, j]][-1])
                        logging.debug(
                            f"Successfully backward filled NaN values in batch {i}, feature {j}.")

            logging.info(f"Backward filling completed.")
            logging.info(f"Completed processing array {idx + 1}.")

    except Exception as e:
        logging.error(
            f"Error while handling infinite and NaN values in NumPy arrays: {e}")
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
