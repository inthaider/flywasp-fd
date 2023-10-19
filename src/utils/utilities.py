import logging
import numpy as np
import pandas as pd
import hashlib


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
        y[i] = data[i + sequence_length, -1]  # Assuming the target column is the last one
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
    total_sequences = sum(len(df[df['file'] == file]) - sequence_length for file in unique_files)
    train_size = int(total_sequences * split_ratio)
    test_size = total_sequences - train_size

    # Pre-allocate numpy arrays
    input_dim = df.shape[1] - 2  # -2 because we're dropping 'Frame' and 'file'
    X_train = np.zeros((train_size, sequence_length, input_dim))
    Y_train = np.zeros(train_size)
    X_test = np.zeros((test_size, sequence_length, input_dim))
    Y_test = np.zeros(test_size)

    train_idx, test_idx = 0, 0

    i=0
    for file in unique_files:
        logging.info(f"===================")
        logging.info(f"Fly-wasp pair # {i}")
        logging.info(f"Processing file {file}...")
        file_data = df[df['file'] == file].drop(['Frame', 'file'], axis=1).values

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

    logging.info(f"Prepared {len(X_train)} training sequences and {len(X_test)} testing sequences.")
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




###
# Will use the function below if/when we have appropriate raw, processed, and interim data
###
# def create_config_dict(model_name, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate, raw_data_path=None, processed_data_path=None, interim_data_path=None, logging_level='INFO', logging_format='%(asctime)s - %(levelname)s - %(message)s'):
#     config = {
#         'model_name': model_name,
#         'model': {
#             'rnn': {
#                 'input_size': input_size,
#                 'hidden_size': hidden_size,
#                 'output_size': output_size,
#                 'num_epochs': num_epochs,
#                 'batch_size': batch_size,
#                 'learning_rate': learning_rate
#             }
#         },
#         'logging': {
#             'level': logging_level,
#             'format': logging_format
#         }
#     }
#     if raw_data_path:
#         config['data'] = {
#             'raw_data_path': raw_data_path
#         }
#     if processed_data_path:
#         if 'data' not in config:
#             config['data'] = {}
#         config['data']['processed_data_path'] = processed_data_path
#     if interim_data_path:
#         if 'data' not in config:
#             config['data'] = {}
#         config['data']['interim_data_path'] = interim_data_path
#     return config



###
# Older versions of the two functions at the top:
###
# def create_sequences(data, sequence_length=3):
#     """
#     Creates sequences of length sequence_length from the input data.

#     Parameters
#     ----------
#     data : pandas.DataFrame
#         The input data to create sequences from.
#     sequence_length : int, optional
#         The length of the sequences to create.

#     Returns
#     -------
#     numpy.ndarray
#         The input data as sequences of length sequence_length.
#     numpy.ndarray
#         The target values for the sequences.
#     """
#     try:
#         logging.info(f"Creating sequences of length {sequence_length}...")
#         x, y = [], []
#         for i in range(len(data) - sequence_length):
#             x.append(data.iloc[i:i+sequence_length].values)
#             y.append(data.iloc[i+sequence_length]['start_walk'])
#         logging.info(f"Created {len(x)} sequences.")
#         return np.array(x), np.array(y)
#     except Exception as e:
#         logging.error(f"Error creating sequences: {e}")
#         raise


# def prepare_train_test_sequences(df, sequence_length=3, split_ratio=2/3):
#     """
#     Prepares training and testing sequences from the input DataFrame.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The input DataFrame to prepare sequences from.
#     sequence_length : int, optional
#         The length of the sequences to create.
#     split_ratio : float, optional
#         The ratio of training to testing data.

#     Returns
#     -------
#     numpy.ndarray
#         The training input sequences.
#     numpy.ndarray
#         The training target values.
#     numpy.ndarray
#         The testing input sequences.
#     numpy.ndarray
#         The testing target values.
#     """
#     try:
#         logging.info("Preparing training and testing sequences...")
#         if not isinstance(df, pd.DataFrame):
#             raise TypeError("df must be a pandas DataFrame.")
#         if not isinstance(sequence_length, int):
#             raise TypeError("sequence_length must be an integer.")
#         if not isinstance(split_ratio, float):
#             raise TypeError("split_ratio must be a float.")
#         if not (0 < split_ratio < 1):
#             raise ValueError("split_ratio must be between 0 and 1.")

#         X_train, Y_train = [], []
#         X_test, Y_test = [], []
#         files = df['file'].unique()

#         for file in files:
#             logging.info(f"Processing file {file}...")
#             file_data = df[df['file'] == file].drop(['Frame', 'file'], axis=1)

#             # Create sequences for each file
#             x, y = create_sequences(file_data, sequence_length=sequence_length)

#             # Calculate the split index
#             train_size = int(len(x) * split_ratio)

#             # Split the sequences for each file
#             X_train.extend(x[:train_size])
#             Y_train.extend(y[:train_size])
#             X_test.extend(x[train_size:])
#             Y_test.extend(y[train_size:])

#         logging.info(
#             f"Prepared {len(X_train)} training sequences and {len(X_test)} testing sequences.")
#         return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
#     except Exception as e:
#         logging.error(f"Error preparing training and testing sequences: {e}")
#         raise