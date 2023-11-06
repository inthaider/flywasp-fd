import logging
import numpy as np
import pandas as pd
import hashlib

def preprocess_data(df):
    """
    Performs preprocessing steps on the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame.
    """
    preprocessor = DataPreprocessor(df=df)
    preprocessor.drop_columns(["plot"])  # Drop the 'plot' column
    # Calculate the mean of 'ANTdis_1' and 'ANTdis_2' and store it in a new column 'ANTdis'
    preprocessor.calculate_means([["ANTdis_1", "ANTdis_2"]], ["ANTdis"])
    # Add a new column 'start_walk' with value 'walk_backwards' for rows where the 'walk_backwards' column has value 'walk_backwards'
    preprocessor.add_labels(["walk_backwards", "walk_backwards"], "start_walk")
    # Replace infinity and NaN values with appropriate values
    preprocessor.handle_infinity_and_na()
    preprocessor.specific_rearrange(
        "F2Wdis_rate", "F2Wdis"
    )  # Rearrange the column names
    preprocessor.rearrange_columns(
        [
            "Frame",
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
            "file",
            "start_walk",
        ]
    )  # Rearrange the columns in a specific order
    return preprocessor.df

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