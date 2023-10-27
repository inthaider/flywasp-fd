import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

from src.data_preprocess.feature_engineering import FeatureEngineer
from src.data_preprocess.preprocessing import DataPreprocessor


class RNNDataPrep:
    """
    A class for preparing data for an RNN model.
    """

    def __init__(self, pickle_path: str = "data/interim/ff-mw.pkl", train_test_data_par_dir: str = "data/processed/rnn_input/"):
        """
        Initializes the RNNDataPrep object.

        Parameters
        ----------
        pickle_path : str
            The path to the .pkl file containing the raw data.
        train_test_data_dir : str, optional
            The path to the directory containing the .pkl files, by default 'data/processed/rnn_input/'
        """
        self.pickle_path = Path(pickle_path)
        self.train_test_data_par_dir = Path(train_test_data_par_dir)
        self.train_test_data_dir = None
        self.interim_data_path = self.pickle_path

        self.timestamp = datetime.now().strftime("%Y%m%d")
        
        self.preprocessor = None
        self.raw_data_path = None
        self.processed_data_path = None
        self.raw_data_id = None
        self.feature_engineer = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the raw data and performs preprocessing and feature engineering steps.

        Returns
        -------
        pandas.DataFrame
            The preprocessed and feature engineered data.
        """
        self.preprocessor = DataPreprocessor(pickle_path=self.pickle_path)
        self.raw_data_path = self.preprocessor.raw_data_path
        self.processed_data_path = self.preprocessor.processed_data_path
        self.raw_data_id = self.preprocessor.raw_data_id
        logging.info("Loading data...")
        df = self.preprocessor.load_data()

        # Perform preprocessing steps
        logging.info("Performing preprocessing steps...")
        df = self.preprocessor.preprocess_data()
        if df is None:
            raise ValueError("DataFrame 'df' is None after preprocessing!!??!!")
        
        # Perform feature engineering steps
        logging.info("Performing feature engineering steps...")
        self.feature_engineer = FeatureEngineer(df)
        df = self.feature_engineer.engineer_features()

        return df

    def prepare_rnn_data(self, df: pd.DataFrame, sequence_length: int = 3, split_ratio: float = 2/3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares the train and test datasets for the RNN model.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to use for preparing the train and test datasets.
        sequence_length : int, optional
            The length of the sequences to create. Default is 3.
        split_ratio : float, optional
            The ratio of training to testing data. Default is 2/3.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The train-test splits as (X_train, Y_train, X_test, Y_test).
        """
        # Prepare sequences and train-test splits
        logging.info("Preparing sequences and train-test splits...")
        X_train, Y_train, X_test, Y_test = self._prep_train_test_seqs(
            df, sequence_length=sequence_length, split_ratio=split_ratio)

        # Perform Random Oversampling
        X_train, Y_train = self._perform_random_oversampling(X_train, Y_train)
        
        # Save the train-test splits
        logging.info("Saving train-test splits...")
        rnn_data_path = self._save_train_test_data(
            X_train, Y_train, X_test, Y_test)

        return X_train, Y_train, X_test, Y_test

    def get_rnn_data(self, load_train_test: bool = False, sequence_length: int = 3, split_ratio: float = 2/3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Provides an interface for either loading preprocessed data or preprocessing raw data, performing feature engineering, preparing sequences and train-test splits, and saving the processed data and train-test splits.

        Parameters
        ----------
        load_train_test : bool, optional
            Whether to load preprocessed data or preprocess raw data. Default is False.
        sequence_length : int, optional
            The length of the sequences to create. Default is 3.
        split_ratio : float, optional
            The ratio of training to testing data. Default is 2/3.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The train-test splits as (X_train, Y_train, X_test, Y_test).
        """
        if load_train_test:
            logging.info("Loading preprocessed train & test datasets...")
            self.train_test_data_dir = Path(f'{self.train_test_data_par_dir}/fd_v0')
            X_train, Y_train, X_test, Y_test = self._load_train_test_data()
            logging.info("Train & test datasets loaded successfully.")
        else:
            logging.info("Loading raw/interim data to be preprocessed/prepared...")
            df = self.load_data()
            logging.info("Raw/interim data loaded successfully.")
            logging.info("Preparing RNN train & test datasets from raw/interim data.")
            X_train, Y_train, X_test, Y_test = self.prepare_rnn_data(
                df, sequence_length=sequence_length, split_ratio=split_ratio)
            logging.info("Train & test datasets prepared successfully.")

        return X_train, Y_train, X_test, Y_test

    def _prep_train_test_seqs(self, df: pd.DataFrame, sequence_length: int, split_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares training and testing sequences for the RNN model.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to use for preparing the train and test datasets.
        sequence_length : int
            The length of the sequences to create.
        split_ratio : float
            The ratio of training to testing data.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The train-test splits as (X_train, Y_train, X_test, Y_test).
        """
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
        # -2 because we're dropping 'Frame' and 'file'
        input_dim = df.shape[1] - 2
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
            x, y = self._create_seqs(
                file_data, sequence_length=sequence_length)

            # Calculate the split index for this file
            n = len(x)
            file_train_size = int(n * split_ratio)

            # Add the sequences to the pre-allocated arrays
            X_train[train_idx:train_idx +
                    file_train_size] = x[:file_train_size]
            Y_train[train_idx:train_idx +
                    file_train_size] = y[:file_train_size]
            X_test[test_idx:test_idx + n -
                   file_train_size] = x[file_train_size:]
            Y_test[test_idx:test_idx + n -
                   file_train_size] = y[file_train_size:]
            
            # Drop the target column from the input sequences
            X_train = X_train[:, :, :-1]
            X_test = X_test[:, :, :-1]

            # Update the indices for the next iteration
            train_idx += file_train_size
            test_idx += n - file_train_size

            i = i+1

        logging.info(
            f"Prepared {len(X_train)} training sequences and {len(X_test)} testing sequences.")
        return X_train, Y_train, X_test, Y_test

    def _create_seqs(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sequences of length `sequence_length` from the input `data`.

        Parameters
        ----------
        data : numpy.ndarray
            The input data, with shape `(n_samples, n_features)`.
        sequence_length : int
            The length of the sequences to create.

        Returns
        -------
        x : numpy.ndarray
            The input sequences, with shape `(n_samples - sequence_length, sequence_length, n_features)`.
        y : numpy.ndarray
            The target values, with shape `(n_samples - sequence_length,)`.

        Notes
        -----
        This function creates sequences of length `sequence_length` from the input `data`. Each sequence consists of `sequence_length`
        consecutive rows of `data`, and the target value for each sequence is the value in the last row of the sequence.
        """
        n = len(data) - sequence_length
        x = np.empty((n, sequence_length, data.shape[1]))
        y = np.empty(n)

        valid_idx = 0
        for i in range(n):
            sequence = data[i:i + sequence_length]
            target = data[i + sequence_length, -1]

            # Check if there are any missing values in the sequence or target
            if not np.isnan(sequence).any() and not np.isnan(target):
                x[valid_idx] = sequence
                y[valid_idx] = target
                valid_idx += 1

        # Trim the arrays to the size of valid sequences
        x = x[:valid_idx]
        y = y[:valid_idx]
        return x, y


    def _perform_random_oversampling(self, X_train, Y_train):
        """
        Performs random oversampling to balance the class distribution.

        Parameters
        ----------
        X_train : numpy.ndarray
            The training input sequences.
        Y_train : numpy.ndarray
            The training target values.

        Returns
        -------
        X_train_resampled : numpy.ndarray
            The resampled training input sequences.
        Y_train_resampled : numpy.ndarray
            The resampled training target values.
        """
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, Y_train_resampled = ros.fit_resample(
            X_train.reshape(X_train.shape[0], -1), Y_train)

        # Reshape X_train back to its original shape
        original_shape = X_train.shape[1:]
        X_train_resampled = X_train_resampled.reshape(-1, *original_shape)

        # Logging messages comparing the original and resampled dataset shapes for both X_train and Y_train with appropriate formatting and spacing in the printed output (using f-strings and :> formatting)
        logging.info(f"Original dataset shape: {X_train.shape:>10}, {Y_train.shape:>10}")
        logging.info(f"Resampled dataset shape: {X_train_resampled.shape:>10}, {Y_train_resampled.shape:>10}")



        return X_train_resampled, Y_train_resampled

    def _load_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the train and test datasets for the RNN model from 4 .pkl files.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The train-test splits as (X_train, Y_train, X_test, Y_test).
        """
        try:
            # Load the train and test datasets from .pkl files
            X_train_file = Path(self.train_test_data_dir) / "X_train.pkl"
            Y_train_file = Path(self.train_test_data_dir) / "Y_train.pkl"
            X_test_file = Path(self.train_test_data_dir) / "X_test.pkl"
            Y_test_file = Path(self.train_test_data_dir) / "Y_test.pkl"

            X_train = pickle.load(open(X_train_file, "rb"))
            Y_train = pickle.load(open(Y_train_file, "rb"))
            X_test = pickle.load(open(X_test_file, "rb"))
            Y_test = pickle.load(open(Y_test_file, "rb"))

            return X_train, Y_train, X_test, Y_test
        except Exception as e:
            logging.error(f"Error loading train and test datasets: {e}")
            raise

    def _save_train_test_data(self, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray) -> str:
        """
        Saves the train and test datasets for the RNN model as .pkl files.

        Parameters
        ----------
        X_train : numpy.ndarray
            The training input sequences.
        Y_train : numpy.ndarray
            The training target values.
        X_test : numpy.ndarray
            The testing input sequences.
        Y_test : numpy.ndarray
            The testing target values.

        Returns
        -------
        str
            The path to the directory containing the saved train-test splits.
        """
        try:
            # Create a timestamped directory for the processed data
            self.train_test_data_dir = Path(
                f"{self.train_test_data_par_dir}/{self.timestamp}")
            self.train_test_data_dir.mkdir(parents=True, exist_ok=True)

            # Save the train and test datasets as .pkl files
            X_train_file = self.train_test_data_dir / "X_train.pkl"
            Y_train_file = self.train_test_data_dir / "Y_train.pkl"
            X_test_file = self.train_test_data_dir / "X_test.pkl"
            Y_test_file = self.train_test_data_dir / "Y_test.pkl"

            with open(X_train_file, "wb") as f:
                pickle.dump(X_train, f)
            with open(Y_train_file, "wb") as f:
                pickle.dump(Y_train, f)
            with open(X_test_file, "wb") as f:
                pickle.dump(X_test, f)
            with open(Y_test_file, "wb") as f:
                pickle.dump(Y_test, f)

            logging.info(
                f"Saved train and test datasets to {self.train_test_data_dir}.")
        except Exception as e:
            logging.error(f"Error saving train and test datasets: {e}")
            raise
        return str(self.train_test_data_dir)
