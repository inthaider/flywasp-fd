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

logger = logging.getLogger(__name__)

class RNNDataPrep:
    """
    A class for preparing data for an RNN model.

    Parameters
    ----------
    pickle_path : str, optional
        The path to the .pkl file containing the raw data. Defaults to 'data/interim/ff-mw.pkl'.
    train_test_data_par_dir : str, optional
        The path to the directory containing the .pkl files. Defaults to 'data/processed/rnn_input/'.
    save_train_test : bool, optional
        Whether to save the train and test datasets as .pkl files. Defaults to False.
    save_data : bool, optional
        Whether to save the processed data as a .pkl file. Defaults to False.

    Attributes
    ----------
    pickle_path : str
        The path to the .pkl file containing the raw data.
    train_test_data_par_dir : str
        The path to the directory containing the .pkl files.
    save_train_test : bool
        Whether to save the train and test datasets as .pkl files.
    save_data : bool
        Whether to save the processed data as a .pkl file.
    timestamp : str
        The current timestamp.
    raw_data_path : pathlib.Path
        The path to the raw data file.
    interim_data_path : pathlib.Path
        The path to the interim data file.
    processed_data_path : pathlib.Path
        The path to the processed data file.
    train_test_data_dir : pathlib.Path
        The path to the directory containing the .pkl files.
    raw_data_id : str
        The ID of the raw data.
    preprocessor : DataPreprocessor
        The DataPreprocessor object used to preprocess the raw data.
    feature_engineer : FeatureEngineer
        The FeatureEngineer object used to engineer features from the preprocessed data.

    Methods
    -------
    load_and_preprocess_data(save_data)
        Loads the raw data and performs preprocessing and feature engineering steps.
    prepare_rnn_data(df, sequence_length, split_ratio, rand_oversample, save_train_test)
        Prepares the train and test datasets for the RNN model.
    get_rnn_data(load_train_test, sequence_length, split_ratio, save_train_test, save_data)
        Provides an interface for either loading preprocessed data or preprocessing raw data, performing feature engineering, preparing sequences and train-test splits, and saving the processed data and train-test splits.
    _prep_train_test_seqs(df, sequence_length, split_ratio)
        Prepares training and testing sequences for the RNN model.
    _create_seqs(data, sequence_length)
        Creates sequences of length `sequence_length` from the input `data`.
    _perform_random_oversampling(X_train, Y_train)
        Performs random oversampling to balance the class distribution.
    _load_train_test_data()
        Loads the train and test datasets for the RNN model from 4 .pkl files.
    _save_train_test_data(X_train, Y_train, X_test, Y_test)
        Saves the train and test datasets for the RNN model as .pkl files.

    Notes
    -----
    This class provides an interface for either loading preprocessed data or preprocessing raw data, performing feature engineering, preparing sequences and train-test splits, and saving the processed data and train-test splits.
    """

    def __init__(self, pickle_path: str = "data/interim/ff-mw.pkl", train_test_data_par_dir: str = "data/processed/rnn_input/", save_train_test: bool = False, save_data: bool = False):
        """
        Initializes the RNNDataPrep object.

        Parameters
        ----------
        pickle_path : str
            The path to the .pkl file containing the raw data.
        train_test_data_dir : str, optional
            The path to the directory containing the .pkl files, by default 'data/processed/rnn_input/'
        save_train_test : bool, optional
            Whether to save the train and test datasets as .pkl files. Defaults to False.
        save_data : bool, optional
            Whether to save the processed data as a .pkl file. Defaults to False.
        """
        self.pickle_path = Path(pickle_path)
        self.train_test_data_par_dir = Path(train_test_data_par_dir)
        self.save_train_test = save_train_test
        self.save_data = save_data

        self.timestamp = datetime.now().strftime("%Y%m%d")
        self.raw_data_path = None
        self.interim_data_path = self.pickle_path
        self.processed_data_path = None
        self.train_test_data_dir = None
        self.raw_data_id = None
        self.preprocessor = None
        self.feature_engineer = None
        self.test_indices = None

    def load_and_preprocess_data(self, save_data: bool = None) -> pd.DataFrame:
        """
        Loads the raw data and performs preprocessing and feature engineering steps.

        Parameters
        ----------
        save_data : bool, optional
            Whether to save the processed data as a .pkl file. Defaults to None, in which case the value of self.save_data is used.

        Returns
        -------
        pandas.DataFrame
            The preprocessed and feature engineered data.
        """
        # Logging messages to indicate the start of the function
        logging.info(
            "\nLoading & preprocessing raw/interim data to be prepared for RNN in RNNDataPrep -> load_and_preprocess_data...")

        # Set the value of self.save_data to the value of save_data if save_data is not None
        if save_data is not None:
            self.save_data = save_data

        self.preprocessor = DataPreprocessor(
            pickle_path=self.pickle_path, save_data=self.save_data)
        self.raw_data_path = self.preprocessor.raw_data_path
        self.processed_data_path = self.preprocessor.processed_data_path
        self.raw_data_id = self.preprocessor.raw_data_id

        logging.info("Loading raw/interim data...")
        df = self.preprocessor.load_data()

        # Perform preprocessing steps
        logging.info("Performing preprocessing steps on raw/interim data...")
        df = self.preprocessor.preprocess_data()
        if df is None:
            raise ValueError(
                "DataFrame 'df' is None after preprocessing!!??!!")

        # Perform feature engineering steps
        logging.info(
            "Performing feature engineering steps on preprocessed raw/interim data...")
        self.feature_engineer = FeatureEngineer(df)
        df = self.feature_engineer.engineer_features()

        # Logging messages to indicate the end of the function
        logging.info(
            "Data loading, preprocessing, and feature engineering completed in RNNDataPrep -> load_and_preprocess_data.\n")
        return df

    def get_rnn_data(self, load_train_test: bool = False, sequence_length: int = 3, split_ratio: float = 2/3, save_train_test: bool = None, save_data: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Provides an interface for either loading preprocessed data or preprocessing raw data, performing feature engineering, preparing sequences and train-test splits, and saving the processed data and train-test splits.

        Parameters
        ----------
        load_train_test : bool, optional
            Whether to load preprocessed data or preprocess raw data. Defaults to False.
        sequence_length : int, optional
            The length of the sequences to create. Defaults to 3.
        split_ratio : float, optional
            The ratio of training to testing data. Defaults to 2/3.
        save_train_test : bool, optional
            Whether to save the train and test datasets as .pkl files. Defaults to None, in which case the value of self.save_train_test is used.
        save_data : bool, optional
            Whether to save the processed data as a .pkl file. Defaults to None, in which case the value of self.save_data is used.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The train-test splits as (X_train, Y_train, X_test, Y_test).
        """
        # Logging messages to indicate the start of the function
        logging.info(
            "\n\nGet RNN train & test datasets (in RNNDataPrep class)...")
        logging.info(f"\tThe steps are:- \n\t\tif load_train_test=True, (1) Load existing train/test data; \n\t\tELSE, (1) Load and preprocess raw/interim data, (2) Prepare RNN train & test datasets from the preprocessed data, (3) Save the train & test datasets (if save_train_test = True).")

        # Set the values of self.save_train_test and self.save_data to the values of save_train_test and save_data if they are not None
        if save_train_test is not None:
            self.save_train_test = save_train_test
        if save_data is not None:
            self.save_data = save_data

        if load_train_test:
            logging.info(
                "Load preprocessed train & test datasets in RNNDataPrep -> get_rnn_data() ...")
            self.train_test_data_dir = Path(
                f'{self.train_test_data_par_dir}/fd_v0')
            X_train, Y_train, X_test, Y_test = self._load_train_test_data()
            logging.info(
                "Train & test datasets loaded successfully in RNNDataPrep -> get_rnn_data().")
        else:
            logging.info(
                "Load raw/interim data to be preprocessed/prepared in RNNDataPrep -> get_rnn_data() ...")
            df = self.load_and_preprocess_data()
            logging.info(
                "Raw/interim data loaded successfully in RNNDataPrep -> get_rnn_data().")
            logging.info(
                "Prepare RNN train & test datasets from raw/interim data in RNNDataPrep -> get_rnn_data() ...")
            X_train, Y_train, X_test, Y_test = self.prepare_rnn_data(
                df, sequence_length=sequence_length, split_ratio=split_ratio)
            logging.info(
                "Train & test datasets prepared successfully in RNNDataPrep -> get_rnn_data().")

        # Logging messages to indicate the end of the function
        logging.info(
            "RNN train & test datasets retrieved successfully in RNNDataPrep -> get_rnn_data().\n\n")
        return X_train, Y_train, X_test, Y_test

    def prepare_rnn_data(self, df: pd.DataFrame, sequence_length: int = 3, split_ratio: float = 2/3, rand_oversample: bool = False, save_train_test: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares the train and test datasets for the RNN model.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to use for preparing the train and test datasets.
        sequence_length : int, optional
            The length of the sequences to create. Defaults to 3.
        split_ratio : float, optional
            The ratio of training to testing data. Defaults to 2/3.
        rand_oversample : bool, optional
            Whether to perform random oversampling to balance the class distribution. Defaults to False.
        save_train_test : bool, optional
            Whether to save the train and test datasets as .pkl files. Defaults to None, in which case the value of self.save_train_test is used.

        Returns
        -------
        X_train : numpy.ndarray
            The training input sequences.
        Y_train : numpy.ndarray
            The training target values.
        X_test : numpy.ndarray
            The testing input sequences.
        Y_test : numpy.ndarray
            The testing target values.
        """
        # Logging messages to indicate the start of the function
        logging.info(
            "\nPreparing sequences and train-test splits for RNN in RNNDataPrep -> prepare_rnn_data...")

        # Set the value of self.save_train_test to the value of save_train_test if save_data is not None
        if save_train_test is not None:
            self.save_train_test = save_train_test

        # Prepare sequences and train-test splits
        X_train, Y_train, X_test, Y_test, _ = self._prep_train_test_seqs(
            df, sequence_length=sequence_length, split_ratio=split_ratio)

        # Perform Random Oversampling if rand_oversample is True
        if rand_oversample:
            X_train, Y_train = self._perform_random_oversampling(
                X_train, Y_train)

        # Save the train-test splits if self.save_train_test is True
        if self.save_train_test:
            logging.info("Saving train-test splits...")
            rnn_data_path = self._save_train_test_data(
                X_train, Y_train, X_test, Y_test)

        # Logging messages to indicate the end of the function
        logging.info(
            "Sequences and train-test splits prepared successfully in RNNDataPrep -> prepare_rnn_data.\n")
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
        X_train : numpy.ndarray
            The training input sequences.
        Y_train : numpy.ndarray
            The training target values.
        X_test : numpy.ndarray
            The testing input sequences.
        Y_test : numpy.ndarray
            The testing target values.
        test_indices : numpy.ndarray
            The indices of the target values in the original test data.
        """
        logging.info(
            "\nStarting to prepare train and test sequences in RNNDataPrep -> _prep_train_test_seqs...")

        # Initial checks and setup
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame."
        assert isinstance(
            sequence_length, int), "sequence_length must be an integer."
        assert isinstance(split_ratio, float), "split_ratio must be a float."
        assert 0 < split_ratio < 1, "split_ratio must be between 0 and 1."

        logging.debug("Initial checks and setup completed.")

        logging.debug("Converting DataFrame to NumPy array...")

        # Convert DataFrame to NumPy array after dropping unnecessary columns
        df_values = df.drop(['Frame', 'file'], axis=1).values
        file_column = df['file'].values

        logging.debug("Converted DataFrame to NumPy array.")

        logging.debug("Calculating sizes in advance for pre-allocation...")

        # Calculate sizes in advance for pre-allocation
        unique_files = df['file'].unique()
        file_lengths = df['file'].value_counts().values
        total_sequences = np.sum(file_lengths - sequence_length)
        train_size = int(total_sequences * split_ratio)
        test_size = total_sequences - train_size

        logging.info(
            f"Calculated train_size: {train_size}, test_size: {test_size}")

        logging.debug("Pre-allocating numpy arrays...")

        # Pre-allocate NumPy arrays
        # -2 because we've dropped 'Frame' and 'file'
        input_dim = df.shape[1] - 2
        X_train = np.zeros((train_size, sequence_length, input_dim - 1))
        Y_train = np.zeros(train_size)
        X_test = np.zeros((test_size, sequence_length, input_dim - 1))
        Y_test = np.zeros(test_size)

        logging.debug("Pre-allocated NumPy arrays for train and test sets.")

        test_indices = []
        train_idx, test_idx = 0, 0
        for i, (file, file_length) in enumerate(zip(unique_files, file_lengths)):
            print(f"===================")
            print(f"Fly-wasp pair # {i}")
            print(f"Processing file {file}...")

            # Use NumPy-based filtering for file_data
            file_data = df_values[file_column == file]

            logging.debug("Creating sequences for the current file...")

            # Create sequences for the current file
            # and extract the indices of the target values
            x, y, idx = self._create_seqs(
                file_data, sequence_length=sequence_length)

            # Calculate the split index for this file
            n = len(x)  # equal to file_length - sequence_length
            # number of training sequences
            file_train_size = int(n * split_ratio)

            logging.info(f"Calculated file_train_size: {file_train_size}")

            logging.debug("Adding sequences to pre-allocated arrays...")

            # Add the sequences to the pre-allocated arrays
            X_train[train_idx:train_idx + file_train_size] = x[:file_train_size]
            Y_train[train_idx:train_idx + file_train_size] = y[:file_train_size]
            X_test[test_idx:test_idx + n - file_train_size] = x[file_train_size:]
            Y_test[test_idx:test_idx + n - file_train_size] = y[file_train_size:]

            # Extract the test indices corresponding to Y_test
            # In the line below, [file_train_size:] is used to get the test indices
            # then, idx[file_train_size:] is used to get the indices of the target values for the test sequences
            # finally, we extend the test_indices list with these indices
            test_indices.extend(idx[file_train_size:])
            
            # Check X array shapes
            logging.debug(
                f"X_train shape: {str(X_train.shape):>10}, X_test shape: {str(X_test.shape):>10}")
            # Check Y array shapes
            logging.debug(
                f"Y_train shape: {str(Y_train.shape):>10}, Y_test shape: {str(Y_test.shape):>10}")

            # Update the indices for the next iteration
            train_idx += file_train_size
            test_idx += n - file_train_size

        print(f"===================")

        logging.info(
            f"\nPrepared {len(X_train)} training sequences and {len(X_test)} testing sequences.")
        logging.info(
            "Train and test sequence preparation completed in RNNDataPrep -> _prep_train_test_seqs.\n")

        self.test_indices = test_indices
        return X_train, Y_train, X_test, Y_test, test_indices


    def _create_seqs(self, data: np.ndarray, sequence_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sequences of length `sequence_length` from the input `data`.

        Parameters
        ----------
        data : numpy.ndarray
            The input data, with shape `(n_samples, n_features)`.
        sequence_length : int
            The length of the sequences to create. Defaults to 5.

        Returns
        -------
        x : numpy.ndarray
            The input sequences, with shape `(n_samples - sequence_length, sequence_length, n_features)`.
        y : numpy.ndarray
            The target values, with shape `(n_samples - sequence_length,)`.
        target_indices : numpy.ndarray
            The indices of the target values in the original data.

        Notes
        -----
        This function creates sequences of length `sequence_length` from the input `data`. Each sequence consists of `sequence_length`
        consecutive rows of `data`, and the target value for each sequence is the value in the last row of the sequence.
        """
        n = len(data) - sequence_length
        # -1 because we're dropping the target column in x
        x = np.empty((n, sequence_length, data.shape[1]-1))
        y = np.empty(n)
        target_indices = np.empty(n, dtype=int)

        valid_idx = 0
        index_start = 0
        for i in range(n):
            target = data[i + sequence_length, -1]
            # -1 because we're dropping the target column in the X sequences
            sequence = data[i:i + sequence_length, :-1]

            # Check if there are any missing values in the sequence or target
            if not np.isnan(sequence).any() and not np.isnan(target):
                x[valid_idx] = sequence
                y[valid_idx] = target

                # Store the index of the target row/observation in the sequence
                target_indices[valid_idx] = index_start + i + sequence_length

                valid_idx += 1

        # Trim the arrays to the size of valid sequences
        x = x[:valid_idx]
        y = y[:valid_idx]
        target_indices = target_indices[:valid_idx]
        return x, y, target_indices

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
        # Logging messages to indicate the start of the function
        logging.info(
            "\nPerforming random oversampling to balance the class distribution...")

        ros = RandomOverSampler(random_state=42)
        X_train_resampled, Y_train_resampled = ros.fit_resample(
            X_train.reshape(X_train.shape[0], -1), Y_train)

        # Reshape X_train back to its original shape
        original_shape = X_train.shape[1:]
        X_train_resampled = X_train_resampled.reshape(-1, *original_shape)

        # Logging messages comparing the original and resampled dataset shapes for both X_train and Y_train with appropriate formatting and spacing in the printed output (using f-strings and :> formatting)
        logging.info(
            f"Original dataset shape: {X_train.shape:>10}, {Y_train.shape:>10}")
        logging.info(
            f"Resampled dataset shape: {X_train_resampled.shape:>10}, {Y_train_resampled.shape:>10}")

        # Logging messages to indicate the end of the function
        logging.info("Random oversampling completed.\n")
        return X_train_resampled, Y_train_resampled

    def _load_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the train and test datasets for the RNN model from 4 .pkl files.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The train-test splits as (X_train, Y_train, X_test, Y_test).
        """
        # Logging messages to indicate the start of the function
        logging.info("\nLoading train and test datasets...")
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

            # Logging messages to indicate the end of the function
            logging.info("Train and test datasets loaded successfully.\n")
            return X_train, Y_train, X_test, Y_test
        except Exception as e:
            logging.error(f"\nERROR loading train and test datasets: {e}\n")
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
        # Logging messages to indicate the start of the function
        logging.info("\nSaving train and test datasets...")
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

            # Logging messages to indicate the end of the function
            logging.info(
                f"Successfully saved train and test datasets to {self.train_test_data_dir}.\n")
            return str(self.train_test_data_dir)
        except Exception as e:
            logging.error(f"\ERROR saving train and test datasets: {e}\n")
            raise
