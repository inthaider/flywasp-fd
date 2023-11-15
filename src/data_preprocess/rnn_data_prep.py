"""
This module contains the `RNNDataPrep` class that prepares input data for
the RNN model.

Classes:
    RNNDataPrep:
        A class for preparing input data for the RNN model. It includes
        methods for preparing train/test data splits, creating sequences,
        and performing random oversampling.
        
Example:
    To use the RNNDataPrep class to prepare train/test data for the RNN
    model:
    
    >>> rnn_data_prep = RNNDataPrep()
    >>> rnn_data_prep.set_data_source(df_processed)
    >>> rnn_data_prep.prepare_rnn_data()

TODO:
    Need to modify DataSaver so that it saves the test_indices along
    with the train/test data splits. Otherwise loading train/test data
    from file is not feasible.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

from src.data_preprocess.data_loader import DataLoader

logger = logging.getLogger(__name__)


class RNNDataPrep:
    """
    Class for preparing input train/test data splits for the RNN model.

    Attributes:
        data_source (Optional[pd.DataFrame | (str | Path)]): The data
            source for RNNDataPrep. Can be either a preprocessed
            DataFrame or the path to the train/test data.
        train_test_dict (Dict[str, np.ndarray]): A dictionary containing
            the train/test splits.
        test_indices (np.ndarray): The indices of the test sequences.

    Methods:
        set_data_source(
            df_processed: Optional[pd.DataFrame] = None,
            train_test_path: Optional[str | Path] = None,
        ): Sets the data source for RNNDataPrep.
        get_rnn_data(
            sequence_length: int = 3,
            split_ratio: float = 2 / 3,
            rand_oversample: bool = False,
        ): Returns the train/test data splits for the RNN model.
        prepare_rnn_data(
            sequence_length: int = 3,
            split_ratio: float = 2 / 3,
            rand_oversample: bool = False,
        ): Prepares the train-test datasets for the RNN model.
        _prep_train_test_seqs(
            sequence_length: int, split_ratio: float
        ): Prepares training and testing sequences for the RNN model.
        _create_seqs(
            data: np.ndarray,
            sequence_length: int = 5,
            index_start: int = 0,
        ): Creates sequences of length `sequence_length` from the input
            `data`.
        _perform_random_oversampling(
            X_train, Y_train
        ): Performs random oversampling to balance the class
            distribution.
    """

    def __init__(self):
        """
        Initializes the RNNDataPrep object.
        """
        self.data_source: Optional[pd.DataFrame | (str | Path)] = None
        self.train_test_dict: Dict[str, np.ndarray] = {}
        self.test_indices: np.ndarray = np.array([])

    def set_data_source(
        self,
        df_processed: Optional[pd.DataFrame] = None,
        train_test_path: Optional[str | Path] = None,
    ):
        """
        Sets the data source for RNNDataPrep.

        Optionally & mutually exclusively takes either a preprocessed
        DataFrame or the path to the train/test data as input.

        Args:
            df_processed (Optional[pd.DataFrame]): The preprocessed
                DataFrame. Defaults to None.
            train_test_path (Optional[str  |  Path]): The path to the
                train/test data. Defaults to None.

        Raises:
            ValueError: If both df_processed and train_test_path are
                provided.
            ValueError: If neither df_processed nor train_test_path are
                provided.
        """
        logger.info("Setting the data source for RNNDataPrep...\n")
        if df_processed is not None and train_test_path is not None:
            raise ValueError(
                "ValueError: Only one of df_processed or train_test_path "
                "should be provided."
            )
        elif df_processed is not None:
            self.data_source = df_processed
            logger.info(
                "DataFrame set as data_source.\nUse get_rnn_data() with the "
                "required params to generate+retrieve train/test splits.\n"
            )
        elif train_test_path is not None:
            train_test_path = Path(train_test_path)
            self.data_source = train_test_path
            data_loader = DataLoader()
            self.train_test_dict = data_loader.load_train_test_data(
                train_test_path
            )
            logger.info(
                "Train/test loaded from file as data_source.\nUse "
                "get_rnn_data() without params to retrieve train/test splits."
                "\n"
            )
        else:
            raise ValueError(
                "ValueError: Either df or data_path must be provided."
            )

    def get_rnn_data(
        self,
        sequence_length: int = 3,
        split_ratio: float = 2 / 3,
        rand_oversample: bool = False,
    ):
        """
        Returns the train/test data splits for the RNN model.

        Args:
            sequence_length (int): The length of the sequences. Defaults
                to 3.
            split_ratio (float): The ratio of train to test sequences.
                Defaults to 2/3.
            rand_oversample (bool): If True, random oversampling is
                performed to balance the class distribution. Defaults to
                False.

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]: A tuple containing
                a dictionary of train/test splits and the test indices,
                all as NumPy arrays.
        """
        logger.info("Getting train-test data splits...\n")
        if isinstance(self.data_source, pd.DataFrame):
            return self.prepare_rnn_data(
                sequence_length, split_ratio, rand_oversample
            )
        else:
            return self.data_source

    def prepare_rnn_data(
        self,
        sequence_length: int = 3,
        split_ratio: float = 2 / 3,
        rand_oversample: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepares the train-test datasets for the RNN model.

        Args:
            sequence_length (int): The length of the sequences. Defaults
                to 3.
            split_ratio (float): The ratio of train to test sequences.
                Defaults to 2/3.
            rand_oversample (bool): If True, random oversampling is
                performed to balance the class distribution. Defaults to
                False.

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]: A tuple containing
                a dictionary of train/test splits and the test indices,
                all as NumPy arrays.
        """
        logger.info("Preparing train-test data splits...\n")
        # ========== Prepare sequences and train-test splits ========= #
        (
            X_train,
            Y_train,
            X_test,
            Y_test,
            self.test_indices,
        ) = self._prep_train_test_seqs(sequence_length, split_ratio)
        # == Perform Random Oversampling if rand_oversample is True == #
        if rand_oversample:
            X_train, Y_train = self._perform_random_oversampling(
                X_train, Y_train
            )

        self.train_test_dict = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_test": X_test,
            "Y_test": Y_test,
        }
        logger.info("Sequences and train-test splits prepared successfully.\n")
        return self.train_test_dict, self.test_indices

    def _prep_train_test_seqs(
        self, sequence_length: int, split_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares training and testing sequences for the RNN model.

        Args:
            sequence_length (int): The length of the sequences.
            split_ratio (float): The ratio of train to test sequences.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                np.ndarray]: A tuple containing X_train, Y_train,
                    X_test, Y_test, and test_indices as NumPy arrays.
        """
        logger.info("Starting to prepare train-test sequences...\n")
        # ================= Initial checks and setup ================= #
        # ? Should we move these assert statements to the lowest level
        # ? method that uses them?
        assert isinstance(
            self.data_source, pd.DataFrame
        ), "df must be a Pandas DataFrame."
        assert isinstance(sequence_length, int), "sequence_length must be int."
        assert 0 < split_ratio < 1, "split_ratio must be float between 0 & 1."
        logger.debug("Initial checks and setup completed.")

        # == Convert DataFrame to NumPy array after dropping columns = #
        logger.debug("Converting DataFrame to NumPy array...")
        df_values = np.array(
            self.data_source.drop(["Frame", "file"], axis=1).values
        )
        file_column = np.array(self.data_source["file"].values)
        logger.debug("Converted DataFrame to NumPy array.")

        # ======= Calculate sizes in advance for pre-allocation ====== #
        logger.debug("Calculating sizes in advance for pre-allocation...")
        unique_files = np.array(self.data_source["file"].unique())
        file_lengths = np.array(self.data_source["file"].value_counts().values)
        total_sequences = np.sum((file_lengths - sequence_length))
        train_size = int(total_sequences * split_ratio)
        test_size = total_sequences - train_size
        logger.info(
            f"Calculated train_size: {train_size}, test_size: {test_size}"
        )

        # ================= Pre-allocate NumPy arrays ================ #
        logger.debug("Pre-allocating numpy arrays...")
        # -2 because we've dropped 'Frame' and 'file'
        input_dim = self.data_source.shape[1] - 2
        X_train = np.zeros((train_size, sequence_length, input_dim - 1))
        Y_train = np.zeros(train_size)
        X_test = np.zeros((test_size, sequence_length, input_dim - 1))
        Y_test = np.zeros(test_size)
        test_indices = np.zeros(test_size, dtype=int)
        # Calculate and store the min indices for each file
        min_indices = (
            self.data_source.groupby("file")
            .apply(lambda x: x.index.min())
            .to_dict()
        )
        logger.debug("Pre-allocated NumPy arrays for train/test sets.")

        train_idx, test_idx = 0, 0
        for i, (file, file_length) in enumerate(
            zip(unique_files, file_lengths)
        ):
            print("===================")
            print(f"Fly-wasp pair # {i}")
            print(f"Processing file {file}...")

            # Use NumPy-based filtering for file_data
            file_data = df_values[file_column == file]

            logger.debug("Creating sequences for the current file...")
            # Create sequences for the current file and extract the
            # indices of the target values
            x, y, idx = self._create_seqs(
                file_data,
                sequence_length=sequence_length,
                index_start=min_indices[file],
            )

            # Calculate the split index for this file
            n = len(x)  # equal to file_length - sequence_length
            file_train_size = int(
                n * split_ratio
            )  # number of training sequences

            logger.info(f"Calculated file_train_size: {file_train_size}")

            # Calculate the end indices for the train/test data
            train_end_idx = train_idx + file_train_size
            test_end_idx = test_idx + n - file_train_size

            # ===== Add the sequences to the pre-allocated arrays ==== #
            logger.debug("Adding sequences to pre-allocated arrays...")
            X_train[train_idx:train_end_idx] = x[:file_train_size]
            Y_train[train_idx:train_end_idx] = y[:file_train_size]
            X_test[test_idx:test_end_idx] = x[file_train_size:]
            Y_test[test_idx:test_end_idx] = y[file_train_size:]
            test_indices[test_idx:test_end_idx] = idx[file_train_size:]

            # Extract the test indices corresponding to Y_test. In the
            # line below, idx[file_train_size:] is used to get the
            # indices of the target values for the test sequences and we
            # add these to test_indices for the index range
            # corresponding to the current file
            test_indices[test_idx:test_end_idx] = idx[file_train_size:]

            # Update the indices for the next iteration
            train_idx = train_end_idx
            test_idx = test_end_idx

            # Check X array shapes
            logger.debug(
                f"X_train shape: {str(X_train.shape):>10}, X_test shape: "
                f"{str(X_test.shape):>10}"
            )
            # Check Y array shapes
            logger.debug(
                f"Y_train shape: {str(Y_train.shape):>10}, Y_test shape: "
                f"{str(Y_test.shape):>10}"
            )

        print("===================")
        logger.info(
            f"Successfully prepared {len(X_train)} training sequences & "
            f"{len(X_test)} testing sequences.\n"
        )
        return X_train, Y_train, X_test, Y_test, test_indices

    def _create_seqs(
        self, data: np.ndarray, sequence_length: int = 5, index_start: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates sequences of length `sequence_length` from the input
        `data`.

        Args:
            data (np.ndarray): The data to create sequences from.
            sequence_length (int): The length of the sequences. Defaults
                to 5.
            index_start (int): The starting index for the sequences.
                Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The sequences,
                target values, and target indices as NumPy arrays.

        Note:
            This function creates sequences of length `sequence_length`
            from the input `data`. Each sequence consists of
            `sequence_length` consecutive rows of `data`, and the target
            value for each sequence is the value in the last row of the
            sequence.
        """
        n = len(data) - sequence_length
        x = np.empty(
            (n, sequence_length, data.shape[1] - 1)
        )  # -1 because we're dropping the target column in x
        y = np.empty(n)
        target_indices = np.empty(n, dtype=int)
        valid_idx = 0
        for i in range(n):
            target = data[i + sequence_length, -1]
            # -1 because we're dropping the target column in the X
            # sequences
            sequence = data[i : i + sequence_length, :-1]
            # Check if there are any missing values in the sequence or
            # target
            if not np.isnan(sequence).any() and not np.isnan(target):
                x[valid_idx] = sequence
                y[valid_idx] = target
                # Store the index of the target row/observation in the
                # sequence
                target_indices[valid_idx] = index_start + i + sequence_length
                valid_idx += 1

        # Trim the arrays to the size of valid sequences
        x = x[:valid_idx]
        y = y[:valid_idx]
        target_indices = target_indices[:valid_idx]
        return x, y, target_indices

    def _perform_random_oversampling(
        self, X_train, Y_train
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs random oversampling to balance the class distribution.

        Args:
            X_train (np.ndarray): The training data features.
            Y_train (np.ndarray): The training data labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The resampled training data
                and target values as NumPy arrays.
        """
        logger.info(
            "Performing random oversampling to balance the class "
            "distribution...\n"
        )
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, Y_train_resampled = cast(
            Tuple[np.ndarray, np.ndarray],
            ros.fit_resample(X_train.reshape(X_train.shape[0], -1), Y_train),
        )
        # ======== Reshape X_train back to its original shape ======== #
        original_shape = X_train.shape[1:]
        X_train_resampled = X_train_resampled.reshape(-1, *original_shape)
        logger.info(
            f"Original dataset shape: {X_train.shape:>10}, {Y_train.shape:>10}"
        )
        logger.info(
            f"Resampled dataset shape: {X_train_resampled.shape:>10}, "
            f"{Y_train_resampled.shape:>10}"
        )
        logger.info("Random oversampling completed.\n")
        return X_train_resampled, Y_train_resampled
