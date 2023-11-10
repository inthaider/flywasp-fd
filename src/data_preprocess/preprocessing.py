"""
This module contains the `DataPreprocessor` class for preprocessing a Pandas DataFrame.

The `DataPreprocessor` class includes methods for loading data, saving processed data, dropping columns, rearranging columns, calculating means, adding labels, handling infinite and NaN values, and performing other preprocessing steps. It uses the `pandas` and `numpy` libraries for data manipulation and the `logging` library for logging.

Classes:
    DataPreprocessor:
        A class for preprocessing a Pandas DataFrame. It includes methods for loading data, saving processed data, dropping columns, rearranging columns, calculating means, adding labels, handling infinite and NaN values, and performing other preprocessing steps.

Example:
    To preprocess a dataset using the DataPreprocessor class:

    >>> preprocessor = DataPreprocessor(pickle_path='path/to/data.pkl')
    >>> df_processed = preprocessor.preprocess_data(save_data=True)

Note:
    This module expects the raw data to be in a Pandas DataFrame format and to be available at the specified pickle path.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class for preprocessing a Pandas DataFrame.

    Attributes:
        df_raw (pd.DataFrame): The raw DataFrame.
        df (pd.DataFrame): The processed DataFrame.
        pickle_path (str): The path to the pickled file.
        raw_data_id (str): The ID of the raw data.
        save_data (bool): Whether to save the processed data to a pickled file.
        raw_data_path (str): The path to the raw data.
        interim_data_path (str): The path to the interim data.
        processed_data_path (str): The path to the processed data.
        timestamp (str): The timestamp of the processed data.

    Methods:
        load_data(): Loads the DataFrame from a pickled file.
        save_processed_data(): Saves the processed data to a pickled file.
        drop_columns(columns_to_drop): Drops the specified columns from the DataFrame.
        specific_rearrange(col_to_move, ref_col): Moves a column to be immediately after a reference column.
        rearrange_columns(cols_order): Rearranges the columns of the DataFrame according to the specified order.
        calculate_means(column_pairs, new_columns): Calculates the means of pairs of columns and adds the results as new columns.
        add_labels(condition_columns, new_column): Adds a new column based on conditions of existing columns.
        handle_infinity_and_na(): Replaces infinite and NaN values in the DataFrame with forward/backward filled values.
        preprocess_data(save_data): Performs preprocessing steps on the DataFrame.
    """

    def __init__(
        self,
        pickle_path=None,
        raw_data_id: str = "ff-mw",
        save_data: bool = False,
    ):
        """
        Initializes a new instance of the DataPreprocessor class.

        Args:
            pickle_path (str, optional): The path to the pickled file. Defaults to None.
            raw_data_id (str, optional): The ID of the raw data. Defaults to "ff-mw".
            save_data (bool, optional): Whether to save the processed data to a pickled file. Defaults to False.
        """
        self.df_raw = None
        self.df = None
        self.pickle_path = pickle_path
        self.raw_data_id = raw_data_id
        self.save_data = save_data
        self.raw_data_path = None
        self.interim_data_path = self.pickle_path
        self.processed_data_path = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    def load_data(self):
        """
        Loads the DataFrame from a pickled file.

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            ValueError: If an invalid pickle path is provided.
            Exception: If an error occurs while loading data.
        """
        # Logging statement to indicate the start of loading data
        logger.debug("\nLoading data in DataPreprocessor -> load_data() ...")
        try:
            if self.pickle_path:
                logger.debug(f"Loading data from {self.pickle_path}...")
                self.df_raw = pd.read_pickle(self.pickle_path)
                # Make a copy of the loaded DataFrame
                self.df = self.df_raw.copy()
                logger.debug(f"Data loaded from {self.pickle_path}.")
                return self.df_raw
            else:
                raise ValueError(
                    "\nVALUE ERROR: Provide a valid pickle path.\n"
                )
        except Exception as e:
            logger.error(f"\nERROR loading data: {e}\n")
            raise

    def save_processed_data(self) -> str:
        """
        Saves the processed data to a pickled file.

        Returns:
            str: The path to the saved file.

        Raises:
            Exception: If an error occurs while saving processed data.
        """
        # Logging statement to indicate the start of saving processed data
        logger.debug(
            "\nSaving processed data in DataPreprocessor -> save_processed_data() ..."
        )
        try:
            # Create the directory for the processed data if it doesn't exist
            processed_data_dir = Path(f"data/processed/{self.raw_data_id}")
            processed_data_dir.mkdir(parents=True, exist_ok=True)

            #
            # Generate a hash based on DataFrame metadata and some sampling
            #
            logger.debug("Generating hash for processed data...")
            # Extracting the shape, columns and a sample of the dataframe
            df_summary = f"{self.df.shape}{self.df.columns}{self.df.sample(n=10, random_state=1)}"
            # Generating the hash using the md5 algorithm based on the dataframe summary
            processed_data_hash = hashlib.md5(df_summary.encode()).hexdigest()
            logger.debug("Processed data hashed.")

            # Construct the output file path
            self.processed_data_path = (
                processed_data_dir
                / f"{self.timestamp}_processed_data_{processed_data_hash}.pkl"
            )

            # Save the processed data to the output file
            logger.debug(
                f"Saving processed data to {self.processed_data_path}..."
            )
            self.df.to_pickle(self.processed_data_path)
            logger.debug(
                f"Processed data saved to {self.processed_data_path}.\n"
            )
            return str(self.processed_data_path)
        except Exception as e:
            logger.error(f"\nERROR saving processed data: {e}\n")
            raise

    def drop_columns(self, columns_to_drop):
        """
        Drops the specified columns from the DataFrame.

        Args:
            columns_to_drop (list of str): The names of the columns to drop.

        Raises:
            Exception: If an error occurs while dropping columns.
        """
        # Logging statement to indicate the start of dropping columns
        logger.debug(
            f"\nDropping columns {columns_to_drop} in DataPreprocessor -> drop_columns() ..."
        )
        try:
            self.df.drop(columns_to_drop, axis=1, inplace=True)
            # Logging statement to indicate the end of dropping columns
            logger.debug("Columns dropped successfully.\n")
        except Exception as e:
            logger.error(f"\nERROR dropping columns: {e}\n")
            raise

    def specific_rearrange(self, col_to_move, ref_col):
        """
        Moves a column to be immediately after a reference column.

        Args:
            col_to_move (str): The name of the column to move.
            ref_col (str): The name of the reference column.

        Raises:
            Exception: If an error occurs while moving a column.
        """
        # Logging statement to indicate the start of moving a column
        logger.debug(
            "\nRearranging specific columns in DataPreprocessor -> specific_rearrange() ..."
        )
        try:
            logger.debug(
                f"Moving column {col_to_move} to be immediately after {ref_col}..."
            )
            cols = self.df.columns.tolist()
            cols.insert(
                cols.index(ref_col) + 1, cols.pop(cols.index(col_to_move))
            )
            self.df = self.df[cols]
            # Logging statement to indicate the end of moving a column
            logger.debug("Column moved successfully.\n")
        except Exception as e:
            logger.error(f"\nERROR moving column: {e}\n")
            raise

    def rearrange_columns(self, cols_order):
        """
        Rearranges the columns of the DataFrame according to the specified order.

        Args:
            cols_order (list of str): The order of the columns.

        Raises:
            Exception: If an error occurs while rearranging columns.
        """
        # Logging statement to indicate the start of rearranging columns
        logger.debug(
            "\nRearranging columns in DataPreprocessor -> rearrange_columns()"
            "..."
        )
        try:
            logger.debug(f"Rearranging columns to {cols_order}...")
            self.df = self.df[cols_order]
            # Logging statement to indicate the end of rearranging columns
            logger.debug("Columns rearranged successfully.\n")
        except Exception as e:
            logger.error(f"\nERROR rearranging columns: {e}\n")
            raise

    def calculate_means(self, column_pairs, new_columns):
        """
        Calculates the means of pairs of columns and adds the results as new columns.

        Args:
            column_pairs (list of list of str): The pairs of columns to calculate the means of.
            new_columns (list of str): The names of the new columns.

        Raises:
            Exception: If an error occurs while calculating means.
        """
        # Logging statement to indicate the start of calculating means
        logger.debug(
            "\nCalculating means in DataPreprocessor -> calculate_means() ..."
        )
        try:
            for pair, new_col in zip(column_pairs, new_columns):
                logger.debug(
                    f"Calculating mean of columns {pair} and adding as {new_col}..."
                )
                self.df[new_col] = self.df[pair].mean(axis=1)

            # Logging statement to indicate the end of calculating means
            logger.debug("Means calculated successfully.\n")
        except Exception as e:
            logger.error(f"\nERROR calculating means: {e}\n")
            raise

    def add_labels(self, condition_columns, new_column):
        """
        Adds a new column based on conditions of existing columns.

        Args:
            condition_columns (list of str): The names of the columns to use for the conditions.
            new_column (str): The name of the new column.

        Raises:
            Exception: If an error occurs while adding labels.
        """
        # Logging statement to indicate the start of adding labels
        logger.debug(
            "\nAdding labels in DataPreprocessor -> add_labels() ..."
        )
        try:
            logger.debug(
                f"Adding new column {new_column} based on conditions of "
                "columns {condition_columns}..."
            )
            self.df[new_column] = (
                (self.df[condition_columns[0]] == 1)
                & (self.df[condition_columns[1]].shift(1) == 0)
            ).astype(int)

            # Logging statement to indicate the end of adding labels
            logger.debug("Labels added successfully.\n")
        except Exception as e:
            logger.error(f"\nERROR adding labels: {e}\n")
            raise

    def handle_infinity_and_na(self):
        """
        Replaces infinite and NaN values in the DataFrame with forward/backward filled values.

        Raises:
            Exception: If an error occurs while handling infinite and NaN values.

        TODO: Do we need the df.reset_index(drop=True) line?
        TODO: Implement forward and backward filling.
        """
        try:
            logger.debug("Handling infinite and NaN values...")

            # Replace infinite values with NaN
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # df.reset_index(drop=True) resets the index of the dataframe to the default index (0, 1, 2, ...)
            # we do this because the index of the dataframe is not continuous after dropping rows
            self.df = self.df.reset_index(drop=True)

            # # Forward fill NaN values
            # self.df.fillna(method='ffill', inplace=True)

            # # Backward fill any remaining NaN values
            # self.df.fillna(method='bfill', inplace=True)

        except Exception as e:
            logger.error(f"\nERROR handling infinite and NaN values: {e}\n")
            raise

    def preprocess_data(self, save_data: bool = None):
        """
        Performs preprocessing steps on the DataFrame.

        Args:
            save_data (bool, optional): Whether to save the processed data to a pickled file. Defaults to None.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        # Add logging statements to indicate the start of preprocessing as part of the DataPreprocessor class
        logger.info("\n\nPreprocessing data (in DataPreprocessor class)...")

        # Set the value of self.save_data to the value of save_data if save_data is not None
        if save_data is not None:
            self.save_data = save_data

        # Drop the 'plot' column
        self.drop_columns(["plot"])

        # Calculate the mean of 'ANTdis_1' and 'ANTdis_2' and store it in a new column 'ANTdis'
        self.calculate_means([["ANTdis_1", "ANTdis_2"]], ["ANTdis"])

        # Add a new column 'start_walk' with value 'walk_backwards' for rows where the 'walk_backwards' column has value 'walk_backwards'
        self.add_labels(["walk_backwards", "walk_backwards"], "start_walk")

        # Replace infinity and NaN values with appropriate values
        self.handle_infinity_and_na()

        # Rearrange the column names
        self.specific_rearrange("F2Wdis_rate", "F2Wdis")
        self.rearrange_columns(
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
        )

        # Save the processed data to a pickled file if self.save_data is True.
        if self.save_data:
            logger.info("!! save_data is True !!")
            self.save_processed_data()

        # Print the shape of the dataframe and its columns using the print module.
        print(
            f"\nDataPreprocessor.preprocess_data --> Shape of the dataframe: {self.df.shape}"
        )
        print(
            f"DataPreprocessor.preprocess_data --> Columns of the dataframe: {self.df.columns}\n"
        )

        # Add logging statements to indicate the end of preprocessing as part of the DataPreprocessor class
        logger.info("Preprocessing complete (in DataPreprocessor class).\n\n")
        return self.df
