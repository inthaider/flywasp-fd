"""
This module contains the `DataPreprocessor` class for preprocessing a
Pandas DataFrame.

Classes:
    DataPreprocessor:
        A class for preprocessing a Pandas DataFrame. It includes
        methods for dropping columns, rearranging columns, calculating
        means, adding labels, handling infinite and NaN values, and
        performing other preprocessing steps.

Example:
    To preprocess a dataset using the DataPreprocessor class:

    >>> preprocessor = DataPreprocessor(df)
    >>> df_raw = preprocessor.get_preprocessed_data()

Note:
    This module expects the raw data to be in a Pandas DataFrame format
    and to be available at the specified pickle path.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_preprocess.data_loader import DataLoader

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Class for preprocessing Pandas DataFrame containing our raw data.

    Attributes:
        raw_data_id (str): The ID of the raw data.
        data_type (str): The type of data at the path. Should be either
            "raw" or "processed".
        df_raw (pd.DataFrame): The raw DataFrame.
        df (pd.DataFrame): The processed DataFrame.
        data_loader (DataLoader): The DataLoader object for loading raw
            or processed data.

    Methods:
        set_data_source(data_type: str, data_path: str | Path): Sets the
            data source for DataPreprocessor.
        get_preprocessed_data() -> pd.DataFrame: High-level method to
            orchestrate preprocessing steps on the DataFrame.
        drop_columns(columns_to_drop: list[str]): Drops the specified
            columns from the DataFrame.
        calculate_means(
            column_pairs: list[list[str]], new_columns: list[str]
        ): Calculates the means of pairs of columns and adds the
            results as new columns.
        add_labels(condition_columns: list[str], new_column: str): Adds
            a new column based on conditions of existing columns.
        handle_infinity_and_na(): Replaces infinite and NaN values in
            the DataFrame with forward/backward filled values.
        specific_rearrange(col_to_move: str, ref_col: str): Moves a
            column to be immediately after a reference column.
        rearrange_columns(cols_order: list[str]): Rearranges the columns
            of the DataFrame according to the specified order.
    """

    def __init__(
        self,
        raw_data_id: str = "ff-mw",
    ):
        """
        Initializes DataPreprocessor with the given DataFrame.
        """
        self.raw_data_id = raw_data_id
        self.data_type: Optional[str] = None
        self.df_raw: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.data_loader = DataLoader()
        # self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    def set_data_source(
        self,
        data_type: str,
        data_path: str | Path,
    ):
        """
        Sets the data source for DataPreprocessor.

        Takes the path to either the raw data or an existing processed
        data pickle file.

        Args:
            data_type (str): The type of data at the path. Should be
                either "raw" or "processed".
            data_path (str | Path): The path to the data.

        Raises:
            ValueError: If data_type is not "raw" or "processed".
        """
        self.data_type = data_type
        logger.info("Setting the data source for DataPreprocessor...\n")
        if data_type == "raw":
            logger.info("Loading raw DataFrame from file...\n")
            self.df_raw = self.data_loader.load_raw_data(data_path)
            self.df = self.df_raw.copy()
            logger.info(
                "Raw DataFrame loaded from file and set as data_source.\nUse "
                "get_processed_data() to preprocess raw data and retrieve the "
                "processed DataFrame.\n"
            )
        elif data_type == "processed":
            self.df = self.data_loader.load_processed_data(data_path)
            logger.info(
                "Processed DataFrame loaded from file as data_source.\nUse "
                "get_processed_data() to retrieve the loaded processed "
                "DataFrame.\n"
            )
        else:
            raise ValueError(
                "ValueError: data_type must be either 'raw' or 'processed'.\n"
            )
        return self

    def get_preprocessed_data(self) -> pd.DataFrame:
        """
        Performs preprocessing steps on the DataFrame if the data source
        is raw. Returns the processed DataFrame if the data source is
        processed.

        Returns:
            pd.DataFrame: The processed DataFrame.

        Raises:
            ValueError: If the data source has not been set.
        """
        if self.df is None:
            raise ValueError(
                "ValueError: data_source is None. Please set the data source "
                "using set_data_source() before calling get_rnn_data()."
            )
        if self.data_type == "raw":
            logger.info("Preprocessing data...\n")
            # Drop the 'plot' column
            self._drop_columns(["plot"])
            # Calculate the mean of 'ANTdis_1' and 'ANTdis_2' and store it
            # in a new column 'ANTdis'
            self._calculate_means([["ANTdis_1", "ANTdis_2"]], ["ANTdis"])
            # Add a new column 'start_walk' with value 'walk_backwards' for
            # rows where the 'walk_backwards' column has value
            self._add_labels(
                ["walk_backwards", "walk_backwards"], "start_walk"
            )
            # Replace infinity and NaN values with appropriate values
            self._handle_infinity_and_na()
            # Rearrange the column names
            self._specific_rearrange("F2Wdis_rate", "F2Wdis")
            self._rearrange_columns(
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
            # Print the shape of the dataframe and its columns
            print(f"Shape of the DataFrame: {self.df.shape}")
            print(f"Columns of the DataFrame: {self.df.columns}\n")
            logger.info("Preprocessing complete.\n\n")
            return self.df
        else:
            logger.info("Returning processed DataFrame.\n")
            return self.df

    def _drop_columns(self, columns_to_drop: list[str]):
        """
        Drops the specified columns from the DataFrame.

        Args:
            columns_to_drop (list of str): The names of the columns to
                drop.

        Raises:
            Exception: If an error occurs while dropping columns.
        """
        assert isinstance(self.df, pd.DataFrame), "df must be a DataFrame."
        logger.debug(f"Dropping columns {columns_to_drop}...\n")
        try:
            self.df.drop(columns_to_drop, axis=1, inplace=True)
            logger.debug("Columns dropped successfully.\n")
            return self
        except Exception as e:
            logger.error(f"ERROR dropping columns: {e}\n")
            raise

    def _calculate_means(
        self, column_pairs: list[list[str]], new_columns: list[str]
    ):
        """
        Calculates the means of pairs of columns and adds the results as
        new columns.

        Args:
            column_pairs (list of list of str): The pairs of columns to
                calculate the means of.
            new_columns (list of str): The names of the new columns.

        Raises:
            Exception: If an error occurs while calculating means.
        """
        assert isinstance(self.df, pd.DataFrame), "df must be a DataFrame."
        logger.debug("Calculating means...")
        try:
            for pair, new_col in zip(column_pairs, new_columns):
                logger.debug(
                    f"Calculating mean of columns {pair} and adding as "
                    f"{new_col}..."
                )
                self.df[new_col] = self.df[pair].mean(axis=1)
            logger.debug("Means calculated successfully.\n")
            return self
        except Exception as e:
            logger.error(f"ERROR calculating means: {e}\n")
            raise

    def _add_labels(self, condition_columns: list[str], new_column: str):
        """
        Adds a new column based on conditions of existing columns.

        Args:
            condition_columns (list of str): The names of the columns to
                use for the conditions.
            new_column (str): The name of the new column.

        Raises:
            Exception: If an error occurs while adding labels.
        """
        assert isinstance(self.df, pd.DataFrame), "df must be a DataFrame."
        logger.debug("Adding labels...")
        try:
            logger.debug(
                f"Adding new column {new_column} based on conditions of "
                "columns {condition_columns}..."
            )
            self.df[new_column] = (
                (self.df[condition_columns[0]] == 1)
                & (self.df[condition_columns[1]].shift(1) == 0)
            ).astype(int)
            logger.debug("Labels added successfully.\n")
            return self
        except Exception as e:
            logger.error(f"ERROR adding labels: {e}\n")
            raise

    def _handle_infinity_and_na(self):
        """
        Replaces infinite and NaN values in the DataFrame with
        forward/backward filled values.

        Raises:
            Exception: If an error occurs while handling infinite and
                NaN values.

        TODO: Do we need the df.reset_index(drop=True) line?
        TODO: Implement forward and backward filling.
        """
        assert isinstance(self.df, pd.DataFrame), "df must be a DataFrame."
        try:
            logger.debug("Handling infinite and NaN values...")
            # Replace infinite values with NaN
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # df.reset_index(drop=True) resets the index of the
            # dataframe to the default index (0, 1, 2, ...) we do this
            # because the index of the dataframe is not continuous after
            # dropping rows
            self.df = self.df.reset_index(drop=True)
            # # Forward fill NaN values
            # self.df.fillna(method='ffill', inplace=True)
            # # Backward fill any remaining NaN values
            # self.df.fillna(method='bfill', inplace=True)
            logger.debug("Infinite and NaN values handled successfully.\n")
            return self
        except Exception as e:
            logger.error(f"ERROR handling infinite and NaN values: {e}\n")
            raise

    def _specific_rearrange(self, col_to_move: str, ref_col: str):
        """
        Moves a column to be immediately after a reference column.

        Args:
            col_to_move (str): The name of the column to move.
            ref_col (str): The name of the reference column.

        Raises:
            Exception: If an error occurs while moving a column.
        """
        assert isinstance(self.df, pd.DataFrame), "df must be a DataFrame."
        logger.debug("Rearranging specific columns...\n")
        try:
            logger.debug(
                f"Moving column {col_to_move} to be immediately after "
                f"{ref_col}..."
            )
            cols = self.df.columns.tolist()
            cols.insert(
                cols.index(ref_col) + 1, cols.pop(cols.index(col_to_move))
            )
            self.df = self.df[cols]
            logger.debug("Column moved successfully.\n")
            return self
        except Exception as e:
            logger.error(f"ERROR moving column: {e}\n")
            raise

    def _rearrange_columns(self, cols_order: list[str]):
        """
        Rearranges the columns of the DataFrame according to the
        specified order.

        Args:
            cols_order (list of str): The order of the columns.

        Raises:
            Exception: If an error occurs while rearranging columns.
        """
        assert isinstance(self.df, pd.DataFrame), "df must be a DataFrame."
        logger.debug("Rearranging columns...\n")
        try:
            logger.debug(f"Rearranging columns to {cols_order}...")
            self.df = self.df[cols_order]
            logger.debug("Columns rearranged successfully.\n")
            return self
        except Exception as e:
            logger.error(f"ERROR rearranging columns: {e}\n")
            raise
