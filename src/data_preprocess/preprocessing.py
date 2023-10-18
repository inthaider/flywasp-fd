import hashlib
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class DataPreprocessor:
    """
    A class for preprocessing a Pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        The DataFrame to preprocess.
    pickle_path : str, optional
        The path to a pickled DataFrame to load.

    Attributes
    ----------
    df : pandas.DataFrame
        The DataFrame to preprocess.
    pickle_path : str
        The path to a pickled DataFrame to load.

    Methods
    -------
    load_data()
        Loads the DataFrame from a pickled file.
    drop_columns(columns_to_drop)
        Drops the specified columns from the DataFrame.
    specific_rearrange(col_to_move, ref_col)
        Moves a column to be immediately after a reference column.
    rearrange_columns(cols_order)
        Rearranges the columns of the DataFrame according to the specified order.
    calculate_means(column_pairs, new_columns)
        Calculates the means of pairs of columns and adds the results as new columns.
    add_labels(condition_columns, new_column)
        Adds a new column based on conditions of existing columns.
    handle_infinity_and_na()
        Replaces infinite and NaN values in the DataFrame with NaN.
    """

    def __init__(self, df=None, pickle_path=None):
        """
        Initializes a new instance of the DataPreprocessor class.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            The DataFrame to preprocess.
        pickle_path : str, optional
            The path to a pickled DataFrame to load.
        """
        self.df = df
        self.pickle_path = pickle_path

    def load_data(self):
        """
        Loads the DataFrame from a pickled file.

        Returns
        -------
        pandas.DataFrame
            The loaded DataFrame.
        """
        try:
            if self.pickle_path:
                logging.info(f"Loading data from {self.pickle_path}...")
                self.df = pd.read_pickle(self.pickle_path)
                return self.df
            else:
                raise ValueError("Provide a valid pickle path.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def save_processed_data(self, input_data: str, timestamp: str = datetime.now().strftime("%Y%m%d_%H%M")) -> str:
        """
        Saves the processed data to a pickled file.

        Parameters
        ----------
        input_data : str
            The name of the input data file.
        timestamp : str, optional
            The timestamp to use in the output file name.

        Returns
        -------
        str
            The path to the saved pickled file.
        """
        try:
            # Create the directory for the processed data if it doesn't exist
            processed_data_dir = Path(f"data/processed/{input_data}")
            processed_data_dir.mkdir(parents=True, exist_ok=True)

            # Generate a unique hash for the processed data
            processed_data_hash = self.df.to_string().encode('utf-8')
            processed_data_hash = hashlib.md5(processed_data_hash).hexdigest()

            # Construct the output file path
            processed_data_path = processed_data_dir / f"{timestamp}_processed_data_{processed_data_hash}.pkl"

            # Save the processed data to the output file
            logging.info(f"Saving processed data to {processed_data_path}...")
            self.df.to_pickle(processed_data_path)

            return str(processed_data_path)
        except Exception as e:
            logging.error(f"Error saving processed data: {e}")
            raise

    def drop_columns(self, columns_to_drop):
        """
        Drops the specified columns from the DataFrame.

        Parameters
        ----------
        columns_to_drop : list of str
            The names of the columns to drop.
        """
        try:
            logging.info(f"Dropping columns {columns_to_drop}...")
            self.df.drop(columns_to_drop, axis=1, inplace=True)
        except Exception as e:
            logging.error(f"Error dropping columns: {e}")
            raise

    def specific_rearrange(self, col_to_move, ref_col):
        """
        Moves a column to be immediately after a reference column.

        Parameters
        ----------
        col_to_move : str
            The name of the column to move.
        ref_col : str
            The name of the reference column.
        """
        try:
            logging.info(
                f"Moving column {col_to_move} to be immediately after {ref_col}...")
            cols = self.df.columns.tolist()
            cols.insert(cols.index(ref_col) + 1,
                        cols.pop(cols.index(col_to_move)))
            self.df = self.df[cols]
        except Exception as e:
            logging.error(f"Error moving column: {e}")
            raise

    def rearrange_columns(self, cols_order):
        """
        Rearranges the columns of the DataFrame according to the specified order.

        Parameters
        ----------
        cols_order : list of str
            The desired order of the columns.
        """
        try:
            logging.info(f"Rearranging columns to {cols_order}...")
            self.df = self.df[cols_order]
        except Exception as e:
            logging.error(f"Error rearranging columns: {e}")
            raise

    def calculate_means(self, column_pairs, new_columns):
        """
        Calculates the means of pairs of columns and adds the results as new columns.

        Parameters
        ----------
        column_pairs : list of list of str
            The pairs of column names to calculate the means of.
        new_columns : list of str
            The names of the new columns to add.
        """
        try:
            for pair, new_col in zip(column_pairs, new_columns):
                logging.info(
                    f"Calculating mean of columns {pair} and adding as {new_col}...")
                self.df[new_col] = self.df[pair].mean(axis=1)
        except Exception as e:
            logging.error(f"Error calculating means: {e}")
            raise

    def add_labels(self, condition_columns, new_column):
        """
        Adds a new column based on conditions of existing columns.

        Parameters
        ----------
        condition_columns : list of str
            The names of the columns to use as conditions.
        new_column : str
            The name of the new column to add.
        """
        try:
            logging.info(
                f"Adding new column {new_column} based on conditions of columns {condition_columns}...")
            self.df[new_column] = ((self.df[condition_columns[0]] == 1) & (
                self.df[condition_columns[1]].shift(1) == 0)).astype(int)
        except Exception as e:
            logging.error(f"Error adding labels: {e}")
            raise

    def handle_infinity_and_na(self):
        """
        Replaces infinite and NaN values in the DataFrame with NaN.
        """
        try:
            logging.info("Handling infinite and NaN values...")
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        except Exception as e:
            logging.error(f"Error handling infinite and NaN values: {e}")
            raise
