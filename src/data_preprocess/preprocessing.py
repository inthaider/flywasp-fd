import hashlib
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


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
    save_processed_data()
        Saves the processed data to a pickled file.
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
    preprocess_data()
        Performs preprocessing steps on the DataFrame.
    """

    def __init__(self, df=None, pickle_path=None, raw_data_id: str = "ff-mw"):
        """
        Initializes a new instance of the DataPreprocessor class.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            The DataFrame to preprocess.
        pickle_path : str, optional
            The path to a pickled DataFrame to load.
        raw_data_id : str, optional
            The name/ID of the raw data.
        """
        self.df = df
        self.pickle_path = pickle_path
        self.raw_data_path = None
        self.interim_data_path = self.pickle_path
        self.processed_data_path = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.raw_data_id = raw_data_id
        
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

    def save_processed_data(self) -> str:
        """
        Saves the processed data to a pickled file.

        Returns
        -------
        str
            The path to the saved pickled file.
        """
        try:
            # Create the directory for the processed data if it doesn't exist
            processed_data_dir = Path(f"data/processed/{self.raw_data_id}")
            processed_data_dir.mkdir(parents=True, exist_ok=True)

            # Generate a hash based on DataFrame metadata and some sampling
            logging.info("Generating hash for processed data...")
            df_summary = f"{self.df.shape}{self.df.columns}{self.df.sample(n=10, random_state=1)}"
            processed_data_hash = hashlib.md5(df_summary.encode()).hexdigest()
            logging.info("Processed data hashed.")

            # Construct the output file path
            self.processed_data_path = processed_data_dir / \
                f"{self.timestamp}_processed_data_{processed_data_hash}.pkl"

            # Save the processed data to the output file
            logging.info(f"Saving processed data to {self.processed_data_path}...")
            self.df.to_pickle(self.processed_data_path)

            return str(self.processed_data_path)
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
        Replaces infinite and NaN values in the DataFrame with forward/backward filled values.
        """
        try:
            logging.info("Handling infinite and NaN values...")

            # Replace infinite values with NaN
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # # Forward fill NaN values
            # self.df.fillna(method='ffill', inplace=True)

            # # Backward fill any remaining NaN values
            # self.df.fillna(method='bfill', inplace=True)

        except Exception as e:
            logging.error(f"Error handling infinite and NaN values: {e}")
            raise

    def preprocess_data(self):
        """
        Performs preprocessing steps on the DataFrame.
        """
        self.drop_columns(["plot"])  # Drop the 'plot' column
        # Calculate the mean of 'ANTdis_1' and 'ANTdis_2' and store it in a new column 'ANTdis'
        self.calculate_means([["ANTdis_1", "ANTdis_2"]], ["ANTdis"])
        # Add a new column 'start_walk' with value 'walk_backwards' for rows where the 'walk_backwards' column has value 'walk_backwards'
        self.add_labels(["walk_backwards", "walk_backwards"], "start_walk")
        # Replace infinity and NaN values with appropriate values
        self.handle_infinity_and_na()
        self.specific_rearrange(
            "F2Wdis_rate", "F2Wdis"
        )  # Rearrange the column names
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
        )  # Rearrange the columns in a specific order

        # print the shape of the dataframe and its columns
        print(f"Shape of the dataframe: {self.df.shape}")
        print(f"Columns of the dataframe: {self.df.columns}")

        return self.df