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
    raw_data_id : str, optional
        The name/ID of the raw data.
    save_data : bool, optional
        Whether to save the processed data to a pickled file. Defaults to False.

    Attributes
    ----------
    df : pandas.DataFrame
        The DataFrame to preprocess. Defaults to None.
    pickle_path : str
        The path to a pickled DataFrame to load. Defaults to None.
    raw_data_id : str
        The name/ID of the raw data. Defaults to "ff-mw".
    save_data : bool
        Whether to save the processed data to a pickled file. Defaults to False.
    raw_data_path : str
        The path to the raw data. Defaults to None.
    interim_data_path : str
        The path to the interim data. Defaults to pickle_path.
    processed_data_path : str
        The path to the processed data. Defaults to None.
    timestamp : str
        The timestamp of the current time in the format YYYYMMDD_HHMM.

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

    def __init__(self, df=None, pickle_path=None, raw_data_id: str = "ff-mw", save_data: bool = False):
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
        save_data : bool, optional
            Whether to save the processed data to a pickled file. Defaults to False.
        """
        self.df = df
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

        Returns
        -------
        pandas.DataFrame
            The loaded DataFrame.
        """
        # Logging statement to indicate the start of loading data
        logging.info("\nLoading data in DataPreprocessor -> load_data() ...")
        try:
            if self.pickle_path:
                logging.info(f"Loading data from {self.pickle_path}...")
                self.df = pd.read_pickle(self.pickle_path)
                logging.info(f"Data loaded from {self.pickle_path}.")
                return self.df
            else:
                raise ValueError(
                    "\nVALUE ERROR: Provide a valid pickle path.\n")
        except Exception as e:
            logging.error(f"\nERROR loading data: {e}\n")
            raise

    def save_processed_data(self) -> str:
        """
        Saves the processed data to a pickled file.

        Returns
        -------
        str
            The path to the saved pickled file.
        """
        # Logging statement to indicate the start of saving processed data
        logging.info(
            "\nSaving processed data in DataPreprocessor -> save_processed_data() ...")
        try:
            # Create the directory for the processed data if it doesn't exist
            processed_data_dir = Path(f"data/processed/{self.raw_data_id}")
            processed_data_dir.mkdir(parents=True, exist_ok=True)

            #
            # Generate a hash based on DataFrame metadata and some sampling
            #
            logging.info("Generating hash for processed data...")
            # Extracting the shape, columns and a sample of the dataframe
            df_summary = f"{self.df.shape}{self.df.columns}{self.df.sample(n=10, random_state=1)}"
            # Generating the hash using the md5 algorithm based on the dataframe summary
            processed_data_hash = hashlib.md5(df_summary.encode()).hexdigest()
            logging.info("Processed data hashed.")

            # Construct the output file path
            self.processed_data_path = processed_data_dir / \
                f"{self.timestamp}_processed_data_{processed_data_hash}.pkl"

            # Save the processed data to the output file
            logging.info(
                f"Saving processed data to {self.processed_data_path}...")
            self.df.to_pickle(self.processed_data_path)
            logging.info(
                f"Processed data saved to {self.processed_data_path}.\n")
            return str(self.processed_data_path)
        except Exception as e:
            logging.error(f"\nERROR saving processed data: {e}\n")
            raise

    def drop_columns(self, columns_to_drop):
        """
        Drops the specified columns from the DataFrame.

        Parameters
        ----------
        columns_to_drop : list of str
            The names of the columns to drop.
        """
        # Logging statement to indicate the start of dropping columns
        logging.info(
            f"\nDropping columns {columns_to_drop} in DataPreprocessor -> drop_columns() ...")
        try:
            self.df.drop(columns_to_drop, axis=1, inplace=True)
            # Logging statement to indicate the end of dropping columns
            logging.info(
                f"Columns dropped successfully.\n")
        except Exception as e:
            logging.error(f"\nERROR dropping columns: {e}\n")
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
        # Logging statement to indicate the start of moving a column
        logging.info(
            f"\nRearranging specific columns in DataPreprocessor -> specific_rearrange() ...")
        try:
            logging.info(
                f"Moving column {col_to_move} to be immediately after {ref_col}...")
            cols = self.df.columns.tolist()
            cols.insert(cols.index(ref_col) + 1,
                        cols.pop(cols.index(col_to_move)))
            self.df = self.df[cols]
            # Logging statement to indicate the end of moving a column
            logging.info(
                f"Column moved successfully.\n")
        except Exception as e:
            logging.error(f"\nERROR moving column: {e}\n")
            raise

    def rearrange_columns(self, cols_order):
        """
        Rearranges the columns of the DataFrame according to the specified order.

        Parameters
        ----------
        cols_order : list of str
            The desired order of the columns.
        """
        # Logging statement to indicate the start of rearranging columns
        logging.info(
            f"\nRearranging columns in DataPreprocessor -> rearrange_columns() ...")
        try:
            logging.info(f"Rearranging columns to {cols_order}...")
            self.df = self.df[cols_order]
            # Logging statement to indicate the end of rearranging columns
            logging.info(
                f"Columns rearranged successfully.\n")
        except Exception as e:
            logging.error(f"\nERROR rearranging columns: {e}\n")
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
        # Logging statement to indicate the start of calculating means
        logging.info(
            f"\nCalculating means in DataPreprocessor -> calculate_means() ...")
        try:
            for pair, new_col in zip(column_pairs, new_columns):
                logging.info(
                    f"Calculating mean of columns {pair} and adding as {new_col}...")
                self.df[new_col] = self.df[pair].mean(axis=1)

            # Logging statement to indicate the end of calculating means
            logging.info(
                f"Means calculated successfully.\n")
        except Exception as e:
            logging.error(f"\nERROR calculating means: {e}\n")
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
        # Logging statement to indicate the start of adding labels
        logging.info(
            f"\nAdding labels in DataPreprocessor -> add_labels() ...")
        try:
            logging.info(
                f"Adding new column {new_column} based on conditions of columns {condition_columns}...")
            self.df[new_column] = ((self.df[condition_columns[0]] == 1) & (
                self.df[condition_columns[1]].shift(1) == 0)).astype(int)

            # Logging statement to indicate the end of adding labels
            logging.info(
                f"Labels added successfully.\n")
        except Exception as e:
            logging.error(f"\nERROR adding labels: {e}\n")
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
            logging.error(f"\nERROR handling infinite and NaN values: {e}\n")
            raise

    def preprocess_data(self, save_data: bool = None):
        """
        Performs preprocessing steps on the DataFrame.

        Parameters
        ----------
        save_data : bool, optional
            Whether to save the processed data to a pickled file. Defaults to None, in which case the value of self.save_data is used.

        Returns
        -------
        pandas.DataFrame
            The preprocessed DataFrame.
        """
        # Add logging statements to indicate the start of preprocessing as part of the DataPreprocessor class
        logging.info("\n\nPreprocessing data (in DataPreprocessor class)...")

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
        self.specific_rearrange(
            "F2Wdis_rate", "F2Wdis"
        )
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
            logging.info(
                "!! save_data is True !!")
            self.save_processed_data()

        # Print the shape of the dataframe and its columns using the print module.
        print(
            f"\nDataPreprocessor.preprocess_data --> Shape of the dataframe: {self.df.shape}")
        print(
            f"DataPreprocessor.preprocess_data --> Columns of the dataframe: {self.df.columns}\n")

        # Add logging statements to indicate the end of preprocessing as part of the DataPreprocessor class
        logging.info("Preprocessing complete (in DataPreprocessor class).\n\n")
        return self.df
