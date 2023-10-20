import logging

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    A class for performing feature engineering on a Pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to perform feature engineering on.

    Attributes
    ----------
    df : pandas.DataFrame
        The DataFrame to perform feature engineering on.
    scaler : sklearn.preprocessing.StandardScaler
        The scaler object used to standardize the features.

    Methods
    -------
    standardize_features(columns_to_scale)
        Standardizes the specified columns in the DataFrame.
    """

    def __init__(self, df=None):
        """
        Initializes a new instance of the FeatureEngineer class.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            The DataFrame to perform feature engineering on.
        """
        if df is None:
            raise ValueError("DataFrame cannot be None.")
        self.df = df
        self.scaler = StandardScaler()

    def standardize_features(self, columns_to_scale):
        """
        Standardizes the specified columns in the DataFrame.

        Parameters
        ----------
        columns_to_scale : list of str
            The names of the columns to standardize.
        """
        if not isinstance(columns_to_scale, list):
            raise TypeError("columns_to_scale must be a list.")
        if not all(isinstance(col, str) for col in columns_to_scale):
            raise TypeError(
                "All elements of columns_to_scale must be strings.")
        if not all(col in self.df.columns for col in columns_to_scale):
            raise ValueError(
                "All elements of columns_to_scale must be columns in the DataFrame.")
        try:
            logging.info("Standardizing features...")
            # sklearn's StandardScaler().fit_transform works by first
            # calculating the mean and standard deviation of each column
            # and then using those values to standardize the column.
            # Specifically, it uses the mean to center the data around
            # zero and the standard deviation to scale the data to unit
            # variance.
            self.df[columns_to_scale] = self.scaler.fit_transform(
                self.df[columns_to_scale])
        except Exception as e:
            logging.error(
                f"An error occurred while standardizing features: {e}")
            raise e
