"""
This module contains the `FeatureEngineer` class for performing feature
engineering on a Pandas DataFrame.

The `FeatureEngineer` class includes methods for standardizing features
and performing other feature engineering steps. It uses the
`StandardScaler` class from the `sklearn.preprocessing` module to
standardize features.

Classes:
    FeatureEngineer:
        A class for performing feature engineering on a Pandas
        DataFrame. It includes methods for standardizing features and
        performing other feature engineering steps.

Example:
    To use the FeatureEngineer class to standardize features of a
    DataFrame:

    >>> df = pd.DataFrame(data)
    >>> feature_engineer = FeatureEngineer(df)
    >>> df_standardized = feature_engineer.engineer_features()
"""

import logging

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A class for performing feature engineering on a Pandas DataFrame.

    Attributes:
        df (pandas.DataFrame): The DataFrame to perform feature
            engineering on.
        scaler (sklearn.preprocessing.StandardScaler): The scaler object
            used to standardize the features.

    Methods:
        standardize_features(columns_to_scale): Standardizes the
            specified columns in the DataFrame.
        engineer_features(): Performs feature engineering steps on the
            DataFrame.
    """

    def __init__(self, df=None):
        """
        Initializes FeatureEngineer with the given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to perform feature
                engineering on.
        """
        if df is None:
            raise ValueError("DataFrame cannot be None.")
        self.df = df
        self.scaler = StandardScaler()

    def standardize_features(self, columns_to_scale):
        """
        Standardizes the specified columns in the DataFrame.

        Args:
            columns_to_scale (list): The columns to standardize.
        """
        if not isinstance(columns_to_scale, list):
            raise TypeError("columns_to_scale must be a list.")
        if not all(isinstance(col, str) for col in columns_to_scale):
            raise TypeError(
                "All elements of columns_to_scale must be strings."
            )
        if not all(col in self.df.columns for col in columns_to_scale):
            raise ValueError(
                "All elements of columns_to_scale must be columns in the "
                "DataFrame."
            )
        try:
            logger.info("Standardizing features...")
            # sklearn's StandardScaler().fit_transform works by first
            # calculating the mean and standard deviation of each column
            # and then using those values to standardize the column.
            # Specifically, it uses the mean to center the data around
            # zero and the standard deviation to scale the data to unit
            # variance.
            self.df[columns_to_scale] = self.scaler.fit_transform(
                self.df[columns_to_scale]
            )
        except Exception as e:
            logger.error(
                f"An error occurred while standardizing features: {e}"
            )
            raise e

    def engineer_features(self):
        """
        Performs feature engineering steps on the DataFrame.
        """
        # Logging the start of the feature engineering step
        logger.info("\n\nEngineering features (in FeatureEngineer class)...")

        # Standardize the selected features
        self.standardize_features(
            [
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
            ]
        )
        # Perform additional feature engineering steps here ...

        # Logging the end of the feature engineering step
        logger.info(
            "Finished engineering features (in FeatureEngineer class).\n\n"
        )
        return self.df
