import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class for loading copies of raw/processed/train-test data from file.

    In all the methods, the loaded data is copied to the data dict to
    avoid modifying the original data. Therefore, one can access the
    original loaded data from the data dict for the DataLoader object.

    Attributes:
        data (Dict[str, Any]): A dictionary containing the data.
            The keys are:
                - "paths": A dictionary containing the Pathlib paths to
                    the raw/processed/train-test data files.
                - "raw": The raw DataFrame.
                - "processed": The processed DataFrame.
                - "train_test": A dictionary containing the 4
                    train/test datasets.

    Methods:
        load_raw_data(pickle_path: str | Path) -> pd.DataFrame:
            Loads the raw DataFrame from a pickled file.
        load_processed_data(pickle_path: str | Path) -> pd.DataFrame:
            Loads the processed DataFrame from a pickled file.
        load_train_test_data(data_dir: str | Path) -> Dict[str, np.ndarray]:
            Loads 4 train/test datasets for the RNN from .pkl files.
    """

    def __init__(self):
        self.data = {
            "paths": {},
            "raw": None,
            "processed": None,
            "train_test": {},
        }

    def _load_data(
        self,
        pickle_path: str | Path,
        data_key: str,
    ) -> pd.DataFrame:
        """
        Loads & returns a copy of the DataFrame from a pickled file.

        Args:
            pickle_path (str | Path): The path to the pickled file.
            data_key (str): Key for the data in the data dict.

        Returns:
            pd.DataFrame: A copy of the loaded DataFrame.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
            ValueError: If an invalid pickle path is provided.
            Exception: For other unexpected errors.
        """
        path = Path(pickle_path)
        logger.debug(f"Loading {data_key} data from {path}...\n")

        if not path.exists():
            raise FileNotFoundError(f"No file found at {path}")
        self.data["paths"][data_key] = path

        try:
            df = pd.read_pickle(path)
            self.data[data_key] = df.copy()  # Make copy of the df
            logger.debug(f"Successfully loaded {data_key} data from {path}.\n")
            return df
        except ValueError as e:
            logger.error(f"ValueError loading {data_key} data: {e}\n")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {data_key} data: {e}\n")
            raise

    def load_raw_data(self, pickle_path: str | Path) -> pd.DataFrame:
        """
        Loads & returns a *copy* of the raw DataFrame. Uses internal
        _load_data() method.

        Args:
            pickle_path (str | Path): The path to the pickled file.

        Returns:
            pd.DataFrame: A copy of the raw DataFrame.
        """
        return self._load_data(pickle_path, "raw")

    def load_processed_data(self, pickle_path: str | Path) -> pd.DataFrame:
        """
        Loads & returns a *copy* of the processed DataFrame. Uses
        internal _load_data() method.

        Args:
            pickle_path (str | Path): The path to the pickled file.

        Returns:
            pd.DataFrame: A copy of the processed DataFrame.
        """
        return self._load_data(pickle_path, "processed")

    def load_train_test_data(
        self, data_dir: str | Path
    ) -> Dict[str, np.ndarray]:
        """
        Loads & returns *copies* of x4 train/test data splits for the
        RNN from .pkl files.

        Args:
            data_dir (str | Path): The path to the directory containing
                the 4 .pkl files.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing copies of the
                4 train/test datasets.

        Raises:
            FileNotFoundError: If the directory/any file doesn't exist.
            ValueError: If an invalid directory is provided.
            Exception: For other unexpected errors.
        """
        dir_path = Path(data_dir)
        logger.info(f"Loading train/test datasets from {dir_path}...\n")
        if not dir_path.exists():
            raise FileNotFoundError(f"No directory found at {dir_path}")
        self.data["paths"]["train_test"] = dir_path

        splits = [
            "X_train",
            "Y_train",
            "X_test",
            "Y_test",
        ]
        train_test_dict = {}
        try:
            for s in splits:
                file_path = dir_path / f"{s}.pkl"
                if not file_path.exists():
                    raise FileNotFoundError(f"No file found at {file_path}\n")

                with open(file_path, "rb") as file:
                    train_test_dict[s] = pickle.load(file)  # Load

            self.data["train_test"] = train_test_dict.copy()  # Copy
            logger.info(
                f"Successfully loaded train/test datasets from {dir_path}.\n"
            )
            return train_test_dict
        except ValueError as e:
            logger.error(f"ValueError loading train/test data: {e}\n")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading train/test data: {e}\n")
            raise
