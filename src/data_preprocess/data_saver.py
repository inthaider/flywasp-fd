import logging
from pathlib import Path
import hashlib

from datetime import datetime
import pickle
from typing import Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataSaver:
    def __init__(self):
        # self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.saved_paths = {}

    def save_processed_data(
        self, df: pd.DataFrame, directory: str, data_id: str, timestamp: str
    ) -> str:
        processed_data_dir = Path(directory) / f"processed/{data_id}"
        processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Generate a hash based on DataFrame metadata and some sampling
        df_summary = f"{df.shape}{df.columns}{df.sample(n=10, random_state=1)}"
        processed_data_hash = hashlib.md5(df_summary.encode()).hexdigest()

        # Construct the output file path
        processed_data_path = (
            processed_data_dir
            / f"{timestamp}_processed_data_{processed_data_hash}.pkl"
        )

        # Save the processed data to the output file
        df.to_pickle(processed_data_path)
        self.saved_paths["processed_data"] = processed_data_path
        return str(processed_data_path)

    def save_train_test_data(
        self, X_train, Y_train, X_test, Y_test, directory: str, timestamp: str
    ) -> str:
        train_test_data_dir = Path(directory) / f"{timestamp}"
        train_test_data_dir.mkdir(parents=True, exist_ok=True)

        for data, name in zip(
            [X_train, Y_train, X_test, Y_test],
            ["X_train", "Y_train", "X_test", "Y_test"],
        ):
            file_path = train_test_data_dir / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(data, f)

        self.saved_paths["train_test_data"] = train_test_data_dir
        return str(train_test_data_dir)

    def save_model_and_config(
        self, model, model_name, timestamp, model_dir, config, config_dir
    ):
        model_hash = hashlib.md5(
            str(model.state_dict()).encode("utf-8")
        ).hexdigest()
        config_hash = hashlib.md5(str(config).encode("utf-8")).hexdigest()

        model_path = (
            model_dir / f"{timestamp}_model_{model_hash}_{model_name}.pt"
        )
        config_path = config_dir / f"{timestamp}_config_{config_hash}.yaml"

        torch.save(model.state_dict(), model_path)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        self.saved_paths["model"] = model_path
        self.saved_paths["config"] = config_path


def save_processed_data(self) -> str:
    # Logging statement to indicate the start of saving processed
    # data
    logger.debug(
        "\nSaving processed data in DataPreprocessor -> "
        "save_processed_data() ..."
    )
    try:
        # Create the directory for the processed data if it doesn't
        # exist
        processed_data_dir = Path(f"data/processed/{self.raw_data_id}")
        processed_data_dir.mkdir(parents=True, exist_ok=True)

        #
        # Generate a hash based on DataFrame metadata and some
        # sampling
        #
        logger.debug("Generating hash for processed data...")
        # Extracting the shape, columns and a sample of the
        # dataframe
        df_summary = (
            f"{self.df.shape}"
            f"{self.df.columns}"
            f"{self.df.sample(n=10, random_state=1)}"
        )
        # Generating the hash using the md5 algorithm based on the
        # dataframe summary
        processed_data_hash = hashlib.md5(df_summary.encode()).hexdigest()
        logger.debug("Processed data hashed.")

        # Construct the output file path
        self.processed_data_path = (
            processed_data_dir
            / f"{self.timestamp}_processed_data_{processed_data_hash}.pkl"
        )

        # Save the processed data to the output file
        logger.debug(f"Saving processed data to {self.processed_data_path}...")
        self.df.to_pickle(self.processed_data_path)
        logger.debug(f"Processed data saved to {self.processed_data_path}.\n")
        return str(self.processed_data_path)
    except Exception as e:
        logger.error(f"\nERROR saving processed data: {e}\n")
        raise


def _save_train_test_data(
    self,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> str:
    # Logging messages to indicate the start of the function
    logger.info("\nSaving train and test datasets...")
    try:
        # Create a timestamped directory for the processed data
        self.train_test_data_dir = Path(
            f"{self.train_test_data_par_dir}/{self.timestamp}"
        )
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
        logger.info(
            "Successfully saved train and test datasets to "
            f"{self.train_test_data_dir}.\n"
        )
        return str(self.train_test_data_dir)
    except Exception as e:
        logger.error(f"\nERROR saving train and test datasets: {e}\n")
        raise


def save_model_and_config(
    model,
    model_name,
    timestamp,
    pickle_path,
    processed_data_path,
    config,
    model_dir,
    config_dir,
):
    # Get the hash values of the model and configuration
    model_hash = hashlib.md5(
        str(model.state_dict()).encode("utf-8")
    ).hexdigest()
    config_hash = hashlib.md5(str(config).encode("utf-8")).hexdigest()

    # Check if the model and configuration already exist
    existing_models = [f.name for f in model_dir.glob("*.pt")]
    existing_configs = [f.name for f in config_dir.glob("*.yaml")]
    if (
        f"rnn_model_{model_hash}.pt" in existing_models
        and f"config_{config_hash}.yaml" in existing_configs
    ):
        logger.info("Model and configuration already exist. Skipping saving.")
    else:
        # Save the trained model
        model_path = (
            model_dir / f"{timestamp}_model_{model_hash}_{config_hash}.pt"
        )
        torch.save(model.state_dict(), model_path)

        # Save the configuration settings
        config_path = config_dir / f"{timestamp}_config_{config_hash}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
