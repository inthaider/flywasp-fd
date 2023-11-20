import hashlib
import logging
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from config import (
    PROCESSED_DATA_DIR,
    RNN_INPUT_DATA_DIR,
    MODELS_DIR,
    CONFIGS_DIR,
)

logger = logging.getLogger(__name__)


class DataSaver:
    """
    Class for saving processed data/model & config to files.

    Attributes:
        saved_paths (Dict[str, Path]): A dictionary containing the
            Pathlib paths to the saved files.
            The keys optionally are:
                - "processed_data": The path to the processed data file.
                - "train_test_data": The path to the train/test data
                    directory.
                - "model": The path to the model file.
                - "config": The path to the configuration file.

    Methods:
        save_processed_data(
            df: pd.DataFrame, dir: str | Path, data_id: str
        ) -> Path:
            Saves the processed data to a pickle file in
            <dir>/<data_id>/.
        save_train_test_data(
            X_train, Y_train, X_test, Y_test, dir: str
        ) -> str:
            Saves the train/test splits to pickle files in
            <dir>/<timestamp>/.
        save_model_and_config(
            model, model_name, timestamp, model_dir, config, config_dir
        ):
            Saves the trained model and configuration settings to files
            in <models_dir>/<model_name>/ and
            <configs_dir>/<model_name>/.
    """

    def __init__(self):
        # self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.saved_paths = {}

    def save_processed_data(
        self,
        df: pd.DataFrame,
        dir: str | Path = PROCESSED_DATA_DIR,
        data_id: str = "ff-mw",
    ) -> Path:
        """
        Saves the processed data to a pickle file in <dir>/<data_id>/.

        This method optionally needs to be given the parent directory
        path as <dir>, not the path to the lowest level directory. By
        default, it uses PROCESSED_DATA_DIR imported from
        config.py.

        It automatically creates/uses the lowest level directory
        appropriately and saves the file in the format:

            <dir>/<data_id>/<timestamp>_processed_data_<hash>.pkl

        Args:
            df (pd.DataFrame): The processed DataFrame.
            dir (str | Path): The directory to save the processed data.
                Defaults to PROCESSED_DATA_DIR == "data/processed/".
            data_id (str | Path): The ID of the processed data. Defaults
                to "ff-mw".

        Returns:
            Path: The path to the saved file.

        Raises:
            Exception: If there is an error saving the processed data.
        """
        processed_data_dir = Path(dir) / data_id
        logger.info(f"Saving processed datasets to {processed_data_dir}...\n")
        try:
            processed_data_dir.mkdir(parents=True, exist_ok=True)

            # Generate a hash based on DataFrame metadata and sampling
            logger.debug("Generating hash for data...")
            df_summary = (
                f"{df.shape}{df.columns}{df.sample(n=10, random_state=1)}"
            )
            processed_data_hash = hashlib.md5(df_summary.encode()).hexdigest()
            logger.debug("Data hashed.\n")

            # Construct the output file path
            timestamp = datetime.now().strftime("%Y%m%d")
            processed_data_path = (
                processed_data_dir
                / f"{timestamp}_processed_data_{processed_data_hash}.pkl"
            )

            # Check if file already exists
            if processed_data_path.exists():
                logger.warning(
                    f"File {processed_data_path} already exists. Appending a"
                    "unique identifier to the filename."
                )
                processed_data_path = (
                    processed_data_dir
                    / f"{timestamp}_processed_data_{processed_data_hash}_"
                    f"{uuid.uuid4()}.pkl"
                )

            # Save the processed data to the output file
            logger.debug("Saving now...")
            df.to_pickle(processed_data_path)
            self.saved_paths["processed_data"] = processed_data_path
            logger.info(f"Processed data saved to {processed_data_path}.\n")

            return processed_data_path
        except Exception as e:
            logger.error(f"\nError saving processed data: {e}\n")
            raise

    def save_train_test_data(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        test_indices: np.ndarray,
        dir: str | Path = RNN_INPUT_DATA_DIR,
    ) -> Path:
        """
        Saves the train/test splits to pickle files in
        <dir>/<timestamp>/.

        This method optionally needs to be given the parent directory
        path as <dir>, not the path to the lowest level directory. By
        default, it uses RNN_INPUT_DATA_DIR imported from
        config.py.

        It automatically creates/uses the lowest level directory
        appropriately and saves the files in the format:

            <dir>/<timestamp>/<X/Y_train/test>.pkl

        Args:
            X_train (np.ndarray): The training data features.
            Y_train (np.ndarray): The training data labels.
            X_test (np.ndarray): The test data features.
            Y_test (np.ndarray): The test data labels.
            test_indices (np.ndarray): The indices of the test data.
            dir (str | Path): The directory to save the train/test data.
                Defaults to RNN_INPUT_DATA_DIR == "data/rnn_input/".

        Returns:
            Path: The path to the saved directory.

        Raises:
            Exception: If there is an error saving the train/test data.
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        train_test_data_dir = Path(dir) / timestamp
        logger.info(f"Saving train/test splits to {dir}...\n")
        try:
            train_test_data_dir.mkdir(parents=True, exist_ok=True)

            for data, name in zip(
                [X_train, Y_train, X_test, Y_test, test_indices],
                ["X_train", "Y_train", "X_test", "Y_test", "test_indices"],
            ):
                file_path = train_test_data_dir / f"{name}.pkl"

                # Check if file already exists
                if file_path.exists():
                    logger.warning(
                        f"File {file_path} already exists. Appending a unique"
                        "identifier to the filename."
                    )
                    file_path = (
                        train_test_data_dir / f"{name}_{uuid.uuid4()}.pkl"
                    )

                with open(file_path, "wb") as f:
                    pickle.dump(data, f)

            logger.debug("Saving now...")
            self.saved_paths["train_test_data"] = train_test_data_dir
            logger.info(f"Train/test data saved to {train_test_data_dir}.\n")

            return train_test_data_dir
        except Exception as e:
            logger.error(f"\nError saving train and test data: {e}\n")
            raise

    def save_model_and_config(
        self,
        model: torch.nn.Module,
        model_name: str,
        config: Dict,
        models_dir: str | Path = MODELS_DIR,
        configs_dir: str | Path = CONFIGS_DIR,
    ) -> None | Tuple[Path, Path]:
        """
        Saves the trained model and configuration settings to files in
        <models_dir>/<model_name>/ and <configs_dir>/<model_name>/.

        This method optionally needs to be given the parent directory
        path as <models_dir> and <configs_dir>, not the path to the
        lowest level directory. By default, it uses MODELS_DIR and
        CONFIGS_DIR imported from config.py.

        It automatically creates/uses the lowest level directory
        appropriately and saves the files in the format:

            <models_dir>/<model_name>/<timestamp>_model_<hash>.pt
            <configs_dir>/<model_name>/<timestamp>_config_<hash>.yaml

        Args:
            model (torch.nn.Module): The trained RNN model.
            model_name (str): The name of the model.
            timestamp (str): The timestamp to use in output file names.
            config (Dict): The configuration settings for the model.
            models_dir (str | Path): Directory to save trained model.
                Defaults to MODELS_DIR == "models/".
            configs_dir (str | Path): Directory to save configuration.
                Defaults to CONFIGS_DIR == "config/".

        Returns:
            Tuple[Path, Path]: The paths to the saved model and
                configuration settings.

        Raises:
            Exception: If there is an error saving the model/config.
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        model_dir, config_dir = (
            Path(models_dir) / model_name,
            Path(configs_dir) / model_name,
        )
        logger.info(
            f"Saving model & config to {model_dir} & {config_dir}...\n"
        )
        try:
            # Generate a hash based on DataFrame metadata and sampling
            logger.debug("Generating hash for model & config...")
            model_hash = hashlib.md5(
                str(model.state_dict()).encode("utf-8")
            ).hexdigest()
            config_hash = hashlib.md5(str(config).encode("utf-8")).hexdigest()
            logger.debug("Data hashed.\n")

            model_file_name = f"{timestamp}_model_{model_hash}.pt"
            config_file_name = f"{timestamp}_config_{config_hash}.yaml"

            # Check if the model and configuration already exist
            existing_models = [f.name for f in model_dir.glob("*.pt")]
            existing_configs = [f.name for f in config_dir.glob("*.yaml")]
            if (
                model_file_name in existing_models
                and config_file_name in existing_configs
            ):
                logger.info(
                    "Model and configuration already exist. Skipping saving."
                )
                return
            else:
                model_dir.mkdir(parents=True, exist_ok=True)
                config_dir.mkdir(parents=True, exist_ok=True)
                model_path = Path(model_dir / model_file_name)
                config_path = Path(config_dir / config_file_name)

                logger.debug("Saving now...")
                torch.save(model.state_dict(), model_path)
                with open(config_path, "w") as f:
                    yaml.dump(config, f)
                self.saved_paths["model"] = model_path
                self.saved_paths["config"] = config_path
                logger.info(
                    f"Model & config saved to {model_dir} & {config_dir}.\n"
                )

                return model_path, config_path
        except Exception as e:
            logger.error(f"\nError saving model & config: {e}\n")
            raise
