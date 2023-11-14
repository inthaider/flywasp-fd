import logging
from pathlib import Path
import hashlib

from datetime import datetime
import pickle
from typing import Dict
import uuid
import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)


class DataSaver:
    def __init__(self):
        # self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.saved_paths = {}

    def save_processed_data(
        self, df: pd.DataFrame, dir: str | Path, data_id: str, timestamp: str
    ) -> Path:
        dir = Path(dir)
        logger.info(f"Saving processed datasets to {dir}...\n")
        try:
            processed_data_dir = dir / f"processed/{data_id}"
            processed_data_dir.mkdir(parents=True, exist_ok=True)

            # Generate a hash based on DataFrame metadata and sampling
            logger.debug("Generating hash for data...")
            df_summary = (
                f"{df.shape}{df.columns}{df.sample(n=10, random_state=1)}"
            )
            processed_data_hash = hashlib.md5(df_summary.encode()).hexdigest()
            logger.debug("Data hashed.\n")

            # Construct the output file path
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
        self, X_train, Y_train, X_test, Y_test, dir: str, timestamp: str
    ) -> str:
        logger.info(f"Saving train/test splits to {dir}...\n")
        try:
            train_test_data_dir = Path(dir) / f"{timestamp}"
            train_test_data_dir.mkdir(parents=True, exist_ok=True)

            for data, name in zip(
                [X_train, Y_train, X_test, Y_test],
                ["X_train", "Y_train", "X_test", "Y_test"],
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

            return str(train_test_data_dir)
        except Exception as e:
            logger.error(f"\nError saving train and test data: {e}\n")
            raise

    def save_model_and_config(
        self, model, model_name, timestamp, model_dir, config, config_dir
    ):
        try:
            model_dir, config_dir = Path(model_dir), Path(config_dir)
            logger.info(
                f"Saving model & config to {model_dir} & {config_dir}...\n"
            )

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
        except Exception as e:
            logger.error(f"\nError saving model & config: {e}\n")
            raise
