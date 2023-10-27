import hashlib
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from data_preprocess.feature_engineering import FeatureEngineer
from data_preprocess.preprocessing import DataPreprocessor
from data_preprocess.rnn_data_prep import RNNDataPrep

from models.rnn_model import train_rnn_model
from models.rnn_model import save_model_and_config

from utils.utilities import (create_config_dict, get_hash)



def main():
    """
    Main function that performs data preprocessing, feature engineering, model training, and model saving.

    Returns
    -------
    None
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    pickle_path = "data/interim/ff-mw.pkl"
    rnn_data_prep = RNNDataPrep()
    X_train, Y_train, X_test, Y_test = rnn_data_prep.get_rnn_data(
        load_train_test=True, sequence_length=5, split_ratio=2/3)

    # # Initialize preprocessing object and load data
    # pickle_path = "data/interim/ff-mw.pkl"
    # preprocessor = DataPreprocessor(pickle_path=pickle_path)
    # logging.info("Loading data...")
    # df = preprocessor.load_data()

    # # Perform preprocessing steps
    # logging.info("Performing preprocessing steps...")
    # df = preprocessor.preprocess_data(df)

    # # Perform feature engineering steps
    # logging.info("Performing feature engineering steps...")
    # feature_engineer = FeatureEngineer(df)
    # df = feature_engineer.engineer_features(df)

    # # Save the processed data
    # logging.info("Saving processed data...")
    # raw_data_id = "ff-mw"
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # processed_data_path = preprocessor.save_processed_data(
    #     raw_data_id, timestamp)  # Save the processed data to a file

    # # Prepare sequences and train-test splits
    # logging.info("Preparing sequences and train-test splits...")
    # X_train, Y_train, X_test, Y_test = prep_train_test_seqs(df)

    # # Save the train-test splits
    # logging.info("Saving train-test splits...")
    # save_train_test_data(X_train, Y_train, X_test, Y_test)

    # Train the RNN model
    print(f"Training RNN Model...\n===============================\n")
    input_size = X_train.shape[2] - 1  # -1 because we drop the target column
    hidden_size = 64
    output_size = 2
    num_epochs = 10
    batch_size = 512
    learning_rate = 0.001
    batch_first = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_rnn_model(X_train, Y_train, X_test, Y_test, input_size,
                        hidden_size, output_size, num_epochs, batch_size, learning_rate, device, batch_first=batch_first)

    # Create the model name
    model_architecture = "rnn"
    raw_data_id = rnn_data_prep.raw_data_id
    version_number = 1
    model_name = f"{model_architecture}_{raw_data_id}_v{version_number}"

    rnn_timestamp = model.timestamp
    interim_data_path = rnn_data_prep.interim_data_path
    processed_data_path = rnn_data_prep.processed_data_path
    # Create the configuration dictionary
    config = create_config_dict(
        model_name=f"{rnn_timestamp}_{model_name}",
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        raw_data_path=None,
        interim_data_path=interim_data_path,
        processed_data_path=processed_data_path,
        logging_level='DEBUG',
        logging_format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )  # Create a dictionary with configuration settings

    # Save the trained model and configuration settings
    model_dir = Path(f"models/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    config_dir = Path(f"config/{model_name}")
    config_dir.mkdir(parents=True, exist_ok=True)

    save_model_and_config(model, model_name, rnn_timestamp, pickle_path,
                          processed_data_path, config, model_dir, config_dir)


if __name__ == "__main__":
    main()
