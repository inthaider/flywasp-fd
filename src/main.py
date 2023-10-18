import logging
import numpy as np
import torch
import yaml
from datetime import datetime
from pathlib import Path
from data_preprocess.preprocessing import DataPreprocessor
from data_preprocess.feature_engineering import FeatureEngineer
from utils.utilities import prepare_train_test_sequences
from utils.utilities import create_config_dict
from utils.utilities import get_hash
from models.rnn_model import train_rnn_model

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize preprocessing object and load data
    pickle_path = "data/interim/ff-mw.pkl"
    preprocessor = DataPreprocessor(pickle_path=pickle_path)
    logging.info("Loading data...")
    df = preprocessor.load_data()

    # Perform preprocessing steps
    logging.info("Performing preprocessing steps...")
    preprocessor.drop_columns(["plot"])
    preprocessor.calculate_means([["ANTdis_1", "ANTdis_2"]], ["ANTdis"])
    preprocessor.add_labels(["walk_backwards", "walk_backwards"], "start_walk")
    preprocessor.handle_infinity_and_na()
    preprocessor.specific_rearrange(
        "F2Wdis_rate", "F2Wdis"
    )  # Rearrange the column names
    preprocessor.rearrange_columns(
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

    # Perform feature engineering steps
    logging.info("Performing feature engineering steps...")
    feature_engineer = FeatureEngineer(df=preprocessor.df)
    feature_engineer.standardize_features(
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

    # Save the processed data
    logging.info("Saving processed data...")
    input_data = "ff-mw"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    processed_data_path = preprocessor.save_processed_data(preprocessor.df, input_data, timestamp)

    # Prepare sequences and train-test splits
    logging.info("Preparing sequences and train-test splits...")
    X_train, Y_train, X_test, Y_test = prepare_train_test_sequences(
        feature_engineer.df)

    # Train the RNN model
    logging.info("Training RNN model...")
    input_size = X_train.shape[2] - 1
    hidden_size = 64
    output_size = 2
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_rnn_model(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate, device)

    # Create the model name
    model_architecture = "rnn"
    version_number = 1
    model_name = f"{model_architecture}_{input_data}_v{version_number}"
    
    # Create the configuration dictionary
    config = create_config_dict(
        model_name=f"{timestamp}_{model_name}",
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        raw_data_path=None,
        interim_data_path=pickle_path,
        processed_data_path=processed_data_path,
        logging_level='DEBUG',
        logging_format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )

    # Get the hash values of the model and configuration
    model_hash = get_hash(model.state_dict())
    config_hash = get_hash(config)
    
    # Check if the model and configuration already exist
    model_dir = Path(f"models/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    config_dir = Path(f"config/{model_name}")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    existing_models = [f.name for f in model_dir.glob("*.pt")]
    existing_configs = [f.name for f in config_dir.glob("*.yaml")]
    if f"rnn_model_{model_hash}.pt" in existing_models and f"config_{config_hash}.yaml" in existing_configs:
        logging.info("Model and configuration already exist. Skipping saving.")
    else:
        # Save the trained model
        model_path = model_dir / f"{timestamp}_model_{model_hash}_{config_hash}.pt"
        torch.save(model.state_dict(), model_path)

        # Save the configuration settings
        config_path = config_dir / f"{timestamp}_config_{config_hash}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

if __name__ == "__main__":
    main()