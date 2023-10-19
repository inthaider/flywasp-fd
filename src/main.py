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
import hashlib


def preprocess_data(df):
    """
    Performs preprocessing steps on the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame.
    """
    preprocessor = DataPreprocessor(df=df)
    preprocessor.drop_columns(["plot"])  # Drop the 'plot' column
    preprocessor.calculate_means([["ANTdis_1", "ANTdis_2"]], ["ANTdis"])  # Calculate the mean of 'ANTdis_1' and 'ANTdis_2' and store it in a new column 'ANTdis'
    preprocessor.add_labels(["walk_backwards", "walk_backwards"], "start_walk")  # Add a new column 'start_walk' with value 'walk_backwards' for rows where the 'walk_backwards' column has value 'walk_backwards'
    preprocessor.handle_infinity_and_na()  # Replace infinity and NaN values with appropriate values
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
    )  # Rearrange the columns in a specific order
    return preprocessor.df


def engineer_features(df):
    """
    Performs feature engineering steps on the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns
    -------
    pandas.DataFrame
        The feature-engineered DataFrame.
    """
    feature_engineer = FeatureEngineer(df=df)
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
    )  # Standardize the selected features
    return feature_engineer.df


def train_model(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate, device, batch_first=True):
    """
    Trains an RNN model on the input data.

    Parameters
    ----------
    X_train : numpy.ndarray
        The training input sequences.
    Y_train : numpy.ndarray
        The training target sequences.
    X_test : numpy.ndarray
        The test input sequences.
    Y_test : numpy.ndarray
        The test target sequences.
    input_size : int
        The size of the input features.
    hidden_size : int
        The size of the hidden layer.
    output_size : int
        The size of the output layer.
    num_epochs : int
        The number of training epochs.
    batch_size : int
        The batch size for training.
    learning_rate : float
        The learning rate for training.
    device : torch.device
        The device to use for training.
    batch_first : bool, optional
        Whether the input sequences have the batch dimension as the first dimension.

    Returns
    -------
    torch.nn.Module
        The trained RNN model.
    """
    model = train_rnn_model(X_train, Y_train, X_test, Y_test, input_size,
                            hidden_size, output_size, num_epochs, batch_size, learning_rate, device, batch_first=batch_first)  # Train the RNN model
    return model


def save_model_and_config(model, model_name, timestamp, pickle_path, processed_data_path, config, model_dir, config_dir):
    """
    Saves the trained model and configuration settings.

    Parameters
    ----------
    model : torch.nn.Module
        The trained RNN model.
    model_name : str
        The name of the model.
    timestamp : str
        The timestamp to use in the output file names.
    pickle_path : str
        The path to the input data pickle file.
    processed_data_path : str
        The path to the processed data pickle file.
    config : dict
        The configuration settings for the model.
    model_dir : pathlib.Path
        The directory to save the trained model.
    config_dir : pathlib.Path
        The directory to save the configuration settings.

    Returns
    -------
    None
    """
    # Get the hash values of the model and configuration
    model_hash = hashlib.md5(str(model.state_dict()).encode('utf-8')).hexdigest()
    config_hash = hashlib.md5(str(config).encode('utf-8')).hexdigest()

    # Check if the model and configuration already exist
    existing_models = [f.name for f in model_dir.glob("*.pt")]
    existing_configs = [f.name for f in config_dir.glob("*.yaml")]
    if f"rnn_model_{model_hash}.pt" in existing_models and f"config_{config_hash}.yaml" in existing_configs:
        logging.info("Model and configuration already exist. Skipping saving.")
    else:
        # Save the trained model
        model_path = model_dir / \
            f"{timestamp}_model_{model_hash}_{config_hash}.pt"
        torch.save(model.state_dict(), model_path)

        # Save the configuration settings
        config_path = config_dir / f"{timestamp}_config_{config_hash}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)


def main():
    """
    Main function that performs data preprocessing, feature engineering, model training, and model saving.

    Returns
    -------
    None
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize preprocessing object and load data
    pickle_path = "data/interim/ff-mw.pkl"
    preprocessor = DataPreprocessor(pickle_path=pickle_path)
    logging.info("Loading data...")
    df = preprocessor.load_data()

    # Perform preprocessing steps
    logging.info("Performing preprocessing steps...")
    df = preprocess_data(df)

    # Perform feature engineering steps
    logging.info("Performing feature engineering steps...")
    df = engineer_features(df)

    # Save the processed data
    logging.info("Saving processed data...")
    input_data = "ff-mw"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    processed_data_path = preprocessor.save_processed_data(input_data, timestamp)  # Save the processed data to a file

    # Prepare sequences and train-test splits
    logging.info("Preparing sequences and train-test splits...")
    X_train, Y_train, X_test, Y_test = prepare_train_test_sequences(df)

    # Train the RNN model
    logging.info("Training RNN model...")
    input_size = X_train.shape[2] - 1  # -1 because we drop the target column
    hidden_size = 64
    output_size = 2
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    batch_first = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(X_train, Y_train, X_test, Y_test, input_size,
                        hidden_size, output_size, num_epochs, batch_size, learning_rate, device, batch_first=batch_first)

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
    )  # Create a dictionary with configuration settings

    # Save the trained model and configuration settings
    model_dir = Path(f"models/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    config_dir = Path(f"config/{model_name}")
    config_dir.mkdir(parents=True, exist_ok=True)

    save_model_and_config(model, model_name, timestamp, pickle_path, processed_data_path, config, model_dir, config_dir)


if __name__ == "__main__":
    main()