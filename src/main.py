import logging
import numpy as np
import torch
from data_preprocess.preprocessing import DataPreprocessor
from data_preprocess.feature_engineering import FeatureEngineer
from utils.utilities import prepare_train_test_sequences
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

if __name__ == "__main__":
    main()