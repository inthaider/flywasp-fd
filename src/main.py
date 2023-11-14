import logging
from pathlib import Path

import numpy as np
import torch

from src.data_preprocess.rnn_data_prep import RNNDataPrep
from src.models.train_eval import train_eval_model
from src.visualization.rnn_visualize import plot_predicted_probabilities
from src.utils.utilities import create_config_dict

# from src.data_preprocess.feature_engineering import FeatureEngineer
# from src.data_preprocess.preprocessing import DataPreprocessor from
# src.models.helpers_rnn import save_model_and_config

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    # ============= Data preprocessing & train/test prep ============= #
    # pickle_path = "data/interim/ff-mw.pkl"
    rnn_data_prep = RNNDataPrep()
    X_train, Y_train, X_test, Y_test = rnn_data_prep.get_rnn_data(
        load_train_test=True, sequence_length=5, split_ratio=2 / 3
    )

    # Print shapes
    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("Y_test shape: ", Y_test.shape)

    # Check for data imbalance in Y_train and Y_test Note that the
    # single feature in Y data is a binary classification 0: no walk 1:
    # walk
    logging.info("Checking for data imbalance...")
    logging.info(f"Y_train: {np.unique(Y_train, return_counts=True)}")
    logging.info(f"Y_test: {np.unique(Y_test, return_counts=True)}")

    # ====================== Train the RNN model ===================== #
    print("Training RNN Model...\n===============================\n")
    input_size = X_train.shape[2]

    # print(f"Input size: {input_size}\n\n")
    hidden_size = 64
    output_size = 2
    num_epochs = 10
    batch_size = 512
    learning_rate = 0.01
    batch_first = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, test_labels_and_probs = train_eval_model(
        X_train,
        Y_train,
        X_test,
        Y_test,
        input_size,
        hidden_size,
        output_size,
        num_epochs,
        batch_size,
        learning_rate,
        device,
        batch_first=batch_first,
    )

    # ========== Model evaluation, visualization, & analysis ========= #
    test_indices = rnn_data_prep.test_indices
    df = rnn_data_prep.df
    # print(test_indices)
    print(f"Test indices shape: {test_indices.shape}")

    print(
        f"test_true_labels shape: {test_labels_and_probs[0].shape}, \n"
        f"test_pred_labels shape: {test_labels_and_probs[1].shape}, \n"
        f"test_pred_probs shape: {test_labels_and_probs[2].shape}\n",
    )
    print(f"df shape: {df.shape}")

    plot_df, mean_df = plot_predicted_probabilities(
        df, test_indices, test_labels_and_probs
    )

    # ====================== Save model & config ===================== #
    # Create the model name
    model_architecture = "rnn"
    # get the raw data id, in this case 'ff-mw'
    raw_data_id = rnn_data_prep.raw_data_id
    version_number = 1
    model_name = f"{model_architecture}_{raw_data_id}_v{version_number}"

    # Define/get config details
    rnn_timestamp = model.timestamp
    interim_data_path = rnn_data_prep.interim_data_path
    processed_data_path = rnn_data_prep.processed_data_path

    # Create the configuration dictionary
    config = create_config_dict(
        model_name=model_name,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        raw_data_path=None,
        interim_data_path=interim_data_path,
        processed_data_path=processed_data_path,
        logging_level="DEBUG",
        logging_format=(
            "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
        ),
    )

    # Save the trained model and configuration settings
    model_dir = Path(f"models/{model_name}")
    config_dir = Path(f"config/{model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    model.save_model_and_config(
        model,
        model_name,
        rnn_timestamp,
        interim_data_path,
        processed_data_path,
        config,
        model_dir,
        config_dir,
    )


if __name__ == "__main__":
    main()


# # Initialize preprocessing object and load data
# pickle_path = "data/interim/ff-mw.pkl" preprocessor =
# DataPreprocessor(pickle_path=pickle_path) logging.info("Loading
# data...") df = preprocessor.load_data()

# # Perform preprocessing steps
# logging.info("Performing preprocessing steps...") df =
# preprocessor.preprocess_data(df)

# # Perform feature engineering steps
# logging.info("Performing feature engineering steps...")
# feature_engineer = FeatureEngineer(df) df =
# feature_engineer.engineer_features(df)

# # Save the processed data
# logging.info("Saving processed data...") raw_data_id = "ff-mw"
# timestamp = datetime.now().strftime("%Y%m%d_%H%M") processed_data_path
# = preprocessor.save_processed_data( raw_data_id, timestamp)  # Save
# the processed data to a file

# # Prepare sequences and train-test splits
# logging.info("Preparing sequences and train-test splits...") X_train,
# Y_train, X_test, Y_test = prep_train_test_seqs(df)

# # Save the train-test splits
# logging.info("Saving train-test splits...")
# save_train_test_data(X_train, Y_train, X_test, Y_test)
