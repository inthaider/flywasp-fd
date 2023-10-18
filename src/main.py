from data_preprocess.preprocessing import DataPreprocessor
from data_preprocess.feature_engineering import FeatureEngineer
from utils.utilities import prepare_train_test_sequences


def main():
    # Initialize preprocessing object and load data
    pickle_path = "data/interim/ff-mw.pkl"
    preprocessor = DataPreprocessor(pickle_path=pickle_path)
    df = preprocessor.load_data()

    # Perform preprocessing steps
    preprocessor.drop_columns(["plot"])
    preprocessor.calculate_means([["ANTdis_1", "ANTdis_2"]], ["ANTdis"])
    preprocessor.add_labels(["walk_backwards", "walk_backwards"], "start_walk")
    preprocessor.handle_infinity_and_na()
    preprocessor.specific_rearrange(
        "F2Wdis_rate", "F2Wdis"
    )  # Rearrange the column names

    # Only keeping in the relevant variables
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
    X_train, Y_train, X_test, Y_test = prepare_train_test_sequences(
        feature_engineer.df)

    # TODO: Model building and evaluation code here


if __name__ == "__main__":
    main()
