"""
This module contains functions for visualizing the results of an RNN model.

Functions:
    plot_predicted_probabilities(df, test_indices, test_labels_and_probs)
        -> pd.DataFrame, pd.DataFrame: Plots the predicted probabilities using
            the _make_df_for_plotting(), _process_df_for_plotting(), and
            _calculate_and_plot_means() functions.
    _make_df_for_plotting(df, test_indices, test_true_labels, test_pred_probs)
        -> pd.DataFrame: Creates a DataFrame for plotting.
    _process_df_for_plotting(plot_df) -> pd.DataFrame: Processes the DataFrame
        for plotting.
    _flag_frames(df, frame_distance=200) -> pd.DataFrame: Flags frames that
        are within frame_distance of the start frame.
    _calculate_and_plot_means(plot_df) -> pd.DataFrame: Calculates and plots
        the means.
"""

import logging

import pandas as pd
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def plot_predicted_probabilities(df, test_indices, test_labels_and_probs):
    """
    Plot the predicted probabilities using the _make_df_for_plotting(),
    _process_df_for_plotting(), and _calculate_and_plot_means() functions.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        test_indices (numpy.ndarray): The indices of the test data.
        test_labels_and_probs (numpy.ndarray): A numpy array containing the
            true labels, predicted labels, and predicted probabilities.

    Returns:
        plot_df (pandas.DataFrame): The processed DataFrame for plotting.
        mean_df (pandas.DataFrame): The DataFrame containing the calculated
            means.
    """
    # Unpack the test_labels_and_probs numpy array
    test_true_labels = test_labels_and_probs[0]
    # test_pred_labels = test_labels_and_probs[1]
    test_pred_probs = test_labels_and_probs[2]

    # Create a DataFrame for plotting
    plot_df = _make_df_for_plotting(
        df, test_indices, test_true_labels, test_pred_probs
    )
    # Process the DataFrame for plotting
    plot_df = _process_df_for_plotting(plot_df)
    # Calculate and plot the means
    mean_df = _calculate_and_plot_means(plot_df)

    return plot_df, mean_df


def _make_df_for_plotting(df, test_indices, test_true_labels, test_pred_probs):
    """
    Creates a DataFrame for plotting the predicted probabilities.

    This function merges the test indices, true labels, and predicted
    probabilities with the original DataFrame.

    Args:
        df (pandas.DataFrame): The original DataFrame containing the data.
        test_indices (numpy.ndarray): The indices of the test data.
        test_true_labels (numpy.ndarray): The true labels of the test data.
        test_pred_probs (numpy.ndarray): The predicted probabilities of the
            test data.

    Returns:
        plot_df (pandas.DataFrame): The merged DataFrame for plotting.
    """
    merged_df = pd.DataFrame(
        {
            "index": test_indices,
            "y_test_true": test_true_labels,
            "y_test_pred_prob": test_pred_probs,
        }
    )
    plot_df = pd.merge(
        df, merged_df, left_index=True, right_on="index", how="right"
    )
    return plot_df


def _process_df_for_plotting(plot_df):
    """
    Processes a DataFrame for plotting the predicted probabilities.

    This function sorts the DataFrame by file and frame, flags frames, assigns
    a unique group number to each file group, and calculates the delta frames.

    Args:
        plot_df (pandas.DataFrame): The DataFrame to process.

    Returns:
        plot_df (pandas.DataFrame): The processed DataFrame for plotting.

    TODO:
        What exactly is plot_df_2 and what do we mean by "plot of predicted
        probabilities 1 second before and after a true backing event"?
    """
    plot_df = plot_df.sort_values(by=["file", "Frame"])

    # Apply the custom function to each file group
    plot_df = _flag_frames(plot_df)

    # Create a new column that assigns a unique group number to each file group
    plot_df["group_number"] = plot_df["flag"].diff().ne(0).cumsum()

    # First, create a mask where Frame equals start_frame
    start_frame_mask = plot_df["start_walk"] == 1

    # Then, for each group, get the value of Frame where Frame equals
    # start_frame
    plot_df["backing_frame"] = plot_df.loc[start_frame_mask, "Frame"]

    # # Forward fill 'backing_frame' within each group
    # plot_df["backing_frame"] = plot_df.groupby("group_number")[
    #     "backing_frame"
    # ].ffill()

    # Backward fill 'backing_frame' within each group
    plot_df["backing_frame"] = plot_df.groupby("group_number")[
        "backing_frame"
    ].bfill()

    # Calculate delta frames
    plot_df["delta_frames"] = plot_df["Frame"] - plot_df["backing_frame"]

    return plot_df


def _flag_frames(df, frame_distance=200):
    """
    Flags frames that are within a certain distance of the start frame.

    This function applies the __flag_frames function to each file group in the
    DataFrame. The __flag_frames function flags frames that are within
    frame_distance of the start frame, where the start frame is the frame
    where 'start_walk' equals 1.

    Args:
        df (pandas.DataFrame): The DataFrame to flag frames in.
        frame_distance (int, optional): The distance from the start frame to
            flag frames within. Default is 200.

    Returns:
        flagged_df (pandas.DataFrame): The DataFrame with flagged frames.
    """

    def __flag_frames(group, frame_distance=200):
        # Reset index for proper index-based operations
        group = group.reset_index(drop=True)
        group["flag"] = 0  # Initialize 'flag' column with zeros
        # Indices where 'start_frame' equals 'Frame'
        start_indices = group.index[group["start_walk"] == 1]
        for idx in start_indices:
            # Flag frames that are within frame_distance of the start frame,
            # which in this case is the frame where 'start_walk' equals 1
            mask = (
                group["Frame"] >= group.loc[idx, "Frame"] - frame_distance
            ) & (group["Frame"] <= group.loc[idx, "Frame"])
            group.loc[mask, "flag"] = 1
        return group

    flagged_df = (
        df.groupby("file")
        .apply(lambda group: __flag_frames(group, frame_distance))
        .reset_index(drop=True)
    )

    return flagged_df


def _calculate_and_plot_means(plot_df):
    """
    Calculates the mean predicted probabilities for each delta frame and plots
    the results.

    This function selects the rows where 'flag' equals 1, calculates the delta
    frames, selects the necessary columns, calculates the mean predicted
    probabilities for each delta frame, and plots the mean predicted
    probabilities against the delta frames.

    Args:
        plot_df (pandas.DataFrame): The DataFrame to calculate and plot means
            from.

    Returns:
        mean_df (pandas.DataFrame): The DataFrame containing the mean
            predicted probabilities for each delta frame.

    TODO: Do we need the plot_df_2 line where we select columns?
    """
    # Plot of predicted probabilities 1 second before and after a true
    # backing event????
    # Select rows where 'flag' equals 1
    plot_df_2 = plot_df[plot_df["flag"] == 1]
    # Calculate delta frames
    plot_df_2.loc[:, "delta_frames"] = (
        plot_df_2["Frame"] - plot_df_2["backing_frame"]
    )
    # Select necessary columns
    plot_df_2 = plot_df_2[
        [
            "file",
            "Frame",
            "start_walk",
            "backing_frame",
            "delta_frames",
            "y_test_true",
            "y_test_pred_prob",
        ]
    ]  # Do we need this line?

    # Calculate mean predicted probabilities for each delta frame
    mean_df = (
        plot_df_2.groupby("delta_frames")[["y_test_pred_prob"]]
        .mean()
        .reset_index()
    )

    #
    # Plot of predictions within 200 frames (5 seconds)
    # Assuming mean_df is your DataFrame and it has columns 'delta_frames'
    # and 'y_test_pred_prob'
    #
    # Plot mean predicted probabilities against delta frames
    plt.figure(figsize=(10, 6))
    plt.plot(mean_df["delta_frames"], mean_df["y_test_pred_prob"])
    plt.xlabel("Delta Frames")
    plt.ylabel("Y_test_pred_prob")
    plt.title("Y_test_pred_prob vs Delta Frames")
    plt.show()

    return mean_df
