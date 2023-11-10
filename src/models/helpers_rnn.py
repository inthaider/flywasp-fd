"""
This module contains helper functions for training a Recurrent Neural Network (RNN) on a dataset.

It includes functions for plotting predicted probabilities, creating and processing dataframes for plotting, flagging frames, calculating and plotting means, saving the trained model and configuration settings, checking for NaN and inf values in the input tensor, computing the sum of squared gradients and parameters for a given model, checking for NaN and inf values in the gradients of a given model, and checking for NaN and inf values in the loss value.

Functions:
    plot_predicted_probabilities(df, test_indices, test_labels_and_probs) -> pd.DataFrame, pd.DataFrame: 
        Plots the predicted probabilities using the _make_df_for_plotting(), _process_df_for_plotting(), and _calculate_and_plot_means() functions.
    _make_df_for_plotting(df, test_indices, test_true_labels, test_pred_probs) -> pd.DataFrame: 
        Creates a DataFrame for plotting.
    _process_df_for_plotting(plot_df) -> pd.DataFrame: 
        Processes the DataFrame for plotting.
    _flag_frames(df, frame_distance=200) -> pd.DataFrame: 
        Flags frames that are within frame_distance of the start frame.
    _calculate_and_plot_means(plot_df) -> pd.DataFrame: 
        Calculates and plots the means.
    save_model_and_config(model, model_name, timestamp, pickle_path, processed_data_path, config, model_dir, config_dir) -> None: 
        Saves the trained model and configuration settings.
    debug_input_nan_inf(inputs) -> None: 
        Checks for NaN and inf values in the input tensor.
    debug_sumsq_grad_param(model, sum_sq_gradients, sum_sq_parameters) -> float, float: 
        Computes the sum of squared gradients and parameters for a given model.
    debug_grad_nan_inf(model, epoch, i) -> None: 
        Checks for NaN and inf values in the gradients of a given model.
    debug_loss_nan_inf(epoch, i, loss) -> None: 
        Checks for NaN and inf values in the loss value.
"""

import hashlib
import logging

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def plot_predicted_probabilities(df, test_indices, test_labels_and_probs):
    """
    Plot the predicted probabilities using the _make_df_for_plotting(), _process_df_for_plotting(), and _calculate_and_plot_means() functions.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        test_indices (numpy.ndarray): The indices of the test data.
        test_labels_and_probs (numpy.ndarray): A numpy array containing the true labels, predicted labels, and predicted probabilities.

    Returns:
        plot_df (pandas.DataFrame): The processed DataFrame for plotting.
        mean_df (pandas.DataFrame): The DataFrame containing the calculated means.
    """
    # Unpack the test_labels_and_probs numpy array
    test_true_labels = test_labels_and_probs[0]
    test_pred_labels = test_labels_and_probs[1]
    test_pred_probs = test_labels_and_probs[2]

    # Create a DataFrame for plotting
    plot_df = _make_df_for_plotting(
        df, test_indices, test_true_labels, test_pred_probs)
    # Process the DataFrame for plotting
    plot_df = _process_df_for_plotting(plot_df)
    # Calculate and plot the means
    mean_df = _calculate_and_plot_means(plot_df)

    return plot_df, mean_df


def _make_df_for_plotting(df, test_indices, test_true_labels, test_pred_probs):
    """
    Creates a DataFrame for plotting the predicted probabilities.

    This function merges the test indices, true labels, and predicted probabilities with the original DataFrame.

    Args:
        df (pandas.DataFrame): The original DataFrame containing the data.
        test_indices (numpy.ndarray): The indices of the test data.
        test_true_labels (numpy.ndarray): The true labels of the test data.
        test_pred_probs (numpy.ndarray): The predicted probabilities of the test data.

    Returns:
        plot_df (pandas.DataFrame): The merged DataFrame for plotting.
    """
    merged_df = pd.DataFrame({
        'index': test_indices,
        'y_test_true': test_true_labels,
        'y_test_pred_prob': test_pred_probs
    })
    plot_df = pd.merge(df, merged_df, left_index=True,
                       right_on='index', how='right')
    return plot_df


def _process_df_for_plotting(plot_df):
    """
    Processes a DataFrame for plotting the predicted probabilities.

    This function sorts the DataFrame by file and frame, flags frames, assigns a unique group number to each file group, and calculates the delta frames.

    Args:
        plot_df (pandas.DataFrame): The DataFrame to process.

    Returns:
        plot_df (pandas.DataFrame): The processed DataFrame for plotting.

    TODO: What exactly is plot_df_2 and what do we mean by "plot of predicted probabilities 1 second before and after a true backing event"?
    """
    plot_df = plot_df.sort_values(by=["file", "Frame"])

    # Apply the custom function to each file group
    plot_df = _flag_frames(plot_df)

    # Create a new column that assigns a unique group number to each file group
    plot_df['group_number'] = plot_df['flag'].diff().ne(0).cumsum()

    # First, create a mask where Frame equals start_frame
    start_frame_mask = (plot_df['start_walk'] == 1)

    # Then, for each group, get the value of Frame where Frame equals start_frame
    plot_df['backing_frame'] = plot_df.loc[start_frame_mask, 'Frame']

    # # Forward fill 'backing_frame' within each group
    # plot_df['backing_frame'] = plot_df.groupby('group_number')['backing_frame'].ffill()

    # Backward fill 'backing_frame' within each group
    plot_df['backing_frame'] = plot_df.groupby(
        'group_number')['backing_frame'].bfill()

    # Calculate delta frames
    plot_df['delta_frames'] = plot_df['Frame'] - plot_df['backing_frame']

    return plot_df


def _flag_frames(df, frame_distance=200):
    """
    Flags frames that are within a certain distance of the start frame.

    This function applies the __flag_frames function to each file group in the DataFrame. The __flag_frames function flags frames that are within frame_distance of the start frame, where the start frame is the frame where 'start_walk' equals 1.

    Args:
        df (pandas.DataFrame): The DataFrame to flag frames in.
        frame_distance (int, optional): The distance from the start frame to flag frames within. Default is 200.

    Returns:
        flagged_df (pandas.DataFrame): The DataFrame with flagged frames.
    """
    def __flag_frames(group, frame_distance=200):
        # Reset index for proper index-based operations
        group = group.reset_index(drop=True)
        group['flag'] = 0  # Initialize 'flag' column with zeros
        # Indices where 'start_frame' equals 'Frame'
        start_indices = group.index[group['start_walk'] == 1]
        for idx in start_indices:
            # Flag frames that are within frame_distance of the start frame,
            # which in this case is the frame where 'start_walk' equals 1
            mask = (group['Frame'] >= group.loc[idx, 'Frame'] -
                    frame_distance) & (group['Frame'] <= group.loc[idx, 'Frame'])
            group.loc[mask, 'flag'] = 1
        return group

    flagged_df = df.groupby('file').apply(lambda group: __flag_frames(
        group, frame_distance)).reset_index(drop=True)

    return flagged_df


def _calculate_and_plot_means(plot_df):
    """
    Calculates the mean predicted probabilities for each delta frame and plots the results.

    This function selects the rows where 'flag' equals 1, calculates the delta frames, selects the necessary columns, calculates the mean predicted probabilities for each delta frame, and plots the mean predicted probabilities against the delta frames.

    Args:
        plot_df (pandas.DataFrame): The DataFrame to calculate and plot means from.

    Returns:
        mean_df (pandas.DataFrame): The DataFrame containing the mean predicted probabilities for each delta frame.

    TODO: Do we need the plot_df_2 line where we select columns?
    """
    # Plot of predicted probabilities 1 second before and after a true backing event????
    # Select rows where 'flag' equals 1
    plot_df_2 = plot_df[plot_df['flag'] == 1]
    # Calculate delta frames
    plot_df_2.loc[:, 'delta_frames'] = plot_df_2['Frame'] - \
        plot_df_2['backing_frame']
    # Select necessary columns
    plot_df_2 = plot_df_2[['file', 'Frame', 'start_walk',
                           'backing_frame', 'delta_frames', 'y_test_true', 'y_test_pred_prob']]  # Do we need this line?

    # Calculate mean predicted probabilities for each delta frame
    mean_df = plot_df_2.groupby('delta_frames')[
        ['y_test_pred_prob']].mean().reset_index()

    #
    # Plot of predictions within 200 frames (5 seconds)
    # Assuming mean_df is your DataFrame and it has columns 'delta_frames' and 'y_test_pred_prob'
    #
    # Plot mean predicted probabilities against delta frames
    plt.figure(figsize=(10, 6))
    plt.plot(mean_df['delta_frames'], mean_df['y_test_pred_prob'])
    plt.xlabel('Delta Frames')
    plt.ylabel('Y_test_pred_prob')
    plt.title('Y_test_pred_prob vs Delta Frames')
    plt.show()

    return mean_df


def save_model_and_config(model, model_name, timestamp, pickle_path, processed_data_path, config, model_dir, config_dir):
    """
    Saves the trained model and configuration settings.

    Args:
        model (torch.nn.Module): The trained RNN model.
        model_name (str): The name of the model.
        timestamp (str): The timestamp to use in the output file names.
        pickle_path (str): The path to the input data pickle file.
        processed_data_path (str): The path to the processed data pickle file.
        config (dict): The configuration settings for the model.
        model_dir (pathlib.Path): The directory to save the trained model.
        config_dir (pathlib.Path): The directory to save the configuration settings.
    """
    # Get the hash values of the model and configuration
    model_hash = hashlib.md5(
        str(model.state_dict()).encode('utf-8')).hexdigest()
    config_hash = hashlib.md5(str(config).encode('utf-8')).hexdigest()

    # Check if the model and configuration already exist
    existing_models = [f.name for f in model_dir.glob("*.pt")]
    existing_configs = [f.name for f in config_dir.glob("*.yaml")]
    if f"rnn_model_{model_hash}.pt" in existing_models and f"config_{config_hash}.yaml" in existing_configs:
        logger.info("Model and configuration already exist. Skipping saving.")
    else:
        # Save the trained model
        model_path = model_dir / \
            f"{timestamp}_model_{model_hash}_{config_hash}.pt"
        torch.save(model.state_dict(), model_path)

        # Save the configuration settings
        config_path = config_dir / f"{timestamp}_config_{config_hash}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)


def debug_input_nan_inf(inputs):
    """
    Checks for NaN and inf values in the input tensor.

    Args:
        inputs (torch.Tensor): The input tensor.

    Raises:
        AssertionError: If any NaN or inf values are found in the input tensor.
    """
    nan_positions = torch.nonzero(torch.isnan(inputs), as_tuple=True)
    assert not torch.isnan(inputs).any(
    ), f"NaN values found at positions {nan_positions}"
    inf_positions = torch.nonzero(torch.isinf(inputs), as_tuple=True)
    assert not torch.isinf(inputs).any(
    ), f"inf values found at positions {inf_positions}"


def debug_sumsq_grad_param(model, sum_sq_gradients, sum_sq_parameters):
    """
    Computes the sum of squared gradients and parameters for a given model.

    Args:
        model (torch.nn.Module): The model to compute the sum of squared gradients and parameters for.
        sum_sq_gradients (float): The current sum of squared gradients.
        sum_sq_parameters (float): The current sum of squared parameters.

    Returns:
        sum_sq_gradients (float): The updated sum of squared gradients.
        sum_sq_parameters (float): The updated sum of squared parameters.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            sum_sq_gradients += torch.sum(param.grad ** 2).item()
    for name, param in model.named_parameters():
        if param.data is not None:
            sum_sq_parameters += torch.sum(param.data ** 2).item()

    return sum_sq_gradients, sum_sq_parameters


def debug_grad_nan_inf(model, epoch, i):
    """
    Checks for NaN and inf values in the gradients of a given model.

    Args:
        model (torch.nn.Module): The model to check the gradients of.
        epoch (int): The current epoch number.
        i (int): The current iteration number.
    """
    log_invalid_grad = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_check = torch.sum(torch.isnan(
                param.grad)) + torch.sum(torch.isinf(param.grad))
            if grad_check > 0:
                if log_invalid_grad:
                    logger.warning(
                        f"First occurrence of invalid GRADIENT:\n"
                        f"\tParameter   : {name:>15s}\n"
                        f"\tIteration   : {i:>15d}\n"
                        f"\tEpoch       : {epoch+1:>15d}\n"
                        f"\tFurther warnings will be suppressed."
                    )
                    log_invalid_grad = False


def debug_loss_nan_inf(epoch, i, loss):
    """
    Checks for NaN and inf values in the loss value.

    Args:
        epoch (int): The current epoch number.
        i (int): The current iteration number.
        loss (torch.Tensor): The loss value.
    """
    log_invalid_loss = True
    if np.isnan(loss.item()) or np.isinf(loss.item()):
        if log_invalid_loss:
            logger.warning(
                f"First occurrence of invalid LOSS:\n"
                f"\tLoss        : {loss.item():>15.4f}\n"
                f"\tIteration   : {i:>15d}\n"
                f"\tEpoch       : {epoch+1:>15d}\n"
                f"\tFurther warnings will be suppressed."
            )
            log_invalid_loss = False
