import hashlib
import logging

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt


def plot_predicted_probabilities(df, test_indices, test_labels_and_probs):
    """
    Plot the predicted probabilities using the _make_df_for_plotting(), _process_df_for_plotting(), and _calculate_and_plot_means() functions.

    """
    # Unpack the test_labels_and_probs numpy array
    test_pred_labels = test_labels_and_probs[1]
    test_pred_probs = test_labels_and_probs[2]

    # Create a DataFrame for plotting
    plot_df = _make_df_for_plotting(
        df, test_indices, test_pred_labels, test_pred_probs)
    # Process the DataFrame for plotting
    plot_df = _process_df_for_plotting(plot_df)
    # Calculate and plot the means
    mean_df = _calculate_and_plot_means(plot_df)

    return plot_df, mean_df


def _make_df_for_plotting(df, test_indices, test_pred_labels, test_pred_probs):
    """
    """
    merged_df = pd.DataFrame({
        'index': test_indices,
        'y_pred_test': test_pred_labels,
        'y_pred': test_pred_probs
    })
    plot_df = pd.merge(df, merged_df, left_index=True,
                       right_on='index', how='right')
    return plot_df


def _process_df_for_plotting(plot_df):
    """
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

    # Calculate delta frmes
    plot_df['delta_frames'] = plot_df['Frame'] - plot_df['backing_frame']

    return plot_df


def _flag_frames(df, frame_distance=200):
    """
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
    TODO: Do we need the plot_df_2 line where we select columns?
    """
    # Plot of predicted probabilities 1 second before and after a true backing event????
    plot_df_2 = plot_df[plot_df['flag'] == 1]
    plot_df_2['delta_frames'] = plot_df_2['Frame'] - plot_df_2['backing_frame']
    plot_df_2 = plot_df_2[['file', 'Frame', 'start_walk',
                           'backing_frame', 'delta_frames', 'y_test', 'y_pred']]  # Do we need this line?

    mean_df = plot_df_2.groupby('delta_frames')[
        ['y_pred']].mean().reset_index()

    #
    # Plot of predictions within 200 frames (5 seconds
    # Assuming mean_df is your DataFrame and it has columns 'delta_frames' and 'y_pred'
    #
    plt.figure(figsize=(10, 6))
    # Plot mean of 'y_pred' vs mean of 'delta_frames'
    plt.plot(mean_df['delta_frames'], mean_df['y_pred'])

    plt.xlabel('Delta Frames')
    plt.ylabel('Y_pred')
    plt.title('Y_pred vs Delta Frames')

    plt.show()

    return mean_df


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
    """
    # Get the hash values of the model and configuration
    model_hash = hashlib.md5(
        str(model.state_dict()).encode('utf-8')).hexdigest()
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


def debug_input_nan_inf(inputs):
    """
    Checks for NaN and inf values in the input tensor.

    Parameters
    ----------
    inputs : torch.Tensor
        The input tensor.
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

    Parameters
    ----------
    model : torch.nn.Module
        The model to compute the sum of squared gradients and parameters for.
    sum_sq_gradients : float
        The current sum of squared gradients.
    sum_sq_parameters : float
        The current sum of squared parameters.

    Returns
    -------
    None
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            sum_sq_gradients += torch.sum(param.grad ** 2).item()
    for name, param in model.named_parameters():
        if param.data is not None:
            sum_sq_parameters += torch.sum(param.data ** 2).item()


def debug_grad_nan_inf(model, epoch, i):
    """
    Checks for NaN and inf values in the gradients of a given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to check the gradients of.
    epoch : int
        The current epoch number.
    i : int
        The current iteration number.
    """
    log_invalid_grad = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_check = torch.sum(torch.isnan(
                param.grad)) + torch.sum(torch.isinf(param.grad))
            if grad_check > 0:
                if log_invalid_grad:
                    logging.warning(
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

    Parameters
    ----------
    epoch : int
        The current epoch number.
    i : int
        The current iteration number.
    loss : torch.Tensor
        The loss value.
    """
    log_invalid_loss = True
    if np.isnan(loss.item()) or np.isinf(loss.item()):
        if log_invalid_loss:
            logging.warning(
                f"First occurrence of invalid LOSS:\n"
                f"\tLoss        : {loss.item():>15.4f}\n"
                f"\tIteration   : {i:>15d}\n"
                f"\tEpoch       : {epoch+1:>15d}\n"
                f"\tFurther warnings will be suppressed."
            )
            log_invalid_loss = False
