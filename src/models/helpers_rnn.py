"""
This module contains helper functions for training a Recurrent Neural
Network (RNN) on a dataset.

It includes functions for saving the trained model and configuration
settings, checking for NaN and inf values in the input tensor, computing
the sum of squared gradients and parameters for a given model, checking
for NaN and inf values in the gradients of a given model, and checking
for NaN and inf values in the loss value.

Functions:
    save_model_and_config(
        model, model_name, timestamp, pickle_path, processed_data_path,
        config, model_dir, config_dir
    ) -> None: Saves the trained model and configuration settings.
    debug_input_nan_inf(inputs) -> None: Checks for NaN and inf values
        in the input tensor.
    debug_sumsq_grad_param(model, sum_sq_gradients, sum_sq_parameters)
    -> float, float: Computes the sum of squared gradients and
        parameters for a given model.
    debug_grad_nan_inf(model, epoch, i) -> None: Checks for NaN and inf
        values in the gradients of a given model.
    debug_loss_nan_inf(epoch, i, loss) -> None: Checks for NaN and inf
        values in the loss value.
"""

import hashlib
import logging

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def save_model_and_config(
    model,
    model_name,
    timestamp,
    pickle_path,
    processed_data_path,
    config,
    model_dir,
    config_dir,
):
    """
    Saves the trained model and configuration settings.

    Args:
        model (torch.nn.Module): The trained RNN model.
        model_name (str): The name of the model.
        timestamp (str): The timestamp to use in the output file names.
        pickle_path (str): The path to the input data pickle file.
        processed_data_path (str): The path to the processed data pickle
            file.
        config (dict): The configuration settings for the model.
        model_dir (pathlib.Path): The directory to save the trained
            model.
        config_dir (pathlib.Path): The directory to save the
            configuration settings.
    """
    # Get the hash values of the model and configuration
    model_hash = hashlib.md5(
        str(model.state_dict()).encode("utf-8")
    ).hexdigest()
    config_hash = hashlib.md5(str(config).encode("utf-8")).hexdigest()

    # Check if the model and configuration already exist
    existing_models = [f.name for f in model_dir.glob("*.pt")]
    existing_configs = [f.name for f in config_dir.glob("*.yaml")]
    if (
        f"rnn_model_{model_hash}.pt" in existing_models
        and f"config_{config_hash}.yaml" in existing_configs
    ):
        logger.info("Model and configuration already exist. Skipping saving.")
    else:
        # Save the trained model
        model_path = (
            model_dir / f"{timestamp}_model_{model_hash}_{config_hash}.pt"
        )
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
        AssertionError: If any NaN or inf values are found in the input
            tensor.
    """
    nan_positions = torch.nonzero(torch.isnan(inputs), as_tuple=True)
    assert not torch.isnan(
        inputs
    ).any(), f"NaN values found at positions {nan_positions}"
    inf_positions = torch.nonzero(torch.isinf(inputs), as_tuple=True)
    assert not torch.isinf(
        inputs
    ).any(), f"inf values found at positions {inf_positions}"


def debug_sumsq_grad_param(model, sum_sq_gradients, sum_sq_parameters):
    """
    Computes the sum of squared gradients and parameters for a given
    model.

    Args:
        model (torch.nn.Module): The model to compute the sum of squared
            gradients and parameters for.
        sum_sq_gradients (float): The current sum of squared gradients.
        sum_sq_parameters (float): The current sum of squared
            parameters.

    Returns:
        sum_sq_gradients (float): The updated sum of squared gradients.
        sum_sq_parameters (float): The updated sum of squared
        parameters.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            sum_sq_gradients += torch.sum(param.grad**2).item()
    for name, param in model.named_parameters():
        if param.data is not None:
            sum_sq_parameters += torch.sum(param.data**2).item()

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
            grad_check = torch.sum(torch.isnan(param.grad)) + torch.sum(
                torch.isinf(param.grad)
            )
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
