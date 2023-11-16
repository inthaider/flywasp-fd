"""
This module contains helper functions for training a Recurrent Neural
Network (RNN) on a dataset.

Functions:

TODO: Update docstrings
"""

import time
import logging
import math
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from config import HOST_NAME, LOGS_DIR, PROJECT_ROOT, get_current_time

logger = logging.getLogger(__name__)


def create_writer(
    n_layers: int,
    hidden_units: int,
    lr,
    device: str | torch.device = "cpu",
    experiment_num: int = 0,
    model_name: str = "rnn",
) -> SummaryWriter:
    """
    Create a SummaryWriter object for logging the training and test results.

    Args:
        n_layers (int): The number of RNN layers in the model.
        hidden_units (int): The number of hidden units in the model.
        lr (float): The learning rate.
        device (str | torch.device): The device used for training.
        experiment_num (int): The name of the experiment.
        model_name (str): The name of the model.

    Returns:
        SummaryWriter: The SummaryWriter object.
    """

    def get_tb_log_dir():
        """Returns the path to the tensorboard log directory."""
        return os.path.join(
            PROJECT_ROOT,
            LOGS_DIR,
            "tb_runs",
            HOST_NAME,
            f"{get_current_time()}_{model_name}_nhl{n_layers}_"
            + f"nhu{hidden_units}_lr{lr}_{device}",
        )

    return SummaryWriter(get_tb_log_dir())


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


# ******************************************************************** #
#                            DEBUGGING STUFF                           #
# ******************************************************************** #
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
