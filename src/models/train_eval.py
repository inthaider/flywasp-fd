"""
This module contains functions for training and evaluating a Recurrent
Neural Network (RNN) model.

It includes functions for training the model for a specified number of
epochs, training the model for one epoch, and evaluating the model on a
test dataset. It also includes helper functions for debugging the model
during training, such as checking for NaN and inf values in the input
tensor, the gradients of the model, and the loss value, and computing
the sum of squared gradients and parameters for the model.

Functions:

Example:

TODO: Update docstrings
"""

import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, f1_score, precision_recall_curve

from tqdm.auto import tqdm

from src.models.helpers_rnn import (
    create_writer,
    debug_input_nan_inf,
    debug_sumsq_grad_param,
)

logger = logging.getLogger(__name__)
writer = None


# ******************************************************************** #
#               HIGHER LEVEL TRAINING+EVALUATION FUNCTION              #
# ******************************************************************** #
def train_eval(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    input_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    learning_rate: float,
    nonlinearity: str,
    batch_size: int,
    num_epochs: int,
    seq_len: int,
    device: str | torch.device = "cpu",
    prints_per_epoch=10,
):
    """
    Trains the RNN model and evaluates it on a test dataset.

    Args:

    Returns:
        model (torch.nn.Module): The trained RNN model.
        labels_and_probs (numpy.ndarray): A numpy array containing the
            true labels, predicted labels, and predicted probabilities.

    TODO: Use torchmetrics to compute and track metrics.
    TODO: Utilize helpers_rnn.timeSince(since) function to track time.
    TODO:
    """
    print(
        f"\nTraining RNN Model ({device})...\n==============================\n"
    )
    # ====================== Tensorboard logging ===================== #
    global writer
    writer = create_writer(
        num_hidden_layers,
        hidden_size,
        learning_rate,
        seq_len=seq_len,
        device=device,
    )  # Tensorboard log
    (
        train_acc,
        test_acc,
        train_loss,
        test_loss,
        train_f1,
        test_f1,
        test_pr_auc,
    ) = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )  # Declare metrics to track
    # **************************************************************** #
    #                   TRAIN AND EVALUATE THE MODEL                   #
    # **************************************************************** #
    # Set the model to training mode - important for batch normalization
    # and dropout layers This is best practice, but is it necessary here
    model.train()
    start_time = time.time()  # Start a timer
    for epoch in tqdm(
        range(num_epochs),
        desc="Epochs",
        total=num_epochs,
        position=0,
        leave=True,
    ):  # Wrap range with tqdm to create progress bar for epochs
        print(f"Epoch {epoch+1}/{num_epochs}\n-------------------------------")

        # ========================= Training ========================= #
        print("Training the model...")
        start_time_train = time.time()  # Start a timer for training
        train_loss, train_acc, train_f1 = _train_loop(
            model,
            batch_size,
            device,
            prints_per_epoch,
            train_loader,
            criterion,
            optimizer,
            epoch,
        )  # Train the model
        end_time_train = time.time()  # Stop the training timer
        elapsed_time_train = round((end_time_train - start_time_train) / 60, 1)
        logger.info(
            f"Training Epoch {epoch+1} took {elapsed_time_train} minutes."
        )
        # ------------------------------------------------------------ #
        # ======================== Evaluating ======================== #
        print("Evaluating the model...")
        start_time_test = time.time()  # Start a timer for evaluation
        (
            test_loss,
            test_acc,
            test_f1,
            test_pr_auc,
            test_labels_and_probs,
        ) = _test_loop(
            model, device, test_loader, criterion, epoch
        )  # Evaluate/Test the model
        end_time_test = time.time()  # Stop the evaluation timer
        elapsed_time_test = round((end_time_test - start_time_test) / 60, 1)
        logger.info(
            f"Evaluating Epoch {epoch+1} took {elapsed_time_test} minutes."
        )
        # ------------------------------------------------------------ #
        scheduler.step()  # Update the learning rate
        """
        According to GitHub Copilot, CosineAnnealingLR does not take a
        metric as an argument so it shouldn't be necessary to pass the
        test loss to the scheduler (like below)
        scheduler.step(test_loss)
        """
        # ======================== Tensorboard ======================= #
        metrics = {
            "Mean Loss/train_epoch": train_loss,
            "Accuracy/train_epoch": train_acc,
            "F1 Score/train_epoch": train_f1,
            "Mean Loss/test_epoch": test_loss,
            "Accuracy/test_epoch": test_acc,
            "F1 Score/test_epoch": test_f1,
            "Precision-Recall AUC/test_epoch": test_pr_auc,
        }
        # Log to TensorBoard
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(metric_name, metric_value, epoch)
        writer.add_pr_curve(
            "PR Curve/test_epoch",
            test_labels_and_probs[0],
            test_labels_and_probs[1],
            epoch,
        )
        # ------------------------------------------------------------ #
        logger.info(
            "Training Data Distribution: "
            f"{pd.Series(test_labels_and_probs[0]).value_counts().to_dict()}"
        )
        logger.info(
            "Predicted Data Distribution: "
            f"{pd.Series(test_labels_and_probs[1]).value_counts().to_dict()}"
        )
        # ------------------------------------------------------------ #
        logger.info(
            f"Epoch {epoch+1} Metrics --\n"
            f"Train Loss: {train_loss:.4f},\n"
            f"Test Loss: {test_loss:.4f},\n"
            f"Train Acc: {(100*train_acc):>0.1f}%,\n"
            f"Test Acc: {(100*test_acc):>0.1f}%,\n"
            f"Train F1 Score: {train_f1:.4f},\n"
            f"Test F1: {test_f1:.4f},\n"
            f"Test PR AUC: {test_pr_auc:.4f}\n\n"
        )
        # ------------------------------------------------------------ #

    end_time = time.time()  # Stop the timer
    elapsed_time = round((end_time - start_time) / 60, 1)
    logger.info(
        f"The entire train+eval code took {elapsed_time} minutes to run.\n\n"
    )
    # =========== Hyperparameter tracking with TensorBoard =========== #
    hparams = {
        "seq_len": seq_len,
        "input_size": input_size,
        "hidden_units": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "learning_rate": learning_rate,
        "nonlinearity": nonlinearity,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "device": str(device),
    }  # Define the hyperparameters to track
    hparam_metrics = {
        "hparam/Accuracy/train": train_acc,
        "hparam/Accuracy/test": test_acc,
        "hparam/Mean Loss/train": train_loss,
        "hparam/Mean Loss/test": test_loss,
        "hparam/F1 Score/train": train_f1,
        "hparam/F1 Score/test": test_f1,
        "hparam/Precision-Recall AUC/test": test_pr_auc,
    }  # define the metrics to track with the hyperparameters
    writer.add_hparams(hparams, hparam_metrics)  # log hparams & metrics
    # ---------------------------------------------------------------- #
    # Flush SummaryWriter to ensure metrics are written to disk
    writer.flush()
    writer.close()  # Close the SummaryWriter
    return model, test_labels_and_probs  # type: ignore


# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #


# ******************************************************************** #
#                          TRAINING STEP/LOOP                          #
# ******************************************************************** #
def _train_loop(
    model,
    batch_size,
    device,
    prints_per_epoch,
    train_loader,
    criterion,
    optimizer,
    epoch,
):
    """
    Trains an RNN model on a training dataset for one epoch.

    Args:
        model (torch.nn.Module): The RNN model to train.
        batch_size (int): The batch size.
        device (torch.device): The device to use for training.
        prints_per_epoch (int): The number of times to print the loss
            per epoch.
        train_loader (torch.utils.data.DataLoader): The training data
            loader.
        criterion (torch.nn.modules.loss._Loss): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current epoch number.

    Returns:
        train_loss (float): The average training loss over all batches.
        train_f1 (float): The training F1 score.
    """
    # ---------------------------------------------------------------- #
    global writer  # Use the global SummaryWriter object
    assert (
        writer is not None
    ), "Call train_eval_model() before calling _train_loop()"
    # ---------------------------------------------------------------- #
    size_train = len(train_loader.dataset)  # Number of training samples
    num_batches = len(train_loader)  # Number of batches
    # Print loss prints_per_epoch times per epoch
    print_interval = int(max(num_batches // prints_per_epoch, 1))
    # --------------------#
    # print(f"Print interval: {print_interval}")  # Debugging line
    # --------------------#
    # Initialize running loss & sum of squared gradients and parameters
    running_loss = 0.0
    correct = 0
    total = 0
    sum_sq_gradients = 0.0
    sum_sq_parameters = 0.0
    # Initialize true and predicted labels for F1 score calculation
    true_labels = []
    pred_labels = []
    # -------------------------------------------------------------------- #
    logger.info(f"Number of batches: {num_batches}")  # Print number of batches
    logger.info(f"Batch size: {batch_size}\n")  # Print batch size
    # ---------------------------------------------------------------- #
    model.train()
    for i, (inputs, labels) in tqdm(
        enumerate(train_loader),
        desc="Training...",
        total=len(train_loader),
        position=0,
        leave=True,
    ):  # Wrap loader with tqdm to create progress bar for batches
        optimizer.zero_grad()  # Zero the parameter gradients
        # Debugging: Check for NaN or inf in inputs
        debug_input_nan_inf(inputs)
        # Note that i is the index of the batch and goes up to
        # num_batches - 1
        inputs, labels = inputs.to(device), labels.to(
            device
        )  # Move tensors to device, e.g. GPU

        # Compute predicted output and loss
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        # log the loss
        writer.add_scalar(
            "Loss Curve/train",
            loss.item(),
            epoch * len(train_loader) + i,
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # # Debugging: Check for NaN or inf in loss
        # debug_loss_nan_inf(epoch, i, loss)
        # # Debugging: Check for NaN or inf in gradients
        # debug_grad_nan_inf(model, epoch, i)
        # Debugging: Monitor sum of squared gradients and parameters
        sum_sq_gradients, sum_sq_parameters = debug_sumsq_grad_param(
            model, sum_sq_gradients, sum_sq_parameters
        )

        # Get predicted class The line below is basically taking the
        # outputs tensor, which has shape (batch_size, 2), and getting
        # the index of the maximum value in each row (i.e. the predicted
        # class) and returning a tensor of shape (batch_size, 1)
        _, predicted = torch.max(outputs.data, 1)

        # Accumulate total number of samples
        total += labels.size(0)
        # Accumulate number of correct predictions
        correct += (predicted == labels).sum().item()

        # Accumulate true and predicted labels for F1 score calculation
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

        # Print loss every print_interval iterations
        if int(i) % print_interval == 0:
            loss, current_iter = loss.item(), (i + 1) * len(
                inputs
            )  # loss and current iteration
            print(f"Loss: {loss:>7f}  [{current_iter:>5d}/{size_train:>5d}]")
        # ------------------------------------------------------------ #
    writer.add_scalar(
        "Sum Squared Gradients/train_epoch",
        sum_sq_gradients,
        epoch,
    )
    writer.add_scalar(
        "Sum Squared Parameters/train_epoch",
        sum_sq_parameters,
        epoch,
    )
    # ---------------------------------------------------------------- #
    # Log sum of squared gradients and parameters after each epoch
    logger.info(
        f"\nSum squared grads/params in Epoch {epoch+1}:\n"
        f"\tSum of squared gradients : {sum_sq_gradients:>12.4f}\n"
        f"\tSum of squared parameters: {sum_sq_parameters:>12.4f}"
    )
    # Calculate average loss over all batches
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Calculate F1 score for training data
    train_f1 = f1_score(true_labels, pred_labels)

    logger.info(
        f"\nTrain Performance: \n Accuracy: {(100*train_acc):>0.1f}%, "
        f"Avg loss: {train_loss:>8f}, F1 Score: {train_f1:.4f} \n"
    )

    return train_loss, train_acc, train_f1


# ******************************************************************** #
#                            TEST STEP/LOOP                            #
# ******************************************************************** #
def _test_loop(model, device, test_loader, criterion, epoch):
    """
    Evaluates the performance of a trained RNN model on a test dataset.

    Args:
        model (torch.nn.Module): The trained RNN model.
        device (torch.device): The device to use for evaluation.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        criterion (torch.nn.modules.loss._Loss): The loss function.

    Returns:
        test_loss (float): The average test loss over all batches.
        test_acc (float): The test accuracy.
        test_f1 (float): The test F1 score.
        test_pr_auc (float): The test precision-recall AUC score.
        labels_and_probs (numpy.ndarray): A numpy array containing the
            true labels, predicted labels, and predicted probabilities.

    TODO: Check/integrate changes from FD.
    """
    global writer  # Use the global SummaryWriter object
    assert (
        writer is not None
    ), "Call train_eval_model() before calling _test_loop()"
    # ---------------------------------------------------------------- #
    # Initialize running loss
    running_loss = 0.0
    correct = 0
    total = 0
    # Initialize true and predicted labels for F1 score calculation
    true_labels = []
    pred_labels = []
    # Initialize probabilities for precision-recall AUC calculation
    pred_probs = []
    # ---------------------------------------------------------------- #
    # Set the model to evaluation mode - important for batch
    # normalization and dropout layers This is best practice, but is it
    # necessary here in this situation?
    model.eval()
    # Evaluating the model with torch.no_grad() ensures that no
    # gradients are computed during test mode also serves to reduce
    # unnecessary gradient computations and memory usage for tensors
    # with requires_grad=True
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(
            enumerate(test_loader),
            desc="Testing...",
            total=len(test_loader),
            position=0,
            leave=True,
        ):  # Wrap loader with tqdm to create progress bar for batches
            inputs, labels = inputs.to(device), labels.to(
                device
            )  # Move tensors to device, e.g. GPU
            outputs = model(inputs)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss
            running_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class

            # log the loss
            writer.add_scalar(
                "Loss Curve/test",
                loss.item(),
                epoch * len(test_loader) + i,
            )

            # Accumulate total number of samples
            total += labels.size(0)
            # Accumulate number of correct predictions
            correct += (predicted == labels).sum().item()
            # Get predicted probabilities
            probabilities = F.softmax(outputs, dim=1)
            prob_of_class_1 = probabilities[:, 1]

            # Accumulate true and predicted labels for F1 score
            # calculation
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
            # Accumulate probabilities for precision-recall AUC
            # calculation
            pred_probs.extend(prob_of_class_1.cpu().numpy())

    # Calculate average loss and accuracy over all batches
    test_loss = running_loss / len(test_loader)
    test_acc = correct / total

    # Calculate F1 score for testing data
    test_f1 = f1_score(true_labels, pred_labels, average="macro")
    # Calculate precision-recall AUC score for testing data
    test_prec, test_rec, _ = precision_recall_curve(true_labels, pred_probs)
    test_pr_auc = auc(test_rec, test_prec)

    logger.info(
        f"Test Performance: \n Accuracy: {(100*test_acc):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}, F1 Score: {test_f1:.4f} \n"
    )

    # Store true_labels, pred_labels, and pred_probs in a numpy array
    # will be used for plotting
    labels_and_probs = np.array([true_labels, pred_labels, pred_probs])

    return test_loss, test_acc, test_f1, test_pr_auc, labels_and_probs
