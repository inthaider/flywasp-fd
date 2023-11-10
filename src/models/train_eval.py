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
    train_eval_model(
        X_train, Y_train, X_test, Y_test, input_size,
        hidden_size,output_size, num_epochs, batch_size, learning_rate,
        device, batch_first=True, prints_per_epoch=10
    ) -> (torch.nn.Module, numpy.ndarray):
        Trains the RNN model and evaluates it on a test dataset.
    train_loop(
        model, batch_size, device, prints_per_epoch,
        train_loader,criterion, optimizer, epoch
    ) -> (float, float):
        Trains an RNN model on a training dataset for one epoch.
    test_loop(model, device, test_loader, criterion) -> (
        float, float, float, float, numpy.ndarray
    ):
        Evaluates the performance of a trained RNN model on a test
        dataset.

Example:
    To train and evaluate an RNN model with the provided functions, you
    would set up your data and hyperparameters, and then call:

    >>> model, labels_and_probs = train_eval_model(
        X_train, Y_train, X_test, Y_test, input_size=10, hidden_size=20,
        output_size=2, num_epochs=100, batch_size=32, learning_rate=0.001,
        device='cuda'
    )

Note:
    The training process is logged using TensorBoard, allowing for
    real-time monitoring of various metrics. It is assumed that the
    input data is preprocessed and formatted as NumPy arrays suitable
    for input to an RNN model.
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, f1_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

from src.models.helpers_rnn import (
    debug_grad_nan_inf,
    debug_input_nan_inf,
    debug_loss_nan_inf,
    debug_sumsq_grad_param,
)
from src.models.rnn_model import configure_model, data_loaders

logger = logging.getLogger(__name__)


def train_eval_model(
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
    batch_first=True,
    prints_per_epoch=10,
):
    """
    Trains the RNN model and evaluates it on a test dataset.

    Args:
        X_train (numpy.ndarray): The training input data.
        Y_train (numpy.ndarray): The training target data.
        X_test (numpy.ndarray): The testing input data.
        Y_test (numpy.ndarray): The testing target data.
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        output_size (int): The number of output features.
        num_epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size.
        learning_rate (float): The learning rate.
        device (str): The device to use for training.
        batch_first (bool, optional): If True, then the input and output
            tensors are provided as (batch, seq, feature). Default is True.
        prints_per_epoch (int, optional): The number of times to print
            the loss per epoch. Default is 10.

    Returns:
        model (torch.nn.Module): The trained RNN model.
        labels_and_probs (numpy.ndarray): A numpy array containing the
            true labels, predicted labels, and predicted probabilities.
    """
    # Create the data loaders
    train_loader, test_loader = data_loaders(
        X_train, Y_train, X_test, Y_test, batch_size
    )
    # Define the model, loss function, and optimizer
    (
        model,
        criterion,
        optimizer,
        scheduler,
        cross_entropy_weights,
    ) = configure_model(
        Y_train,
        input_size,
        hidden_size,
        output_size,
        learning_rate,
        device,
        batch_first,
    )

    # Create a SummaryWriter for logging to TensorBoard
    writer = SummaryWriter()

    # **************************************************************** #
    #                   TRAIN AND EVALUATE THE MODEL                   #
    # **************************************************************** #
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n-------------------------------")

        print("Training the model...")
        # Train the model
        train_loss, train_f1 = train_loop(
            model,
            batch_size,
            device,
            prints_per_epoch,
            train_loader,
            criterion,
            optimizer,
            epoch,
        )
        print("Evaluating the model...")
        # Evaluate/Test the model
        (
            test_loss,
            test_acc,
            test_f1,
            test_pr_auc,
            test_labels_and_probs,
        ) = test_loop(model, device, test_loader, criterion)

        # Update the learning rate
        scheduler.step()
        # According to GitHub Copilot, CosineAnnealingLR does not take a
        # metric as an argument so it shouldn't be necessary to pass the
        # test loss to the scheduler (like below)
        # scheduler.step(test_loss)

        # Log loss and accuracy to TensorBoard
        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Train F1 Score", train_f1, epoch)
        writer.add_scalar("Test Loss", test_loss, epoch)
        writer.add_scalar("Test Accuracy", test_acc, epoch)
        writer.add_scalar("Test F1 Score", test_f1, epoch)
        writer.add_scalar("Test Precision-Recall AUC", test_pr_auc, epoch)

        print("Training Data Distribution:")
        print(pd.Series(test_labels_and_probs[0]).value_counts().to_dict())
        print("Predicted Data Distribution:")
        print(pd.Series(test_labels_and_probs[1]).value_counts().to_dict())

        print(
            f"\n Epoch {epoch+1} Metrics -- Train Loss: {train_loss:.4f}, "
            f"Train F1 Score: {train_f1:.4f}, Test Loss: {test_loss:.4f}, "
            f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, "
            f"Test PR AUC: {test_pr_auc:.4f}\n"
        )

    # Flush the SummaryWriter doing this ensures that the metrics are
    # written to disk
    writer.flush()
    # Close the SummaryWriter
    writer.close()

    return model, test_labels_and_probs


def train_loop(
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
        device (str): The device to use for training.
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
    size_train = len(train_loader.dataset)  # Number of training samples
    num_batches = len(train_loader)  # Number of batches
    # Print loss prints_per_epoch times per epoch
    print_interval = int(max(num_batches // prints_per_epoch, 1))
    # --------------------#
    print(f"Print interval: {print_interval}")  # Debugging line
    # --------------------# Set the model to training mode - important
    # for batch normalization and dropout layers This is best practice,
    # but is it necessary here in this situation?
    model.train()
    # Initialize running loss & sum of squared gradients and parameters
    running_loss = 0.0
    sum_sq_gradients = 0.0
    sum_sq_parameters = 0.0
    # Initialize true and predicted labels for F1 score calculation
    true_labels = []
    pred_labels = []

    print(f"Number of batches: {num_batches}")  # Print number of batches
    print(f"Batch size: {batch_size}")  # Print batch size
    for i, (inputs, labels) in enumerate(train_loader):
        # Debugging: Check for NaN or inf in inputs
        debug_input_nan_inf(inputs)

        # Note that i is the index of the batch and goes up to
        # num_batches - 1
        inputs, labels = inputs.to(device), labels.to(
            device
        )  # Move tensors to device, e.g. GPU
        optimizer.zero_grad()  # Zero the parameter gradients

        # Compute predicted output and loss
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Debugging: Check for NaN or inf in loss
        debug_loss_nan_inf(epoch, i, loss)
        # Debugging: Check for NaN or inf in gradients
        debug_grad_nan_inf(model, epoch, i)
        # Debugging: Monitor sum of squared gradients and parameters
        sum_sq_gradients, sum_sq_parameters = debug_sumsq_grad_param(
            model, sum_sq_gradients, sum_sq_parameters
        )

        # Get predicted class The line below is basically taking the
        # outputs tensor, which has shape (batch_size, 2), and getting
        # the index of the maximum value in each row (i.e. the predicted
        # class) and returning a tensor of shape (batch_size, 1)
        _, predicted = torch.max(outputs.data, 1)
        # Accumulate true and predicted labels for F1 score calculation
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

        # Print loss every print_interval iterations
        if int(i) % print_interval == 0:
            loss, current_iter = loss.item(), (i + 1) * len(
                inputs
            )  # loss and current iteration
            print(f"Loss: {loss:>7f}  [{current_iter:>5d}/{size_train:>5d}]")

    # Log sum of squared gradients and parameters after each epoch
    logger.info(
        f"\nSum squared grads/params in Epoch {epoch+1}:\n"
        f"\tSum of squared gradients : {sum_sq_gradients:>12.4f}\n"
        f"\tSum of squared parameters: {sum_sq_parameters:>12.4f}"
    )
    # Calculate average loss over all batches
    train_loss = running_loss / len(train_loader)
    # Calculate F1 score for training data
    train_f1 = f1_score(true_labels, pred_labels)

    print(
        f"\nTrain Performance: \n Avg loss: {train_loss:>8f}, "
        f"F1 Score: {train_f1:.4f} \n"
    )

    return train_loss, train_f1


def test_loop(model, device, test_loader, criterion):
    """
    Evaluates the performance of a trained RNN model on a test dataset.

    Args:
        model (torch.nn.Module): The trained RNN model.
        device (str): The device to use for evaluation.
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
    # Set the model to evaluation mode - important for batch
    # normalization and dropout layers This is best practice, but is it
    # necessary here in this situation?
    model.eval()

    # Initialize running loss
    running_loss = 0.0
    correct = 0
    total = 0

    # Initialize true and predicted labels for F1 score calculation
    true_labels = []
    pred_labels = []
    # Initialize probabilities for precision-recall AUC calculation
    pred_probs = []

    # Evaluating the model with torch.no_grad() ensures that no
    # gradients are computed during test mode also serves to reduce
    # unnecessary gradient computations and memory usage for tensors
    # with requires_grad=True
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(
                device
            )  # Move tensors to device, e.g. GPU
            outputs = model(inputs)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss
            running_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class

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

    print(
        f"Test Performance: \n Accuracy: {(100*test_acc):>0.1f}%, "
        f"Avg loss: {test_loss:>8f}, F1 Score: {test_f1:.4f} \n"
    )

    # Store true_labels, pred_labels, and pred_probs in a numpy array
    # will be used for plotting
    labels_and_probs = np.array([true_labels, pred_labels, pred_probs])

    return test_loss, test_acc, test_f1, test_pr_auc, labels_and_probs
