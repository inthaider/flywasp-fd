import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


class RNN(nn.Module):
    """
    A class representing the RNN model.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input.
    hidden_size : int
        The number of features in the hidden state.
    output_size : int
        The number of output features.
    batch_first : bool, optional
        If True, then the input and output tensors are provided as (batch, seq, feature).
        Default is True.
    """

    def __init__(self, input_size, hidden_size, output_size, batch_first=True):
        """
        Initialize the RNN model.

        Parameters
        ----------
        input_size : int
            The number of expected features in the input.
        hidden_size : int
            The number of features in the hidden state.
        output_size : int
            The number of output features.
        batch_first : bool, optional
            If True, then the input and output tensors are provided as (batch, seq, feature).
            Default is True.
        """
        # Get the current timestamp as a string in the format YYYYMMDD
        self.timestamp = datetime.now().strftime("%Y%m%d")

        # Call the __init__ method of the parent class (nn.Module)
        super(RNN, self).__init__()

        # Set the hidden_size attribute of the RNN object
        self.hidden_size = hidden_size

        # Create an RNN layer with the specified input_size, hidden_size, and batch_first parameters
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=batch_first)

        # Create a linear layer with the specified hidden_size and output_size parameters
        self.fc = nn.Linear(hidden_size, output_size)

        # Create a sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the RNN model.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The output data.

        Notes
        -----
        Using `.to(x.device)` in the `forward()` method ensures that the model
            is moved to the same device as the input data.
        """
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(
            x.device)  # initial hidden state
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x, h0)
        # decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class WalkDataset(Dataset):
    """
    A class representing the dataset.

    Parameters
    ----------
    X : numpy.ndarray
        The input data.
    Y : numpy.ndarray
        The target data.
    """

    def __init__(self, X, Y):
        """
        Initialize the dataset.

        Parameters
        ----------
        X : numpy.ndarray
            The input data.
        Y : numpy.ndarray
            The target data.
        """
        # Convert the input and target data to PyTorch tensors
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        len(self.X) : int
            The length of the dataset.
        """
        # Return the length of the input data
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Parameters
        ----------
        idx : int
            The index of the item to return.

        Returns
        -------
        self.X[idx] : torch.Tensor
            The input data at the given index.
        self.Y[idx] : torch.Tensor
            The target data at the given index.
        """
        # Return a tuple containing the input and target data at the given index
        return self.X[idx], self.Y[idx]


def init_param_weights(m):
    """
    Optional function for weight initialization.

    Uses Xavier uniform initialization for weights and constant initialization
    for biases.

    Parameters
    ----------
    m : torch.nn.Module
        The module to initialize. Only applies to Linear layers.
    """
    # If the module is a linear layer, initialize the weights with Xavier uniform initialization
    # and the biases with a constant value of 0.01
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def data_loaders(X_train, Y_train, X_test, Y_test, batch_size):
    """
    Loads the training and testing data into PyTorch DataLoader objects.

    Parameters
    ----------
    X_train : numpy.ndarray
        The training input data.
    Y_train : numpy.ndarray
        The training target data.
    X_test : numpy.ndarray
        The testing input data.
    Y_test : numpy.ndarray
        The testing target data.
    batch_size : int
        The batch size.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        The training data loader.
    test_loader : torch.utils.data.DataLoader
        The testing data loader.
    """
    train_dataset = WalkDataset(X_train, Y_train)
    test_dataset = WalkDataset(X_test, Y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _init_cross_entropy_weights(Y_train):
    """
    """
    class_counts = [pd.Series(Y_train).value_counts()[
        0], pd.Series(Y_train).value_counts()[1]]

    # Compute class weights
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    return weights


def loss_function(Y_train):
    """
    """
    # Calculating weights for class imbalance to pass to the loss function
    cross_entropy_weights = _init_cross_entropy_weights(Y_train)

    # Using CrossEntropyLoss as the loss function with weights
    criterion = nn.CrossEntropyLoss(weight=cross_entropy_weights)

    return criterion, cross_entropy_weights


def configure_model(Y_train, input_size, hidden_size, output_size, learning_rate, device, batch_first=True):
    """
    Configures an RNN model for training.

    Parameters
    ----------
    input_size : int
        The number of expected features in the input.
    hidden_size : int
        The number of features in the hidden state.
    output_size : int
        The number of output features.
    learning_rate : float
        The learning rate.
    device : str
        The device to use for training.
    batch_first : bool, optional
        If True, then the input and output tensors are provided as (batch, seq, feature).
        Default is True.

    Returns
    -------
    model : torch.nn.Module
        The RNN model.
    criterion : torch.nn.modules.loss._Loss
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    cross_entropy_weights : torch.Tensor
        The weights for the CrossEntropyLoss function.
    """
    # Create the RNN model
    model = RNN(input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, batch_first=batch_first).to(device)

    # Apply the weight initialization
    model.apply(init_param_weights)

    # Configure loss function
    criterion, cross_entropy_weights = loss_function(Y_train)

    # Using SGD as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Using CosineAnnealingLR as the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    return model, criterion, optimizer, scheduler, cross_entropy_weights
