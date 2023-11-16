"""
This module contains the implementation of a Recurrent Neural Network
(RNN) model for time series prediction.

It includes the `RNN` class for the model, the `WalkDataset` class for
the dataset, and several helper functions for initializing the weights
of the model parameters, loading the training and testing data into
PyTorch DataLoader objects, initializing the weights for the
CrossEntropyLoss function, and configuring the model for training.

Classes:
    RNN: A class representing the RNN model.
    WalkDataset: A class representing the dataset.

Functions:
    init_param_weights(m): Initializes the weights of the model
        parameters.
    data_loaders(X_train, Y_train, X_test, Y_test, batch_size): Loads
        the training and testing data into PyTorch DataLoader objects.
    _init_cross_entropy_weights(Y_train): Initializes the weights for
        the CrossEntropyLoss function.
    loss_function(Y_train): Returns the CrossEntropyLoss function with
        weights.
    configure_model(
        Y_train, input_size, hidden_size, output_size,
        learning_rate, device, batch_first=True
    ): Configures an RNN model for training.
"""

import logging
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class RNN(nn.Module):
    """
    A class representing the RNN model.

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        output_size (int): The number of output features.
        batch_first (bool, optional): If True, then the input and output
            tensors are provided as (batch, seq, feature). Default is
            True.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        nonlinearity="tanh",
        batch_first=True,
    ):
        """
        Initialize the RNN model.

        Args:
            input_size (int): The number of expected features in the
                input.
            hidden_size (int): The number of features in the hidden
                state.
            output_size (int): The number of output features.
            num_layers (int): The number of RNN layers.
            nonlinearity (str, optional): The non-linearity to use.
                Default is "tanh".
            batch_first (bool, optional): If True, then the input and
                output tensors are provided as (batch, seq, feature).
                Default is True.
        """
        # Get the current timestamp as a string in the format YYYYMMDD
        self.timestamp = datetime.now().strftime("%Y%m%d")
        # Call the __init__ method of the parent class (nn.Module)
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create an RNN layer with the specified hyperparameters
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=batch_first,
        )
        # Create a fully connected linear layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Create sigmoid activation

    def forward(self, x):
        """
        Forward pass of the RNN model.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output data.

        Notes:
            Using `.to(x.device)` in the `forward()` method ensures that
            the model is moved to the same device as the input data.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            x.device
        )  # initial hidden state
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x, h0)
        # decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class WalkDataset(Dataset):
    """
    A class representing the dataset.

    Args:
        X (numpy.ndarray): The input data.
        Y (numpy.ndarray): The target data.
    """

    def __init__(self, X, Y):
        """
        Initialize the dataset.

        Args:
            X (numpy.ndarray): The input data.
            Y (numpy.ndarray): The target data.
            device (torch.device): The device to move the tensors to.
        """
        # Convert the input and target data to PyTorch tensors
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        # Return the length of the input data
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            tuple: A tuple containing the input data at the given index
                and the target data at the given index.
        """
        # Return a tuple containing the input and target data at the
        #  given index
        return self.X[idx], self.Y[idx]


def data_loaders(
    X_train, Y_train, X_test, Y_test, batch_size, device, **kwargs
):
    """
    Loads the training and testing data into PyTorch DataLoader objects.

    Args:
        X_train (numpy.ndarray): The training input data.
        Y_train (numpy.ndarray): The training target data.
        X_test (numpy.ndarray): The testing input data.
        Y_test (numpy.ndarray): The testing target data.
        batch_size (int): The batch size.
        device (torch.device): The device to move the tensors to.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: A tuple containing the training data loader and the
            testing data loader.
    """
    print(f"\n\n{device.type}")
    if device.type == "mps":
        print("Using kwargs for data loaders\n\n")
    kwargs = kwargs if device.type == "mps" else {}
    train_dataset = WalkDataset(X_train, Y_train)
    test_dataset = WalkDataset(X_test, Y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    return train_loader, test_loader


def configure_model(
    Y_train,
    input_size: int,
    hidden_size: int,
    output_size: int,
    learning_rate,
    num_hidden_layers: int,
    nonlinearity,
    device: str | torch.device = "cpu",
    batch_first=True,
):
    """
    Configures an RNN model for training.

    Args:
        Y_train (numpy.ndarray): The training target data.
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        output_size (int): The number of output features.
        learning_rate (float): The learning rate.
        num_hidden_layers (int): The number of hidden layers.
        nonlinearity (str): The non-linearity to use.
        device (str): The device to use for training.
        batch_first (bool, optional): If True, then the input and output
            tensors are provided as (batch, seq, feature). Default is
            True.

    Returns:
        tuple: A tuple containing the RNN model, the loss function, the
            optimizer, the learning rate scheduler, and the weights for
            the CrossEntropyLoss function.
    """
    # Create the RNN model
    model = RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_hidden_layers,
        nonlinearity=nonlinearity,
        batch_first=batch_first,
    ).to(device)

    # Apply the weight initialization
    model = (model.apply(init_param_weights)).to(device)

    # Configure loss function
    criterion, cross_entropy_weights = loss_function(Y_train, device)

    # Using SGD as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Using CosineAnnealingLR as the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    return model, criterion, optimizer, scheduler, cross_entropy_weights


def init_param_weights(m):
    """
    Optional function for weight initialization.

    Uses Xavier uniform initialization for weights and constant
    initialization for biases.

    Args:
        m (torch.nn.Module): The module to initialize. Only applies to
        Linear layers.
    """
    # If the module is a linear layer, initialize the weights with
    #  Xavier uniform initialization and the biases with a constant
    #  value of 0.01
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def loss_function(Y_train, device):
    """
    Initialize the loss function.

    Args:
        Y_train (numpy.ndarray): The training target data.

    Returns:
        tuple: A tuple containing the loss function and the weights for
            the cross entropy loss function.
    """
    # Calculating weights for class imbalance to pass to the loss
    # function
    cross_entropy_weights = _init_cross_entropy_weights(Y_train)

    # Using CrossEntropyLoss as the loss function with weights
    criterion = nn.CrossEntropyLoss(weight=cross_entropy_weights).to(device)

    return criterion, cross_entropy_weights


def _init_cross_entropy_weights(Y_train):
    """
    Initialize the weights for the cross entropy loss function.

    Args:
        Y_train (numpy.ndarray): The training target data.

    Returns:
        torch.Tensor: The weights for the cross entropy loss function.
    """
    class_counts = [
        pd.Series(Y_train).value_counts()[0],
        pd.Series(Y_train).value_counts()[1],
    ]

    # Compute class weights
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    return weights
