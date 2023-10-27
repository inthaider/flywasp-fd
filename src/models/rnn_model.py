import hashlib
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np  # Added for debugging
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


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
        int
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
        tuple
            A tuple containing the input and target data.
        """
        # Return a tuple containing the input and target data at the given index
        return self.X[idx], self.Y[idx]


def init_weights(m):
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
    torch.utils.data.DataLoader
        The training data loader.
    torch.utils.data.DataLoader
        The testing data loader.
    """
    train_dataset = WalkDataset(X_train, Y_train)
    test_dataset = WalkDataset(X_test, Y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def configure_model(input_size, hidden_size, output_size, learning_rate, device, batch_first=True):
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
    tuple
        A tuple containing the configured RNN model, the CrossEntropyLoss criterion, and the SGD optimizer.
    """
    model = RNN(input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, batch_first=batch_first).to(device)
    # Apply the weight initialization
    model.apply(init_weights)
    # Using CrossEntropyLoss as the loss function
    criterion = nn.CrossEntropyLoss()
    # Using SGD as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def train_loop(model, batch_size, device, prints_per_epoch, train_loader, criterion, optimizer, epoch):
    """
    Trains an RNN model on a training dataset for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The RNN model to train.
    batch_size : int
        The batch size.
    device : str
        The device to use for training.
    prints_per_epoch : int
        The number of times to print the loss per epoch.
    train_loader : torch.utils.data.DataLoader
        The training data loader.
    criterion : torch.nn.modules.loss._Loss
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    epoch : int
        The current epoch number.

    Returns
    -------
    float
        The average training loss over all batches.
    """
    size_train = len(train_loader.dataset)  # Number of training samples
    num_batches = len(train_loader)  # Number of batches
    # Print loss prints_per_epoch times per epoch
    # print_interval = num_batches // prints_per_epoch
    print_interval = int(max(num_batches // prints_per_epoch, 1))
    # --------------------#
    print(f"Print interval: {print_interval}")  # Debugging line
    # --------------------#
    # Set the model to training mode - important for batch normalization and dropout layers
    # This is best practice, but is it necessary here in this situation?
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

        # Note that i is the index of the batch and goes up to num_batches - 1
        inputs, labels = inputs.to(device), labels.to(
            device)  # Move tensors to device, e.g. GPU
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
        debug_sumsq_grad_param(model, sum_sq_gradients, sum_sq_parameters)

        # Get predicted class
        # The line below is basically taking the outputs tensor, which has shape (batch_size, 2), and getting the index of the maximum value in each row (i.e. the predicted class) and returning a tensor of shape (batch_size, 1)
        _, predicted = torch.max(outputs.data, 1)
        # Accumulate true and predicted labels for F1 score calculation
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

        # Print loss every print_interval iterations
        if int(i) % print_interval == 0:
            loss, current_iter = loss.item(), (i + 1) * len(inputs)  # loss and current iteration
            print(
                f"Loss: {loss:>7f}  [{current_iter:>5d}/{size_train:>5d}]")

    # Log sum of squared gradients and parameters after each epoch
    logging.info(
        f"\nSum squared grads/params in Epoch {epoch+1}:\n"
        f"\tSum of squared gradients : {sum_sq_gradients:>12.4f}\n"
        f"\tSum of squared parameters: {sum_sq_parameters:>12.4f}"
    )
    # Calculate average loss over all batches
    train_loss = running_loss / len(train_loader)
    # Calculate F1 score for training data
    train_f1 = f1_score(true_labels, pred_labels)

    print(f"\nTrain Performance: \n Avg loss: {train_loss:>8f}, F1 Score: {train_f1:.4f} \n")

    return train_loss, train_f1


def test_loop(model, device, test_loader, criterion):
    """
    Evaluates the performance of a trained RNN model on a test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The trained RNN model.
    device : str
        The device to use for evaluation.
    test_loader : torch.utils.data.DataLoader
        The test data loader.
    criterion : torch.nn.modules.loss._Loss
        The loss function.

    Returns
    -------
    float
        The average test loss over all batches.
    float
        The test accuracy.
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # This is best practice, but is it necessary here in this situation?
    model.eval()
    # Initialize running loss
    running_loss = 0.0
    correct = 0
    total = 0
    # Initialize true and predicted labels for F1 score calculation
    true_labels = []
    pred_labels = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(
                device)  # Move tensors to device, e.g. GPU
            
            outputs = model(inputs)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss
            running_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(
                outputs.data, 1)  # Get predicted class
            
            total += labels.size(0)  # Accumulate total number of samples
            # Accumulate number of correct predictions
            correct += (predicted == labels).sum().item()
            
            # Accumulate true and predicted labels for F1 score calculation
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # Calculate average loss and accuracy over all batches
    test_loss = running_loss / len(test_loader)
    test_acc = correct / total
    # Calculate F1 score for testing data
    test_f1 = f1_score(true_labels, pred_labels, average='macro')

    print(
        f"Test Performance: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}, F1 Score: {test_f1:.4f} \n")

    return test_loss, test_acc, test_f1


def train_rnn_model(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate, device, batch_first=True, prints_per_epoch=10):
    """
    Trains the RNN model.

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
    input_size : int
        The number of expected features in the input.
    hidden_size : int
        The number of features in the hidden state.
    output_size : int
        The number of output features.
    num_epochs : int
        The number of epochs to train the model.
    batch_size : int
        The batch size.
    learning_rate : float
        The learning rate.
    device : str
        The device to use for training.
    batch_first : bool, optional
        If True, then the input and output tensors are provided as (batch, seq, feature).
        Default is True.
    prints_per_epoch : int, optional
        The number of times to print the loss per epoch. Default is 10.

    Returns
    -------
    torch.nn.Module
        The trained RNN model.
    """
    # Create the data loaders
    train_loader, test_loader = data_loaders(
        X_train, Y_train, X_test, Y_test, batch_size)
    # Define the model, loss function, and optimizer
    model, criterion, optimizer = configure_model(
        input_size, hidden_size, output_size, learning_rate, device, batch_first)
    # Create a SummaryWriter for logging to TensorBoard
    writer = SummaryWriter()

    ################################
    # Train and evaluate the model #
    ################################
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n-------------------------------")
        # Train the model
        train_loss, train_f1 = train_loop(model, batch_size, device,
                                prints_per_epoch, train_loader, criterion, optimizer, epoch)

        # Evaluate/Test the model
        test_loss, test_acc, test_f1 = test_loop(model, device, test_loader, criterion)

        # print(
        #     f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        # Log loss and accuracy to TensorBoard
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train F1 Score', train_f1, epoch)
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', test_acc, epoch)
        writer.add_scalar('Test F1 Score', test_f1, epoch)

        print(f"\n Epoch {epoch+1} Metrics -- Train Loss: {train_loss:.4f}, Train F1 Score: {train_f1:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    # Close the SummaryWriter
    writer.close()

    return model


def debug_input_nan_inf(inputs):
    """
    Checks for NaN and inf values in the input tensor.

    Parameters
    ----------
    inputs : torch.Tensor
        The input tensor.

    Returns
    -------
    tuple
        A tuple containing the positions of NaN and inf values in the input tensor.
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

    Returns
    -------
    None
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

    Returns
    -------
    None
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

    Returns
    -------
    None
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
