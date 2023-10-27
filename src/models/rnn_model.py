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
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Define the RNN model


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_first=True):
        self.timestamp = datetime.now().strftime("%Y%m%d")

        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Define the dataset class


class WalkDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# For weight initialization


def init_weights(m):
    """Optional function for weight initialization.

    Uses Xavier uniform initialization for weights and constant initialization
    for biases.

    Parameters
    ----------
    m : torch.nn.Module
        The module to initialize. Only applies to Linear layers.
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# Define the training function

def train_rnn_model(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate, device, batch_first=True, prints_per_epoch=10):
    # Create the dataset and data loader
    train_dataset = WalkDataset(X_train, Y_train)
    test_dataset = WalkDataset(X_test, Y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model, loss function, and optimizer
    model = RNN(input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, batch_first=batch_first).to(device)
    # Apply the weight initialization
    model.apply(init_weights)
    # Using CrossEntropyLoss as the loss function
    criterion = nn.CrossEntropyLoss()
    # Using SGD as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Create a SummaryWriter for logging to TensorBoard
    writer = SummaryWriter()

    ################################
    # Train and evaluate the model #
    ################################
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n-------------------------------")
        #################
        # Training loop #
        #################
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

        print(f"Number of batches: {num_batches}")  # Print number of batches
        print(f"Batch size: {batch_size}")  # Print batch size
        for i, (inputs, labels) in enumerate(train_loader):
            # Debugging: Check for NaN or inf in inputs
            nan_positions = torch.nonzero(torch.isnan(inputs), as_tuple=True)
            inf_positions = torch.nonzero(torch.isinf(inputs), as_tuple=True)

            assert not torch.isnan(inputs).any(
            ), f"NaN values found at positions {nan_positions}"
            assert not torch.isinf(inputs).any(
            ), f"inf values found at positions {inf_positions}"

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

                    # # Print loss every print_interval iterations
                    # if int(i) % print_interval == 0:
                    #     loss, current_iter = loss.item(), (i + 1) * len(inputs)  # loss and current iteration
                    #     print(
                    #         f"Loss: {loss:>7f}  [{current_iter:>5d}/{size_train:>5d}]")

            # Debugging: Check for NaN or inf in gradients
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

            # Debugging: Monitor sum of squared gradients and parameters
            for name, param in model.named_parameters():
                if param.grad is not None:
                    sum_sq_gradients += torch.sum(param.grad ** 2).item()
            for name, param in model.named_parameters():
                if param.data is not None:
                    sum_sq_parameters += torch.sum(param.data ** 2).item()

            # # Testing print loss stuff
            # if i % 5000:
            #     print("\n\t\tTESTING PRINT LOSS STUFF")
            #     print(f"\t\ti = {i}")
            #     print(f"\t\tprint_interval * 2 = {print_interval * 2}")
            #     print(f"\t\ti % print_interval = {i % print_interval}\n")
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
        print(f"\nTrain Error: \n Avg loss: {train_loss:>8f}")

        #############
        # Test loop #
        #############
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # This is best practice, but is it necessary here in this situation?
        model.eval()
        # Initialize running loss & sum of squared gradients and parameters
        running_loss = 0.0
        correct = 0
        total = 0
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

        # Calculate average loss and accuracy over all batches
        test_loss = running_loss / len(test_loader)
        test_acc = correct / total

        print(
            f"Test Error: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        # print(
        #     f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

        # Log loss and accuracy to TensorBoard
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', test_acc, epoch)

    # Close the SummaryWriter
    writer.close()

    return model


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
