import logging

import numpy as np  # Added for debugging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define the RNN model


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_first=True):
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
    criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss as the loss function
    # Using SGD as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Flags to control logging
    log_invalid_loss = True
    log_invalid_grad = True

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
        print_interval = num_batches // prints_per_epoch  # Print loss prints_per_epoch times per epoch
        
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
                        f"First occurrence of invalid loss {loss.item()} at iteration {i} of epoch {epoch}. Further warnings will be suppressed.")
                    log_invalid_loss = False
                continue

            # Debugging: Check for NaN or inf in gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_check = torch.sum(torch.isnan(
                        param.grad)) + torch.sum(torch.isinf(param.grad))
                    if grad_check > 0:
                        if log_invalid_grad:
                            logging.warning(
                                f"First occurrence of invalid gradient in {name} at iteration {i} of epoch {epoch}. Further warnings will be suppressed.")
                            log_invalid_grad = False

            # Debugging: Monitor sum of squared gradients and parameters
            for name, param in model.named_parameters():
                if param.grad is not None:
                    sum_sq_gradients += torch.sum(param.grad ** 2).item()
            for name, param in model.named_parameters():
                if param.data is not None:
                    sum_sq_parameters += torch.sum(param.data ** 2).item()

            # Print loss every print_freq iterations
            if i % print_interval == 0:
                loss, current_iter = loss.item(), (i + 1) * len(inputs)  # loss and current iteration
                print(
                    f"Loss: {loss:>7f}  [{current_iter:>5d}/{size_train:>5d}]")

        # Log sum of squared gradients and parameters after each epoch
        logging.info(
            f"Epoch {epoch+1}: Sum of squared gradients: {sum_sq_gradients:.4f}, Sum of squared parameters: {sum_sq_parameters:.4f}")

        # Calculate average loss over all batches
        train_loss = running_loss / len(train_loader)
        print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")

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
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        # print(
        #     f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    return model
