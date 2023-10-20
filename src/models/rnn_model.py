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


def train_rnn_model(X_train, Y_train, X_test, Y_test, input_size, hidden_size, output_size, num_epochs, batch_size, learning_rate, device, batch_first=True):
    # Create the dataset and data loader
    train_dataset = WalkDataset(X_train, Y_train)
    test_dataset = WalkDataset(X_test, Y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model, loss function, and optimizer
    model = RNN(input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, batch_first=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Debugging: Check for NaN or inf in loss
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                print(
                    f"Skipping iteration {i} of epoch {epoch} due to invalid loss {loss.item()}")
                continue

            loss.backward()

            # Debugging: Check for NaN or inf in gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_check = torch.sum(torch.isnan(
                        param.grad)) + torch.sum(torch.isinf(param.grad))
                    if grad_check > 0:
                        print(
                            f"Skipping iteration {i} of epoch {epoch} due to invalid gradient in {name}")

            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss = running_loss / len(test_loader)
        test_acc = correct / total

        print(
            f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    return model
