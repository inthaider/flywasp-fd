# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:18:48 2023

@author: Faizan
"""

# Load packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
# from docx import Document
# from docx.shared import Inches
from datetime import datetime
import time

## Load in the dataframe
pickle_path = r"D:\Fly-Wasp\Data\Pickle Objects\Wild-Type\\"

# And you can read it back into memory like this:
df = pd.read_pickle(pickle_path + 'csmh_f_X_LH_m_Z_26.pkl')
df = df.drop('plot', axis = 1)

cols = df.columns.tolist()

# Rearrange the column names
cols.insert(cols.index('F2Wdis') + 1, cols.pop(cols.index('F2Wdis_rate')))

# Reindex the DataFrame with the new column order
df = df[cols]

# Calculating the mean of the 'ANTdis_1', 'ANTdis_2 vars
df['ANTdis'] = df[['ANTdis_1', 'ANTdis_2']].mean(axis=1)

# Adding the label
# create new variable 'start_walk'
df['start_walk'] = ((df['walk_backwards'] == 1) & (df['walk_backwards'].shift(1) == 0)).astype(int)

# Only keeping in the relevant variables

df = df[['Frame', 'Fdis', 'FdisF', 'FdisL', 'Wdis', 'WdisF',
       'WdisL', 'Fangle', 'Wangle', 'F2Wdis', 'F2Wdis_rate', 'F2Wangle',
       'W2Fangle', 'ANTdis', 'F2W_blob_dis', 'bp_F_delta',
       'bp_W_delta', 'ap_F_delta', 'ap_W_delta', 'ant_W_delta', 'file', 'start_walk']]

#################
# Preprocessing #
#################

def create_sequences(data, sequence_length=2):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        # does iloc include the last index?
        # Answer: No, it does not
        x.append(data.iloc[i:i+sequence_length].values)
        y.append(data.iloc[i+sequence_length]['start_walk'])
        #print(x, y)
    return np.array(x), np.array(y)

X_train, Y_train = [], []
X_test, Y_test = [], []
files = df['file'].unique()

for file in files:
    file_data = df[df['file'] == file].drop(['Frame', 'file'], axis=1)
    
    # Create sequences for each file
    x, y = create_sequences(file_data)
    
    # Calculate the split index
    train_size = int(len(x) * 2/3)
    
    # Split the sequences for each file
    X_train.extend(x[:train_size])
    Y_train.extend(y[:train_size])
    X_test.extend(x[train_size:])
    Y_test.extend(y[train_size:])

X_train = np.array(X_train)
X_train = X_train[:, :, :-1]
Y_train = np.array(Y_train)
X_test = np.array(X_test)
X_test = X_test[:, :, :-1]
Y_test = np.array(Y_test)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)



######################
# RNN Implementation #
######################


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
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
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Define the evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(val_loader), correct / total

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


############ Training Process ######################
# Create the dataset and data loader
train_dataset = WalkDataset(X_train, Y_train)
test_dataset = WalkDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model, loss function, and optimizer
model = RNN(input_size=X_train.shape[2], hidden_size=64, output_size=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')











