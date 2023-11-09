# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:18:48 2023
DATE LAST MODIFIED: 2023-10-31

@author: Faizan

Latest version of FD's code for RNN implementation, for REFERENCE.
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

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df = df.reset_index(drop=True)

'''
def create_sequences(data, sequence_length=5):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i+sequence_length]
        target = data.iloc[i+sequence_length]['start_walk']
        
        # Check if there are any missing values in the sequence or target
        if sequence.isnull().values.any() or pd.isnull(target):
            continue
        
        x.append(sequence.values)
        y.append(target)
    
    return np.array(x), np.array(y)
'''

def create_sequences(data, sequence_length=5, start_index = 0):
    n = len(data) - sequence_length
    x = np.empty((n, sequence_length, data.shape[1]))
    y = np.empty(n)
    indices = np.empty(n, dtype=int)
    
    valid_idx = 0
    for i in range(n):
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length, -1]
        
        # Check if there are any missing values in the sequence or target
        if not np.isnan(sequence).any() and not np.isnan(target):
            x[valid_idx] = sequence
            y[valid_idx] = target
            indices[valid_idx] = start_index + i + sequence_length
            valid_idx += 1
            
    # Trim the arrays to the size of valid sequences
    x = x[:valid_idx]
    y = y[:valid_idx]
    indices = indices[:valid_idx]
    return x, y, indices

X_train, Y_train = [], []
X_test, Y_test = [], []
files = df['file'].unique()

test_indices = []
for i, file in enumerate(files):
    print(i, file)
    file_df = df[df['file'] == file]
    file_data = file_df.drop(['Frame', 'file'], axis=1).values
    
    # Create sequences for each file
    x, y, idx = create_sequences(file_data, start_index=file_df.index.min())
    
    # Calculate the split index
    train_size = int(len(x) * 2/3)
    
    # Split the sequences for each file
    X_train.extend(x[:train_size])
    Y_train.extend(y[:train_size])
    X_test.extend(x[train_size:])
    Y_test.extend(y[train_size:])
    
    # Extract the test indices corresponding to Y_test
    test_indices.extend(idx[train_size:])
    

X_train = np.array(X_train)
X_train = X_train[:, :, :-1]
Y_train = np.array(Y_train)
X_test = np.array(X_test)
X_test = X_test[:, :, :-1]
Y_test = np.array(Y_test)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#######################
# Random Oversampling #
#######################
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_train_resampled, Y_train_resampled = ros.fit_resample(X_train.reshape(X_train.shape[0], -1), Y_train)

# Reshape X_train back to its original shape
X_train_resampled = X_train_resampled.reshape(-1, 3, 19)

print("Original dataset shape:", X_train.shape, Y_train.shape)
print("Resampled dataset shape:", X_train_resampled.shape, Y_train_resampled.shape)

# Saving train and test datasets as pickle objects
# Define your directory path
directory = r'D:\Fly-Wasp\Data\Pickle Objects\Wild-Type\Train Test Data For ff-mw RNN'

# Ensure the directory exists or create it
import os
import pickle
if not os.path.exists(directory):
    os.makedirs(directory)

# List of objects and their names
data_objects = [('X_train', X_train), 
                ('X_test', X_test), 
                ('y_train', Y_train), 
                ('y_test', Y_test)]

# Save each object
for name, obj in data_objects:
    with open(os.path.join(directory, f'{name}.pkl'), 'wb') as file:
        pickle.dump(obj, file)

print("Data saved successfully!")

######################
# RNN Implementation #
######################

test_rnn_output = []
test_nnl_output = []
test_sigmoid_output = []

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

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        
        test_rnn_output.append(out.shape)
        
        out = self.fc(out[:, -1, :])
        
        test_nnl_output.append(out.shape)
        
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
from sklearn.metrics import f1_score, precision_recall_curve, auc
#all_preds = []
#all_labels = []
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            #print(outputs.data, torch.max(outputs.data,1))
            
            probabilities = F.softmax(outputs, dim=1)
            prob_of_class_1 = probabilities[:, 1]
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds_probs.extend(prob_of_class_1.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds)
    prec, rec, _ = precision_recall_curve(all_labels, all_preds_probs)
    pr_auc = auc(rec, prec)
    print("Training Data Distribution:")
    print(pd.Series(all_labels).value_counts().to_dict())
    
    print("Predicted Data Distribution:")
    print(pd.Series(all_preds).value_counts().to_dict())
    return running_loss / len(val_loader), correct / total, f1, pr_auc

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

# Calculating weights for class imbalance to pass to the loss function
class_counts = [pd.Series(Y_train).value_counts()[0], pd.Series(Y_train).value_counts()[1]]

# Compute class weights
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Train the model
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=50)
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc, test_f1, test_pr_auc = evaluate(model, test_loader, criterion, device)
    scheduler.step(test_loss)
    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test PR AUC: {test_pr_auc:.4f}')


# Plotting predicted probabilities
import torch.nn.functional as F
all_preds = []
all_preds_probs = []
all_labels = []
model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        #loss = criterion(outputs, labels)
        #running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()
        
        #print(outputs.data, torch.max(outputs.data,1))
        
        
        
        probabilities = F.softmax(outputs, dim=1)
        prob_of_class_1 = probabilities[:, 1]
        #print(prob_of_class_1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_preds_probs.extend(prob_of_class_1.cpu().numpy())


import pandas as pd

# Convert the lists to a DataFrame
merged_df = pd.DataFrame({
    'index': test_indices,
    'y_test': all_labels,
    'y_pred': all_preds_probs
})

plot_df = pd.merge(df, merged_df, left_index=True, right_on='index', how='right')

plot_df = plot_df.sort_values(by=["file", "Frame"])

########## Flagging frames that are 5 seconds prior of a backing event
def flag_frames(group):
    group = group.reset_index(drop=True)  # Reset index for proper index-based operations
    group['flag'] = 0  # Initialize 'flag' column with zeros
    start_indices = group.index[group['start_walk'] == 1]  # Indices where 'start_frame' equals 'Frame'
    for idx in start_indices:
        # Flag frames that are within a 40-frame value distance before the current frame
        mask = (group['Frame'] >= group.loc[idx, 'Frame'] - 200) & (group['Frame'] <= group.loc[idx, 'Frame'])
        group.loc[mask, 'flag'] = 1
    return group

# Apply the custom function to each file group
plot_df = plot_df.groupby('file').apply(flag_frames).reset_index(drop=True)

# Create a new column that marks when the flag changes
plot_df['flag_change'] = plot_df['flag'].diff().ne(0)

# Assign a unique group number to each change in flag
plot_df['group_number'] = plot_df['flag_change'].cumsum()

# Cleanup: remove the 'flag_change' column
plot_df.drop(columns=['flag_change'], inplace=True)

# First, create a mask where Frame equals start_frame
start_frame_mask = (plot_df['start_walk'] == 1)

# Then, for each group, get the value of Frame where Frame equals start_frame
plot_df['backing_frame'] = plot_df.loc[start_frame_mask, 'Frame']

# Forward fill 'backing_frame' within each group
#plot_df['backing_frame'] = plot_df.groupby('group_number')['backing_frame'].ffill()

# Backward fill 'backing_frame' within each group
plot_df['backing_frame'] = plot_df.groupby('group_number')['backing_frame'].bfill()

# Calculate delta frmes
plot_df['delta_frames'] = plot_df['Frame'] - plot_df['backing_frame']


# Plot of predicted probabilities 1 second before and after a true backing event
plot_df_2 = plot_df[plot_df['flag']==1]
plot_df_2['delta_frames'] = plot_df_2['Frame'] - plot_df_2['backing_frame']
plot_df_2 = plot_df_2[['file', 'Frame', 'start_walk', 'backing_frame', 'delta_frames', 'y_test', 'y_pred']]


mean_df = plot_df_2.groupby('delta_frames')[['y_pred']].mean().reset_index()

# Plot of predictions within 200 frames (5 seconds
# Assuming mean_df is your DataFrame and it has columns 'delta_frames' and 'y_pred'
plt.figure(figsize=(10,6))  # Create a new figure with a specific size (optional)
plt.plot(mean_df['delta_frames'], mean_df['y_pred'])  # Plot 'y_pred' against 'delta_frames'

# Optional: add labels and title
plt.xlabel('Delta Frames')
plt.ylabel('y_pred')
plt.title('y_pred vs Delta Frames')

plt.show()  # Display the plot)



### Debugging









