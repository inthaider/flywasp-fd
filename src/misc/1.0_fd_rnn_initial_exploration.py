"""
Old Jupyter notebook code by Faizan.

Not sure when this was written, but around early October.
"""

# %%
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


# %%
# Load in the dataframe
pickle_path = "../data/interim/"
# And you can read it back into memory like this:
df = pd.read_pickle(pickle_path + 'ff-mw.pkl')

df = df.drop('plot', axis=1)

cols = df.columns.tolist()

# Rearrange the column names
cols.insert(cols.index('F2Wdis') + 1, cols.pop(cols.index('F2Wdis_rate')))

# Reindex the DataFrame with the new column order
df = df[cols]

# Calculating the mean of the 'ANTdis_1', 'ANTdis_2 vars
df['ANTdis'] = df[['ANTdis_1', 'ANTdis_2']].mean(axis=1)

# Adding the label
# create new variable 'start_walk'
df['start_walk'] = ((df['walk_backwards'] == 1) & (
    df['walk_backwards'].shift(1) == 0)).astype(int)

# Only keeping in the relevant variables

df = df[['Frame', 'Fdis', 'FdisF', 'FdisL', 'Wdis', 'WdisF',
         'WdisL', 'Fangle', 'Wangle', 'F2Wdis', 'F2Wdis_rate', 'F2Wangle',
         'W2Fangle', 'ANTdis', 'F2W_blob_dis', 'bp_F_delta',
         'bp_W_delta', 'ap_F_delta', 'ap_W_delta', 'ant_W_delta', 'file', 'start_walk']]


# %%
#################
# Preprocessing #
#################

# Replacing infinity values with nan
df = df.replace([np.inf, -np.inf], np.nan)


# Assuming df is already loaded
# If not, uncomment the line below
# df = pd.read_csv('your_data.csv')

# Standardize the data
columns_to_scale = ['Fdis', 'FdisF', 'FdisL', 'Wdis', 'WdisF', 'WdisL', 'Fangle',
                    'Wangle', 'F2Wdis', 'F2Wdis_rate', 'F2Wangle', 'W2Fangle',
                    'ANTdis', 'F2W_blob_dis', 'bp_F_delta', 'bp_W_delta', 'ap_F_delta',
                    'ap_W_delta', 'ant_W_delta']

scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


# %%
# Reshape data into sequences
def create_sequences(data, sequence_length=3):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        # does iloc include the last index?
        # Answer: No, it does not
        x.append(data.iloc[i:i+sequence_length].values)
        y.append(data.iloc[i+sequence_length]['start_walk'])
        print(x, y)
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
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# %%
