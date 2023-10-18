import numpy as np


def create_sequences(data, sequence_length=3):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data.iloc[i:i+sequence_length].values)
        y.append(data.iloc[i+sequence_length]['start_walk'])
    return np.array(x), np.array(y)


def prepare_train_test_sequences(df, sequence_length=3, split_ratio=2/3):
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    files = df['file'].unique()

    for file in files:
        file_data = df[df['file'] == file].drop(['Frame', 'file'], axis=1)

        # Create sequences for each file
        x, y = create_sequences(file_data, sequence_length=sequence_length)

        # Calculate the split index
        train_size = int(len(x) * split_ratio)

        # Split the sequences for each file
        X_train.extend(x[:train_size])
        Y_train.extend(y[:train_size])
        X_test.extend(x[train_size:])
        Y_test.extend(y[train_size:])

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
