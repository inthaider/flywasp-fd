import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, df=None, pickle_path=None):
        self.df = df
        self.pickle_path = pickle_path

    def load_data(self):
        if self.pickle_path:
            self.df = pd.read_pickle(self.pickle_path)
            return self.df
        else:
            raise ValueError("Provide a valid pickle path.")

    def drop_columns(self, columns_to_drop):
        self.df.drop(columns_to_drop, axis=1, inplace=True)

    def specific_rearrange(self, col_to_move, ref_col):
        cols = self.df.columns.tolist()
        cols.insert(cols.index(ref_col) + 1, cols.pop(cols.index(col_to_move)))
        self.df = self.df[cols]

    def rearrange_columns(self, cols_order):
        self.df = self.df[cols_order]

    def calculate_means(self, column_pairs, new_columns):
        for pair, new_col in zip(column_pairs, new_columns):
            self.df[new_col] = self.df[pair].mean(axis=1)

    def add_labels(self, condition_columns, new_column):
        self.df[new_column] = ((self.df[condition_columns[0]] == 1) & (
            self.df[condition_columns[1]].shift(1) == 0)).astype(int)

    def handle_infinity_and_na(self):
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
