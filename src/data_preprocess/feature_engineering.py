from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self, df=None):
        self.df = df
        self.scaler = StandardScaler()

    def standardize_features(self, columns_to_scale):
        self.df[columns_to_scale] = self.scaler.fit_transform(
            self.df[columns_to_scale])
