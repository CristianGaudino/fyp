from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class InsertDivideFeature(BaseEstimator, TransformerMixin):
    def __init__(self, new_feature_name, feature_one, feature_two, insert=True):
        self.insert = insert
        self.new_feature_name = new_feature_name
        self.feature_one = feature_one
        self.feature_two = feature_two

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.insert:
            X[self.new_feature_name] = X[self.feature_one] / X[self.feature_two]
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.replace([np.nan], 0)
        return X


'''
train_df["CS_per_W_Home"] = train_df["CS_Home"]/train_df["W_Home"]
train_df["CS_per_L_Away"] = train_df["CS_Away"]/train_df["L_Away"]
train_df["Gls_per_W_Home"] = train_df["Gls_Home"] / train_df["W_Home"]
train_df["Gls_per_L_Away"] = train_df["Gls_Away"] / train_df["L_Away"]
train_df["Gls_per_SoT_per_W_Home"] = train_df["G/SoT_Home"] / train_df["W_Home"]
train_df["Gls_per_SoT_per_W_Away"] = train_df["G/SoT_Away"] / train_df["W_Away"]
train_df["SoT_per_W_Home"] = train_df["SoT_Home"] / train_df["W_Home"]
train_df["SoT_per_L_Away"] = train_df["SoT_Away"] / train_df["L_Away"]
'''

