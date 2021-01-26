import pandas as pd
import numpy as np
import seaborn as sb

from pandas.plotting import scatter_matrix
from seaborn import scatterplot
from seaborn import lmplot, stripplot
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)


# Setting display options for pandas
desired_width = 200
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

# Read in the csv file to a DataFrame
train_df = pd.read_csv("../datasets/prem_final_datasets/prem_training_set.csv")

train_df = train_df.sample(frac=1, random_state=2)
train_df.reset_index(drop=True, inplace=True)

train_df = train_df.drop(["Date", "Home", "Score", "Away", "Venue", "MP_Home", "MP_Away"], axis=1)

unaltered_df = train_df.copy()  # The unaltered dataset

per_90_features = ["GlsPer90_Home", "AstPer90_Home", "Sh/90_Home", "SoT/90_Home", "GA90_Home",
                   "GlsPer90_Away", "AstPer90_Away", "Sh/90_Away", "SoT/90_Away", "GA90_Away",
                   "GlsPer90_2_Home", "AstPer90_2_Home", "GlsPer90_3_Home", "AstPer90_3_Home", "GlsPer90_4_Home",
                   "AstPer90_4_Home", "GlsPer90_5_Home", "AstPer90_5_Home", "GlsPer90_6_Home", "AstPer90_6_Home",
                   "GlsPer90_7_Home", "AstPer90_7_Home", "GlsPer90_8_Home", "AstPer90_8_Home",
                   "GlsPer90_9_Home", "AstPer90_9_Home", "GlsPer90_10_Home", "AstPer90_10_Home",
                   "GlsPer90_11_Home", "AstPer90_11_Home", "GlsPer90_2_Away", "AstPer90_2_Away",
                   "GlsPer90_3_Away", "AstPer90_3_Away", "GlsPer90_4_Away", "AstPer90_4_Away",
                   "GlsPer90_5_Away", "AstPer90_5_Away", "GlsPer90_6_Away", "AstPer90_6_Away",
                   "GlsPer90_7_Away", "AstPer90_7_Away", "GlsPer90_8_Away", "AstPer90_8_Away",
                   "GlsPer90_9_Away", "AstPer90_9_Away", "GlsPer90_10_Away", "AstPer90_10_Away",
                   "GlsPer90_11_Away", "AstPer90_11_Away"]

train_df = train_df.drop(per_90_features, axis=1)

redundant_features = ["CrdR_Home", "D_Away", "CrdR_Away", "Sh_Away", "MP_2_Home", "Min_2_Home", "MP_4_Home",
                      "Min_4_Home", "MP_5_Home", "Min_5_Home", "MP_6_Home", "Min_6_Home", "MP_8_Home", "Min_8_Home",
                      "MP_11_Home", "Min_11_Home", "MP_2_Away", "Min_2_Away"]

train_df = train_df.drop(redundant_features, axis=1)

train_df = train_df.sample(frac=1, random_state=2)
train_df.reset_index(drop=True, inplace=True)

feature_sel_df = train_df.copy()   # The dataset after feature selection

label_encoder = LabelEncoder()
label_encoder.fit(train_df["Result"])
unaltered_df["Result"] = label_encoder.transform(unaltered_df["Result"])
feature_sel_df["Result"] = label_encoder.transform(feature_sel_df["Result"])


train_df["CS_per_W_Home"] = train_df["CS_Home"]/train_df["W_Home"]
train_df["CS_per_L_Away"] = train_df["CS_Away"]/train_df["L_Away"]
train_df["Gls_per_W_Home"] = train_df["Gls_Home"] / train_df["W_Home"]
train_df["Gls_per_L_Away"] = train_df["Gls_Away"] / train_df["L_Away"]
train_df["Gls_per_SoT_per_W_Home"] = train_df["G/SoT_Home"] / train_df["W_Home"]
train_df["Gls_per_SoT_per_W_Away"] = train_df["G/SoT_Away"] / train_df["W_Away"]
train_df["SoT_per_W_Home"] = train_df["SoT_Home"] / train_df["W_Home"]
train_df["SoT_per_L_Away"] = train_df["SoT_Away"] / train_df["L_Away"]

train_df = train_df.sample(frac=1, random_state=2)
train_df.reset_index(drop=True, inplace=True)

feature_eng_df = train_df.copy()    # dataset to be used for feature engineering
feature_eng_df["Result"] = label_encoder.transform(feature_eng_df["Result"])


#pd.plotting.scatter_matrix(feature_eng_df, alpha=0.2)

''' Diagonal Correlation Matrix
# Dropped features for reduced diagonal correlation matrix
diagonal_matrix_features = ["Result", "W_Home", "D_Home", "L_Home", "Pts_Home", "Gls_Home", "Ast_Home", "SoT_Home",
                            "GA_Home", "SoTA_Home", "W_Away", "L_Away", "Gls_Away", "Gls_2_Home", "Ast_5_Home",
                            "Gls_9_Home", "Ast_2_Away", "CS_per_W_Home", "Gls_per_W_Home",
                            "Gls_per_SoT_per_W_Away", "SoT_per_W_Home"]


corr_matrix_df = feature_eng_df.copy()  # Another copy of the dataset to allow for a reduction for display purposes
corr_matrix_df = corr_matrix_df[diagonal_matrix_features]
print(corr_matrix_df.corr())
corr = corr_matrix_df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 11))
# Generate a custom diverging colormap
cmap = sb.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sb.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=.5)

plt.show()
'''

unaltered_features = unaltered_df.columns.tolist()
del unaltered_features[1]
sel_features = feature_sel_df.columns.tolist()
del sel_features[1]
features = train_df.columns.tolist()
del features[1]


unaltered_X = unaltered_df[unaltered_features]
unaltered_y = unaltered_df["Result"]
feature_sel_X = feature_sel_df[sel_features]
feature_sel_y = feature_sel_df["Result"]
train_X = train_df[features]
train_y = train_df["Result"]


# Class for testing multiple scalers with grid search
class TransformerFromHyperP(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=None):
        self.transformer = transformer

    def fit(self, X, y=None):
        if self.transformer:
            self.transformer.fit(X, y)
        return self

    def transform(self, X, y=None):
        if self.transformer:
            return self.transformer.transform(X)
        else:
            return X


# Using holdout with stratification
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=2)   # Split for validation set, 60-20-20 overall


print("----UNALTERED----")
unaltered_preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", TransformerFromHyperP())]),
         unaltered_features)],
        remainder="passthrough")

unaltered_knn = Pipeline([
    ("preprocessor", unaltered_preprocessor),
    ("predictor", KNeighborsClassifier())])

# Dictionary for the hyperparameters, as well as the new features. I also included 3 scalers.
unaltered_knn_param_grid = {"predictor__n_neighbors": [33, 34, 35, 36],
                            "preprocessor__num__scaler__transformer": [StandardScaler(), MinMaxScaler(), RobustScaler()],
                            }
# This dataset is quiet large, so holdout has been used.
knn_gs = GridSearchCV(unaltered_knn, unaltered_knn_param_grid, scoring="accuracy", cv=sss)
knn_gs.fit(unaltered_X, unaltered_y)
print("KNN BEST PARAMS/SCORE: ", knn_gs.best_params_, knn_gs.best_score_)
# Checking for under/overfitting
unaltered_knn.set_params(**knn_gs.best_params_)
scores = cross_validate(unaltered_knn, unaltered_X, unaltered_y, cv=sss, scoring="accuracy", return_train_score=True)
print("KNN Training Accuracy: ", np.mean(scores["train_score"]))
print("KNN Validation Accuracy: ", np.mean(scores["test_score"]))


print("----FEATURE_SELECTION----")
sel_preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", TransformerFromHyperP())]),
         sel_features)],
        remainder="passthrough")

sel_knn = Pipeline([
    ("preprocessor", sel_preprocessor),
    ("predictor", KNeighborsClassifier())])

# Dictionary for the hyperparameters, as well as the new features. I also included 3 scalers.
selection_knn_param_grid = {"predictor__n_neighbors": [37, 38, 39],
                            "preprocessor__num__scaler__transformer": [StandardScaler(), MinMaxScaler(), RobustScaler()],
                            }
# This dataset is quiet large, so holdout has been used.
knn_gs = GridSearchCV(sel_knn, selection_knn_param_grid, scoring="accuracy", cv=sss)
knn_gs.fit(feature_sel_X, feature_sel_y)
print("KNN BEST PARAMS/SCORE: ", knn_gs.best_params_, knn_gs.best_score_)
# Checking for under/overfitting
sel_knn.set_params(**knn_gs.best_params_)
scores = cross_validate(sel_knn, feature_sel_X, feature_sel_y, cv=sss, scoring="accuracy", return_train_score=True)
print("KNN Training Accuracy: ", np.mean(scores["train_score"]))
print("KNN Validation Accuracy: ", np.mean(scores["test_score"]))


print("----FINAL----")
final_preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", TransformerFromHyperP())]),
         features)],
        remainder="passthrough")

final_knn = Pipeline([
    ("preprocessor", final_preprocessor),
    ("predictor", KNeighborsClassifier())])

# Dictionary for the hyperparameters, as well as the new features. I also included 3 scalers.
final_knn_param_grid = {"predictor__n_neighbors": [29, 30, 31],
                        "preprocessor__num__scaler__transformer": [StandardScaler(), MinMaxScaler(), RobustScaler()],
                        }
# This dataset is quiet large, so holdout has been used.
knn_gs = GridSearchCV(final_knn, final_knn_param_grid, scoring="accuracy", cv=sss)
knn_gs.fit(train_X, train_y)
print("KNN BEST PARAMS/SCORE: ", knn_gs.best_params_, knn_gs.best_score_)
# Checking for under/overfitting
final_knn.set_params(**knn_gs.best_params_)
scores = cross_validate(final_knn, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
print("KNN Training Accuracy: ", np.mean(scores["train_score"]))
print("KNN Validation Accuracy: ", np.mean(scores["test_score"]))


