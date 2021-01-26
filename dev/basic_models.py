import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
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
pd.set_option('display.max_columns', 12)

# Read in the csv file to a DataFrame
df = pd.read_csv("../datasets/prem_final_datasets/prem_whole_set.csv")
train_df = pd.read_csv("../datasets/prem_final_datasets/prem_training_set.csv")
test_df = pd.read_csv("../datasets/prem_final_datasets/prem_test_set.csv")

df = df.sample(frac=1, random_state=2)
df.reset_index(drop=True, inplace=True)
train_df = train_df.sample(frac=1, random_state=2)
train_df.reset_index(drop=True, inplace=True)
test_df = test_df.sample(frac=1, random_state=2)
test_df.reset_index(drop=True, inplace=True)

#print(train_df.describe(include="all"))

#print(train_df.shape)

#print(train_df.head())


df = df.drop(["Date", "Home", "Score", "Away", "Venue", "Attendance"], axis=1)
train_df = train_df.drop(["Date", "Home", "Score", "Away", "Venue", "Attendance"], axis=1)
test_df = test_df.drop(["Date", "Home", "Score", "Away", "Venue", "Attendance"], axis=1)

#print(train_df.head())

features = df.columns.tolist()
features = features[1:]

train_X = train_df[features]
test_X = test_df[features]

label_encoder = LabelEncoder()
label_encoder.fit(df["Result"])
train_y = label_encoder.transform(train_df["Result"])
test_y = label_encoder.transform(test_df["Result"])


preprocessor = ColumnTransformer([
    ("num", Pipeline([("scaler", StandardScaler())]), features)], remainder="passthrough")

# -------------- kNN ------------------
# Majority-class Classifier
maj = DummyClassifier()
maj.fit(train_X, train_y)
print(accuracy_score(test_y, maj.predict(test_X)))
# 0.34

knn = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsClassifier())])

knn_param_grid = {"predictor__n_neighbors": [49]}

knn_gs = GridSearchCV(knn, knn_param_grid, scoring="accuracy", cv=10)

knn_gs.fit(train_X, train_y)
print("Best Params: ", knn_gs.best_params_, knn_gs.best_score_)

# Under/over-fitting
knn.set_params(**knn_gs.best_params_)
scores = cross_validate(knn, train_X, train_y, cv=10, scoring="accuracy", return_train_score=True)
print("Training Accuracy: ", np.mean(scores["train_score"]))
print("Validation Accuracy: ", np.mean(scores["test_score"]))


# -------------- Decision Tree ------------------
tree = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", DecisionTreeClassifier())])

tree_param_grid = {"predictor__max_depth": [3]}

tree_gs = GridSearchCV(tree, tree_param_grid, scoring="accuracy", cv=10)

tree_gs.fit(train_X, train_y)
print("Best Params: ", tree_gs.best_params_, tree_gs.best_score_)

# Under/over-fitting
tree.set_params(**tree_gs.best_params_)
scores = cross_validate(tree, train_X, train_y, cv=10, scoring="accuracy", return_train_score=True)
print("Training Accuracy: ", np.mean(scores["train_score"]))
print("Validation Accuracy: ", np.mean(scores["test_score"]))


