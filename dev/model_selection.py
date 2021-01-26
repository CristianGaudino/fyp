import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix
from seaborn import scatterplot
from seaborn import lmplot, stripplot
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from dev.feature_insertion_class import InsertFeature


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
df = pd.read_csv("../datasets/prem_final_datasets/prem_whole_set.csv")
train_df = pd.read_csv("../datasets/prem_final_datasets/prem_training_set.csv")
test_df = pd.read_csv("../datasets/prem_final_datasets/prem_test_set.csv")

df = df.sample(frac=1, random_state=2)
df.reset_index(drop=True, inplace=True)
train_df = train_df.sample(frac=1, random_state=2)
train_df.reset_index(drop=True, inplace=True)
test_df = test_df.sample(frac=1, random_state=2)
test_df.reset_index(drop=True, inplace=True)

df = df.drop(["Date", "Home", "Score", "Away", "Venue"], axis=1)
train_df = train_df.drop(["Date", "Home", "Score", "Away", "Venue"], axis=1)
test_df = test_df.drop(["Date", "Home", "Score", "Away", "Venue"], axis=1)

features = df.columns.tolist()
del features[1]
#features = features[1:]

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

features = [x for x in features if x not in per_90_features]

df = df.drop(per_90_features, axis=1)
train_df = train_df.drop(per_90_features, axis=1)
test_df = test_df.drop(per_90_features, axis=1)


# -----MODEL SELECTION-----
train_X = train_df[features]
test_X = test_df[features]

label_encoder = LabelEncoder()
label_encoder.fit(df["Result"])
train_y = label_encoder.transform(train_df["Result"])
test_y = label_encoder.transform(test_df["Result"])



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

preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", TransformerFromHyperP())]),
         features)],
        remainder="passthrough")

# Majority-class Classifier
maj = DummyClassifier()
maj.fit(train_X, train_y)
print("MAJORITY CLASSIFIER -- ", accuracy_score(test_y, maj.predict(test_X)))


# Using holdout with stratification
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=2)   # Split for validation set, 60-20-20 overall

'''
print("--------")
# -- kNN --
knn = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsClassifier())])

# Dictionary for the hyperparameters, as well as the new features. I also included 3 scalers.
knn_param_grid = {"predictor__n_neighbors": [35, 36, 37],
                  "preprocessor__num__scaler__transformer": [StandardScaler(), MinMaxScaler(), RobustScaler()],
                  }  #, RobustScaler(), MinMaxScaler()
# This dataset is quiet large, so holdout has been used.
knn_gs = GridSearchCV(knn, knn_param_grid, scoring="accuracy", cv=sss)
#print(train_X.dtypes)
#print(train_X.head())
knn_gs.fit(train_X, train_y)

print("KNN BEST PARAMS/SCORE: ", knn_gs.best_params_, knn_gs.best_score_)
# Checking for under/overfitting
knn.set_params(**knn_gs.best_params_)
scores = cross_validate(knn, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
print("KNN Training Accuracy: ", np.mean(scores["train_score"]))
print("KNN Validation Accuracy: ", np.mean(scores["test_score"]))

# 36NN, Standard Scaler, Score .584, no over/under-fitting

print("--------")
# -- Logistic Regression --
logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", LogisticRegression(max_iter=5000))])  # Regularisation by default is l2, which I'm fine with.

# Dictionary for the hyperparameters, in this case C for the amount of regularisation,
# as well as the new features. I also included 3 scalers.
logistic_param_grid = {"predictor__C": [0.07, 0.085, 0.1, 0.12, 0.15],
                       "preprocessor__num__scaler__transformer": [StandardScaler(), RobustScaler(), MinMaxScaler()],
                       }
# This dataset is quiet large, so holdout has been used.
logistic_gs = GridSearchCV(logistic, logistic_param_grid, scoring="accuracy", cv=sss)

logistic_gs.fit(train_X, train_y)

print("LOGISTIC REGRESSION BEST PARAMS/SCORE: ", logistic_gs.best_params_, logistic_gs.best_score_)
# Checking for under/overfitting
logistic.set_params(**logistic_gs.best_params_)
scores = cross_validate(logistic, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
print("LOGISTIC Training Accuracy: ", np.mean(scores["train_score"]))
print("LOGISTIC Validation Accuracy: ", np.mean(scores["test_score"]))

# Logistic Regression, MinMax Scaler, Score .560, over-fitting by 7.5%

print("--------")
# -- Decision Tree --
# Decision Tree
# Preprocessor is needed as we need the new features and we need to one-hot encode nominal features.
tree = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", DecisionTreeClassifier())])

# Dictionary for the hyperparameters, in this case max depth. While it's not needed for
# decision trees, the same preprocessor is used here, so there's scalers for the transformer.
tree_param_grid = {"predictor__max_depth": [2, 3, 4],
                   "preprocessor__num__scaler__transformer": [StandardScaler(), MinMaxScaler(), RobustScaler()],
                   }
# This dataset is quiet large, so holdout has been used.
tree_gs = GridSearchCV(tree, tree_param_grid, scoring="accuracy", cv=sss)

tree_gs.fit(train_X, train_y)

print("DECISION TREE BEST PARAMS/SCORE:", tree_gs.best_params_, tree_gs.best_score_)

# Checking for under/overfitting
tree.set_params(**tree_gs.best_params_)
scores = cross_validate(tree, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
print("TREE Training Accuracy: ", np.mean(scores["train_score"]))
print("TREE Validation Accuracy: ", np.mean(scores["test_score"]))

# Decision Tree, Standard Scaler, Score .565, no over/under-fitting

print("--------")
# -- Random Forest --
# Preprocessor is needed as we need the new features and we need to one-hot encode nominal features.
forest = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", RandomForestClassifier())])

# Dictionary for the hyperparameters, in this case n_estimators (the number of trees used, default being 100),
# max_depth and random_state.
# As with decision trees, a scaler isn't needed, but the same preprocessor is used.
forest_param_grid = {"predictor__max_depth": [2, 3, 4, 5, 7, 9, 12],
                     "predictor__n_estimators": [5, 15, 25, 40],
                     "predictor__random_state": [1, 2],
                     "preprocessor__num__scaler__transformer": [StandardScaler(), RobustScaler(), MinMaxScaler()],
                 }
# This dataset is quiet large, so holdout has been used.
forest_gs = GridSearchCV(forest, forest_param_grid, scoring="accuracy", cv=sss)

forest_gs.fit(train_X, train_y)

print("RANDOM FOREST BEST PARAMS/SCORE: ", forest_gs.best_params_, forest_gs.best_score_)

# Checking for under/overfitting
forest.set_params(**forest_gs.best_params_)
scores = cross_validate(forest, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
print("RANDOM FOREST Training Accuracy: ", np.mean(scores["train_score"]))
print("RANDOM FOREST Validation Accuracy: ", np.mean(scores["test_score"]))

# Random Forest, can over-fit by a lot depending on hyper-parameters -- tune it better
'''
print("--------")
# -- AdaBoostClassifier --
# Preprocessor is needed as we need the new features and we need to one-hot encode nominal features.
forest = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", AdaBoostClassifier())])

# Dictionary for the hyperparameters, in this case n_estimators (the number of trees used, default being 100),
# max_depth and random_state.
# As with decision trees, a scaler isn't needed, but the same preprocessor is used.
forest_param_grid = {"predictor__learning_rate": [0.1, 0.2, 0.4, 0.7, 1],
                     "predictor__n_estimators": [50, 70, 100, 150],
                     "predictor__random_state": [1, 2, 3],
                     "preprocessor__num__scaler__transformer": [MinMaxScaler(), RobustScaler(), StandardScaler()],
                 }
# This dataset is quiet large, so holdout has been used.
forest_gs = GridSearchCV(forest, forest_param_grid, scoring="accuracy", cv=sss)

forest_gs.fit(train_X, train_y)

print("ADABOOST ENSEMBLE BEST PARAMS/SCORE: ", forest_gs.best_params_, forest_gs.best_score_)

# Checking for under/overfitting
forest.set_params(**forest_gs.best_params_)
scores = cross_validate(forest, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
print("AdaBoost Training Accuracy: ", np.mean(scores["train_score"]))
print("AdaBoost Validation Accuracy: ", np.mean(scores["test_score"]))

# AdaBoost, MinMax Scaler, Score .560, 2% over-fit -- needs to be tuned

# Research others and write out why they weren't tried
