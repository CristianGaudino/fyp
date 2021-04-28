import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix
from seaborn import scatterplot
from seaborn import lmplot, stripplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

import scipy.stats as stats
from sklearn.utils.fixes import loguniform

from dev.feature_insertion_class import InsertDivideFeature
from sklearn.feature_selection import RFECV

from joblib import dump, load

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

df = df.drop(["Date", "Home", "Score", "Away", "Venue", "MP_Home", "MP_Away"], axis=1)
train_df = train_df.drop(["Date", "Home", "Score", "Away", "Venue", "MP_Home", "MP_Away"], axis=1)
test_df = test_df.drop(["Date", "Home", "Score", "Away", "Venue", "MP_Home", "MP_Away"], axis=1)


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

# features = [x for x in features if x not in per_90_features]    # Update the list of features

df = df.drop(per_90_features, axis=1)
train_df = train_df.drop(per_90_features, axis=1)
test_df = test_df.drop(per_90_features, axis=1)

redundant_features = ["CrdR_Home", "D_Away", "CrdR_Away", "Sh_Away", "MP_2_Home", "Min_2_Home", "MP_4_Home",
                      "Min_4_Home", "MP_5_Home", "Min_5_Home", "MP_6_Home", "Min_6_Home", "MP_8_Home", "Min_8_Home",
                      "MP_11_Home", "Min_11_Home", "MP_2_Away", "Min_2_Away"]

df = df.drop(redundant_features, axis=1)
train_df = train_df.drop(redundant_features, axis=1)
test_df = test_df.drop(redundant_features, axis=1)

'''
def addEngineeredFeatures(df):
    # Funtion to add the resulting features of feature engineering to the given dataframe
    df["CS_per_W_Home"] = df["CS_Home"]/df["W_Home"]
    df["CS_per_L_Away"] = df["CS_Away"]/df["L_Away"]
    df["Gls_per_W_Home"] = df["Gls_Home"] / df["W_Home"]
    df["Gls_per_L_Away"] = df["Gls_Away"] / df["L_Away"]
    df["Gls_per_SoT_per_W_Home"] = df["G/SoT_Home"] / df["W_Home"]
    df["Gls_per_SoT_per_W_Away"] = df["G/SoT_Away"] / df["W_Away"]
    df["SoT_per_W_Home"] = df["SoT_Home"] / df["W_Home"]
    df["SoT_per_L_Away"] = df["SoT_Away"] / df["L_Away"]
    return df


df = addEngineeredFeatures(df)
train_df = addEngineeredFeatures(train_df)
test_df = addEngineeredFeatures(test_df)
'''

df = df.sample(frac=1, random_state=2)
df.reset_index(drop=True, inplace=True)
train_df = train_df.sample(frac=1, random_state=2)
train_df.reset_index(drop=True, inplace=True)
test_df = test_df.sample(frac=1, random_state=2)
test_df.reset_index(drop=True, inplace=True)

features = df.columns.tolist()
del features[1]


# -----MODEL SELECTION-----
train_X = train_df[features]
test_X = test_df[features]
df_X = df[features]

train_y = train_df["Result"]
test_y = test_df["Result"]
df_y = df["Result"]


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
        ("num", Pipeline([("cs_per_w_home", InsertDivideFeature("cs_per_w_home", "CS_Home", "W_Home")),
                          ("cs_per_l_away", InsertDivideFeature("cs_per_l_away", "CS_Away", "L_Away")),
                          ("gls_per_w_home", InsertDivideFeature("gls_per_w_home", "Gls_Home", "W_Home")),
                          ("gls_per_l_away", InsertDivideFeature("gls_per_l_away", "Gls_Away", "L_Away")),
                          ("gls_per_sot_per_w_home", InsertDivideFeature("gls_per_sot_per_w_home", "G/SoT_Home", "W_Home")),
                          ("gls_per_sot_per_w_away", InsertDivideFeature("gls_per_sot_per_w_away", "G/SoT_Away", "W_Away")),
                          ("sot_per_w_home", InsertDivideFeature("sot_per_w_home", "SoT_Home", "W_Home")),
                          ("sot_per_l_away", InsertDivideFeature("sot_per_l_away", "SoT_Away", "L_Away")),
                          ("scaler", TransformerFromHyperP())]),
         features)],
        remainder="passthrough")
'''
preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", TransformerFromHyperP())]),
         features)],
        remainder="passthrough")
'''
'''
# Majority-class Classifier
maj = DummyClassifier()
maj.fit(train_X, train_y)
print("MAJORITY CLASSIFIER -- ", accuracy_score(test_y, maj.predict(test_X)))
'''

# Using holdout with stratification
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=2)   # Split for validation set, 60-20-20 overall


'''
print("--------")
# -- kNN --

knn = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsClassifier(n_jobs=-1))])

# Dictionary for the hyperparameters, as well as the new features. I also included 3 scalers.
knn_param_grid = {"predictor__n_neighbors": [53],
                  "predictor__weights": ["uniform"],
                  "preprocessor__num__cs_per_w_home__insert": [False],
                  "preprocessor__num__cs_per_l_away__insert": [False],
                  "preprocessor__num__gls_per_w_home__insert": [True],
                  "preprocessor__num__gls_per_l_away__insert": [True],
                  "preprocessor__num__gls_per_sot_per_w_home__insert": [False],
                  "preprocessor__num__gls_per_sot_per_w_away__insert": [False],
                  "preprocessor__num__sot_per_w_home__insert": [True],
                  "preprocessor__num__sot_per_l_away__insert": [True],
                  "preprocessor__num__scaler__transformer": [RobustScaler()],
                  }

knn_param_dist = {"predictor__n_neighbors": list(range(15, 70)),
                  "predictor__weights": ["uniform"],
                  "preprocessor__num__cs_per_w_home__insert": [True, False],
                  "preprocessor__num__cs_per_l_away__insert": [True, False],
                  "preprocessor__num__gls_per_w_home__insert": [True, False],
                  "preprocessor__num__gls_per_l_away__insert": [True, False],
                  "preprocessor__num__gls_per_sot_per_w_home__insert": [True, False],
                  "preprocessor__num__gls_per_sot_per_w_away__insert": [True, False],
                  "preprocessor__num__sot_per_w_home__insert": [True, False],
                  "preprocessor__num__sot_per_l_away__insert": [True, False],
                  "preprocessor__num__scaler__transformer": [StandardScaler(), MinMaxScaler(), RobustScaler()],
                  }

# This dataset is quiet large, so holdout has been used.
# n_iter_search = 600
# knn_rs = RandomizedSearchCV(knn, param_distributions=knn_param_dist, n_iter=n_iter_search, scoring="accuracy", cv=sss)
knn_gs = GridSearchCV(knn, knn_param_grid, scoring="accuracy", cv=sss)

knn_gs.fit(train_X, train_y)

print("KNN BEST PARAMS/SCORE: ", knn_gs.best_params_, knn_gs.best_score_)
# Checking for under/overfitting
knn.set_params(**knn_gs.best_params_)
scores = cross_validate(knn, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
knn_score_train = np.mean(scores["train_score"])
knn_score_test = np.mean(scores["test_score"])
print("KNN Training Accuracy: ", knn_score_train)
print("KNN Validation Accuracy: ", knn_score_test)

knn.set_params(**knn_gs.best_params_)
knn.fit(train_X, train_y)
plot = plot_confusion_matrix(knn, test_X, test_y, cmap="Blues")
plt.title('Confusion Matrix - k Nearest Neighbour')

#KNN BEST PARAMS/SCORE:  {'preprocessor__num__sot_per_w_home__insert': True, 'preprocessor__num__sot_per_l_away__insert': True, 'preprocessor__num__scaler__transformer': RobustScaler(), 'preprocessor__num__gls_per_w_home__insert': True, 'preprocessor__num__gls_per_sot_per_w_home__insert': False, 'preprocessor__num__gls_per_sot_per_w_away__insert': False, 'preprocessor__num__gls_per_l_away__insert': True, 'preprocessor__num__cs_per_w_home__insert': False, 'preprocessor__num__cs_per_l_away__insert': False, 'predictor__weights': 'uniform', 'predictor__n_neighbors': 53} 0.5868421052631579
#KNN Training Accuracy:  0.5614035087719298
#KNN Validation Accuracy:  0.5868421052631579

'''

print("--------")
# -- Logistic Regression --
logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", LogisticRegression(max_iter=2000, n_jobs=-1))])  # Regularisation by default is l2, which I'm fine with.

# Dictionary for the hyperparameters, in this case C for the amount of regularisation,
# as well as the new features. I also included 3 scalers.
logistic_param_grid = {"predictor__C": [0.002],
                       "predictor__penalty": ["l2"],
                       "preprocessor__num__cs_per_w_home__insert": [True],
                       "preprocessor__num__cs_per_l_away__insert": [False],
                       "preprocessor__num__gls_per_w_home__insert": [True],
                       "preprocessor__num__gls_per_l_away__insert": [False],
                       "preprocessor__num__gls_per_sot_per_w_home__insert": [True],
                       "preprocessor__num__gls_per_sot_per_w_away__insert": [False],
                       "preprocessor__num__sot_per_w_home__insert": [True],
                       "preprocessor__num__sot_per_l_away__insert": [True],
                       "preprocessor__num__scaler__transformer": [RobustScaler()],
                       }


logistic_param_dist = {"predictor__C": loguniform(1e-5, 1),
                       "preprocessor__num__cs_per_w_home__insert": [True, False],
                       "preprocessor__num__cs_per_l_away__insert": [True, False],
                       "preprocessor__num__gls_per_w_home__insert": [True, False],
                       "preprocessor__num__gls_per_l_away__insert": [True, False],
                       "preprocessor__num__gls_per_sot_per_w_home__insert": [True, False],
                       "preprocessor__num__gls_per_sot_per_w_away__insert": [True, False],
                       "preprocessor__num__sot_per_w_home__insert": [True, False],
                       "preprocessor__num__sot_per_l_away__insert": [True, False],
                       "preprocessor__num__scaler__transformer": [StandardScaler(), RobustScaler(), MinMaxScaler()]
                       }

# This dataset is quiet large, so holdout has been used.
#n_iter_search = 200
#logistic_rs = RandomizedSearchCV(logistic, param_distributions=logistic_param_dist, n_iter=n_iter_search, scoring="accuracy", cv=sss)
logistic_gs = GridSearchCV(logistic, logistic_param_grid, scoring="accuracy", cv=sss)


logistic_gs.fit(train_X, train_y)

#print("LOGISTIC REGRESSION BEST PARAMS/SCORE: ", logistic_gs.best_params_, logistic_gs.best_score_)
# Checking for under/overfitting
logistic.set_params(**logistic_gs.best_params_)
scores = cross_validate(logistic, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
logistic_score_train = np.mean(scores["train_score"])
logistic_score_test = np.mean(scores["test_score"])
#print("LOGISTIC Training Accuracy: ", logistic_score_train)
#print("LOGISTIC Validation Accuracy: ", logistic_score_test)

logistic.set_params(**logistic_gs.best_params_)
logistic.fit(train_X, train_y)
plot = plot_confusion_matrix(logistic, test_X, test_y, cmap="Blues")
plt.title('Confusion Matrix - Logistic Regression')

#LOGISTIC REGRESSION BEST PARAMS/SCORE:  {'predictor__C': 0.0020096223623516747, 'predictor__penalty': 'l2', 'preprocessor__num__cs_per_l_away__insert': False, 'preprocessor__num__cs_per_w_home__insert': True, 'preprocessor__num__gls_per_l_away__insert': True, 'preprocessor__num__gls_per_sot_per_w_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_home__insert': False, 'preprocessor__num__gls_per_w_home__insert': False, 'preprocessor__num__scaler__transformer': RobustScaler(), 'preprocessor__num__sot_per_l_away__insert': False, 'preprocessor__num__sot_per_w_home__insert': True} 0.5921052631578947
#LOGISTIC Training Accuracy:  0.5947368421052631
#LOGISTIC Validation Accuracy:  0.5921052631578947

# LOGISTIC REGRESSION BEST PARAMS/SCORE:  {'predictor__C': 0.002, 'predictor__penalty': 'l2', 'preprocessor__num__cs_per_l_away__insert': False, 'preprocessor__num__cs_per_w_home__insert': True, 'preprocessor__num__gls_per_l_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_home__insert': True, 'preprocessor__num__gls_per_w_home__insert': True, 'preprocessor__num__scaler__transformer': RobustScaler(), 'preprocessor__num__sot_per_l_away__insert': True, 'preprocessor__num__sot_per_w_home__insert': True} 0.6
# LOGISTIC Training Accuracy:  0.5956140350877193
# LOGISTIC Validation Accuracy:  0.6


'''
logistic_param_grid = {"predictor__C": [0.0020096223623516747],
                       "predictor__penalty": ["l2"],
                       "preprocessor__num__cs_per_w_home__insert": [True],
                       "preprocessor__num__cs_per_l_away__insert": [False],
                       "preprocessor__num__gls_per_w_home__insert": [False],
                       "preprocessor__num__gls_per_l_away__insert": [True],
                       "preprocessor__num__gls_per_sot_per_w_home__insert": [False],
                       "preprocessor__num__gls_per_sot_per_w_away__insert": [False],
                       "preprocessor__num__sot_per_w_home__insert": [True],
                       "preprocessor__num__sot_per_l_away__insert": [False],
                       "preprocessor__num__scaler__transformer": [RobustScaler()],
                       }
'''


'''
print("--------")
# -- Decision Tree --
tree_rfecv = RFECV(estimator=DecisionTreeClassifier(), verbose=0, step=10, cv=sss, scoring="accuracy")

# Preprocessor is needed as we need the new features and we need to one-hot encode nominal features.
tree = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", tree_rfecv)])

# Dictionary for the hyperparameters, in this case max depth. While it's not needed for
# decision trees, the same preprocessor is used here, so there's scalers for the transformer.
tree_param_grid = {"predictor__estimator__max_depth": [3],
                   "predictor__estimator__criterion": ["entropy"],  # gini
                   "predictor__estimator__splitter": ["best"],  # random
                   "preprocessor__num__cs_per_w_home__insert": [False],
                   "preprocessor__num__cs_per_l_away__insert": [False],
                   "preprocessor__num__gls_per_w_home__insert": [False],
                   "preprocessor__num__gls_per_l_away__insert": [False],
                   "preprocessor__num__gls_per_sot_per_w_home__insert": [False],
                   "preprocessor__num__gls_per_sot_per_w_away__insert": [True],
                   "preprocessor__num__sot_per_w_home__insert": [False],
                   "preprocessor__num__sot_per_l_away__insert": [True],
                   "preprocessor__num__scaler__transformer": [StandardScaler()],
                   }

tree_param_dist = {"predictor__estimator__max_depth": list(range(1, 9)),
                   "predictor__estimator__criterion": ["entropy", "gini"],  # gini
                   "predictor__estimator__splitter": ["best", "random"],  # random
                   "preprocessor__num__cs_per_w_home__insert": [True, False],
                   "preprocessor__num__cs_per_l_away__insert": [True, False],
                   "preprocessor__num__gls_per_w_home__insert": [True, False],
                   "preprocessor__num__gls_per_l_away__insert": [True, False],
                   "preprocessor__num__gls_per_sot_per_w_home__insert": [True, False],
                   "preprocessor__num__gls_per_sot_per_w_away__insert": [True, False],
                   "preprocessor__num__sot_per_w_home__insert": [True, False],
                   "preprocessor__num__sot_per_l_away__insert": [True, False],
                   "preprocessor__num__scaler__transformer": [StandardScaler(), MinMaxScaler(), RobustScaler()],
                   }

# This dataset is quiet large, so holdout has been used.
#n_iter_search = 300
#tree_rs = RandomizedSearchCV(tree, param_distributions=tree_param_dist, n_iter=n_iter_search, scoring="accuracy", cv=sss)
tree_gs = GridSearchCV(tree, tree_param_grid, scoring="accuracy", cv=sss)
tree_gs.fit(train_X, train_y)

print("DECISION TREE BEST PARAMS/SCORE:", tree_gs.best_params_, tree_gs.best_score_)

# Checking for under/overfitting
tree.set_params(**tree_gs.best_params_)
scores = cross_validate(tree, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
tree_score_train = np.mean(scores["train_score"])
tree_score_test = np.mean(scores["test_score"])
print("TREE Training Accuracy: ", tree_score_train)
print("TREE Validation Accuracy: ", tree_score_test)

tree.set_params(**tree_gs.best_params_)
tree.fit(train_X, train_y)
plot = plot_confusion_matrix(tree, test_X, test_y, cmap="Blues")
plt.title('Confusion Matrix - Decision Tree')

#DECISION TREE BEST PARAMS/SCORE: {'predictor__estimator__criterion': 'entropy', 'predictor__estimator__max_depth': 3, 'predictor__estimator__splitter': 'best', 'preprocessor__num__cs_per_l_away__insert': False, 'preprocessor__num__cs_per_w_home__insert': False, 'preprocessor__num__gls_per_l_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_away__insert': True, 'preprocessor__num__gls_per_sot_per_w_home__insert': False, 'preprocessor__num__gls_per_w_home__insert': False, 'preprocessor__num__scaler__transformer': StandardScaler(), 'preprocessor__num__sot_per_l_away__insert': True, 'preprocessor__num__sot_per_w_home__insert': False} 0.5789473684210527
#TREE Training Accuracy:  0.5745614035087719
#TREE Validation Accuracy:  0.5789473684210527


print("--------")
# -- Random Forest --

forest_rfecv = RFECV(estimator=RandomForestClassifier(n_jobs=-1), step=20, cv=sss, scoring="accuracy")

# Preprocessor is needed as we need the new features and we need to one-hot encode nominal features.
forest = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", forest_rfecv)])

# Dictionary for the hyperparameters, in this case n_estimators (the number of trees used, default being 100),
# max_depth and random_state.
# As with decision trees, a scaler isn't needed, but the same preprocessor is used.
forest_param_grid = {"predictor__estimator__max_depth": [5],
                     "predictor__estimator__n_estimators": [6],
                     "predictor__estimator__random_state": [1],
                     "predictor__estimator__criterion": ["entropy"],
                     "preprocessor__num__cs_per_w_home__insert": [False],
                     "preprocessor__num__cs_per_l_away__insert": [False],
                     "preprocessor__num__gls_per_w_home__insert": [False],
                     "preprocessor__num__gls_per_l_away__insert": [False],
                     "preprocessor__num__gls_per_sot_per_w_home__insert": [False],
                     "preprocessor__num__gls_per_sot_per_w_away__insert": [False],
                     "preprocessor__num__sot_per_w_home__insert": [True],
                     "preprocessor__num__sot_per_l_away__insert": [True],
                     "preprocessor__num__scaler__transformer": [StandardScaler()],
                     }


forest_param_dist = {"predictor__estimator__max_depth": list(range(1, 6)),
                     "predictor__estimator__n_estimators": list(range(1, 50)),
                     "predictor__estimator__random_state": [1, 2],
                     "predictor__estimator__criterion": ["gini", "entropy"],
                     "preprocessor__num__cs_per_w_home__insert": [True, False],
                     "preprocessor__num__cs_per_l_away__insert": [True, False],
                     "preprocessor__num__gls_per_w_home__insert": [True, False],
                     "preprocessor__num__gls_per_l_away__insert": [True, False],
                     "preprocessor__num__gls_per_sot_per_w_home__insert": [True, False],
                     "preprocessor__num__gls_per_sot_per_w_away__insert": [True, False],
                     "preprocessor__num__sot_per_w_home__insert": [True, False],
                     "preprocessor__num__sot_per_l_away__insert": [True, False],
                     "preprocessor__num__scaler__transformer": [StandardScaler(), RobustScaler(), MinMaxScaler()]
                     }

# This dataset is quiet large, so holdout has been used.
#n_iter_search = 50
#forest_rs = RandomizedSearchCV(forest, param_distributions=forest_param_dist, n_iter=n_iter_search, scoring="accuracy", cv=sss)
forest_gs = GridSearchCV(forest, forest_param_grid, scoring="accuracy", cv=sss)

forest_gs.fit(train_X, train_y)

print("RANDOM FOREST BEST PARAMS/SCORE: ", forest_gs.best_params_, forest_gs.best_score_)

# Checking for under/overfitting
forest.set_params(**forest_gs.best_params_)
scores = cross_validate(forest, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
forest_score_train = np.mean(scores["train_score"])
forest_score_test = np.mean(scores["test_score"])
print("RANDOM FOREST Training Accuracy: ", forest_score_train)
print("RANDOM FOREST Validation Accuracy: ", forest_score_test)

forest.set_params(**forest_gs.best_params_)
forest.fit(train_X, train_y)
plot = plot_confusion_matrix(forest, test_X, test_y, cmap="Blues")
plt.title('Confusion Matrix - Random Forest')

#RANDOM FOREST BEST PARAMS/SCORE:  {'preprocessor__num__sot_per_w_home__insert': True, 'preprocessor__num__sot_per_l_away__insert': False, 'preprocessor__num__scaler__transformer': RobustScaler(), 'preprocessor__num__gls_per_w_home__insert': False, 'preprocessor__num__gls_per_sot_per_w_home__insert': True, 'preprocessor__num__gls_per_sot_per_w_away__insert': False, 'preprocessor__num__gls_per_l_away__insert': True, 'preprocessor__num__cs_per_w_home__insert': False, 'preprocessor__num__cs_per_l_away__insert': True, 'predictor__estimator__random_state': 2, 'predictor__estimator__n_estimators': 6, 'predictor__estimator__max_depth': 5, 'predictor__estimator__criterion': 'gini'} 0.5815789473684211
#RANDOM FOREST Training Accuracy:  0.6491228070175439
#RANDOM FOREST Validation Accuracy:  0.5815789473684211

# RANDOM FOREST BEST PARAMS/SCORE:  {'predictor__estimator__criterion': 'entropy', 'predictor__estimator__max_depth': 5, 'predictor__estimator__n_estimators': 6, 'predictor__estimator__random_state': 1, 'preprocessor__num__cs_per_l_away__insert': False, 'preprocessor__num__cs_per_w_home__insert': False, 'preprocessor__num__gls_per_l_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_home__insert': False, 'preprocessor__num__gls_per_w_home__insert': False, 'preprocessor__num__scaler__transformer': StandardScaler(), 'preprocessor__num__sot_per_l_away__insert': True, 'preprocessor__num__sot_per_w_home__insert': True} 0.5868421052631579
# RANDOM FOREST Training Accuracy:  0.624561403508772
# RANDOM FOREST Validation Accuracy:  0.5868421052631579



print("--------")
# -- AdaBoostClassifier --

ada_rfecv = RFECV(estimator=AdaBoostClassifier(), verbose=0, step=20, cv=sss, scoring="accuracy")

# Preprocessor is needed as we need the new features and we need to one-hot encode nominal features.
ada = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", AdaBoostClassifier())]) # base_estimator=LogisticRegression(C=0.002009)

# Dictionary for the hyperparameters, in this case n_estimators (the number of trees used, default being 100),
# max_depth and random_state.
# As with decision trees, a scaler isn't needed, but the same preprocessor is used.
ada_param_grid = {"predictor__learning_rate": [0.1815301546808771],
                  "predictor__n_estimators": [71],
                  "predictor__random_state": [2],
                  "preprocessor__num__cs_per_w_home__insert": [False],
                  "preprocessor__num__cs_per_l_away__insert": [False],
                  "preprocessor__num__gls_per_w_home__insert": [False],
                  "preprocessor__num__gls_per_l_away__insert": [False],
                  "preprocessor__num__gls_per_sot_per_w_home__insert": [False],
                  "preprocessor__num__gls_per_sot_per_w_away__insert": [False],
                  "preprocessor__num__sot_per_w_home__insert": [False],
                  "preprocessor__num__sot_per_l_away__insert": [False],
                  "preprocessor__num__scaler__transformer": [StandardScaler()],
                  }

ada_param_dist = {"predictor__learning_rate": loguniform(1e-5, 1),
                  "predictor__n_estimators": list(range(1, 120)),
                  "predictor__random_state": [1, 2, 3],
                  "preprocessor__num__cs_per_w_home__insert": [True, False],
                  "preprocessor__num__cs_per_l_away__insert": [True, False],
                  "preprocessor__num__gls_per_w_home__insert": [True, False],
                  "preprocessor__num__gls_per_l_away__insert": [True, False],
                  "preprocessor__num__gls_per_sot_per_w_home__insert": [True, False],
                  "preprocessor__num__gls_per_sot_per_w_away__insert": [True, False],
                  "preprocessor__num__sot_per_w_home__insert": [True, False],
                  "preprocessor__num__sot_per_l_away__insert": [True, False],
                  "preprocessor__num__scaler__transformer": [StandardScaler(), RobustScaler(), MinMaxScaler()]
                  }

# This dataset is quiet large, so holdout has been used.
#n_iter_search = 100
#ada_rs = RandomizedSearchCV(ada, param_distributions=ada_param_dist, n_iter=n_iter_search, scoring="accuracy", cv=sss)
ada_gs = GridSearchCV(ada, ada_param_grid, scoring="accuracy", cv=sss)

ada_gs.fit(train_X, train_y)

print("ADABOOST ENSEMBLE BEST PARAMS/SCORE: ", ada_gs.best_params_, ada_gs.best_score_)

# Checking for under/overfitting
ada.set_params(**ada_gs.best_params_)
scores = cross_validate(ada, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
ada_score_train = np.mean(scores["train_score"])
ada_score_test = np.mean(scores["test_score"])
print("ADABOOST Training Accuracy: ", ada_score_train)
print("ADABOOST Validation Accuracy: ", ada_score_test)

ada.set_params(**ada_gs.best_params_)
ada.fit(train_X, train_y)
plot = plot_confusion_matrix(ada, train_X, train_y, cmap="Blues")
plt.title('Confusion Matrix - AdaBoost')

#ADABOOST ENSEMBLE BEST PARAMS/SCORE:  {'predictor__learning_rate': 0.1815301546808771, 'predictor__n_estimators': 71, 'predictor__random_state': 2, 'preprocessor__num__cs_per_l_away__insert': False, 'preprocessor__num__cs_per_w_home__insert': False, 'preprocessor__num__gls_per_l_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_home__insert': False, 'preprocessor__num__gls_per_w_home__insert': False, 'preprocessor__num__scaler__transformer': StandardScaler(), 'preprocessor__num__sot_per_l_away__insert': False, 'preprocessor__num__sot_per_w_home__insert': False} 0.5736842105263158
#ADABOOST Training Accuracy:  0.6052631578947368
#ADABOOST Validation Accuracy:  0.5736842105263158


print("--------")
# -----SVM---------

svc = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", SVC(cache_size=2500))])  # base_estimator=LogisticRegression(C=0.002009)

# Dictionary for the hyperparameters, in this case n_estimators (the number of trees used, default being 100),
# max_depth and random_state.
# As with decision trees, a scaler isn't needed, but the same preprocessor is used.
svc_param_grid = {"predictor__C": [0.412946203459679],
                  "predictor__kernel": ["poly"],
                  "predictor__degree": [2],
                  "predictor__gamma": ["scale"],
                  "predictor__tol": [0.08807192804961164],
                  "preprocessor__num__cs_per_w_home__insert": [False],
                  "preprocessor__num__cs_per_l_away__insert": [True],
                  "preprocessor__num__gls_per_w_home__insert": [True],
                  "preprocessor__num__gls_per_l_away__insert": [True],
                  "preprocessor__num__gls_per_sot_per_w_home__insert": [True],
                  "preprocessor__num__gls_per_sot_per_w_away__insert": [False],
                  "preprocessor__num__sot_per_w_home__insert": [False],
                  "preprocessor__num__sot_per_l_away__insert": [True],
                  "preprocessor__num__scaler__transformer": [MinMaxScaler()]
                  }
#SVC ENSEMBLE BEST PARAMS/SCORE:  {'predictor__C': 0.412946203459679, 'predictor__degree': 2, 'predictor__gamma': 'scale', 'predictor__kernel': 'poly', 'predictor__tol': 0.08807192804961164, 'preprocessor__num__cs_per_l_away__insert': True, 'preprocessor__num__cs_per_w_home__insert': False, 'preprocessor__num__gls_per_l_away__insert': True, 'preprocessor__num__gls_per_sot_per_w_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_home__insert': True, 'preprocessor__num__gls_per_w_home__insert': True, 'preprocessor__num__scaler__transformer': MinMaxScaler(), 'preprocessor__num__sot_per_l_away__insert': True, 'preprocessor__num__sot_per_w_home__insert': False} 0.5842105263157895

svc_param_dist = {"predictor__C": loguniform(1e-5, 1),
                  "predictor__kernel": ["linear", "poly", "rbf"],
                  "predictor__degree": list(range(1, 4)),
                  "predictor__gamma": ["scale", "auto"],
                  "predictor__tol": loguniform(1e-4, 1),
                  "preprocessor__num__cs_per_w_home__insert": [True, False],
                  "preprocessor__num__cs_per_l_away__insert": [True, False],
                  "preprocessor__num__gls_per_w_home__insert": [True, False],
                  "preprocessor__num__gls_per_l_away__insert": [True, False],
                  "preprocessor__num__gls_per_sot_per_w_home__insert": [True, False],
                  "preprocessor__num__gls_per_sot_per_w_away__insert": [True, False],
                  "preprocessor__num__sot_per_w_home__insert": [True, False],
                  "preprocessor__num__sot_per_l_away__insert": [True, False],
                  "preprocessor__num__scaler__transformer": [StandardScaler(), RobustScaler(), MinMaxScaler()]
                  }

# This dataset is quiet large, so holdout has been used.
n_iter_search = 600
svc_rs = RandomizedSearchCV(svc, param_distributions=svc_param_dist, n_iter=n_iter_search, scoring="accuracy", cv=sss)
#svc_gs = GridSearchCV(svc, svc_param_grid, scoring="accuracy", cv=sss)

svc_rs.fit(train_X, train_y)

print("SVC ENSEMBLE BEST PARAMS/SCORE: ", svc_rs.best_params_, svc_rs.best_score_)

# Checking for under/overfitting
svc.set_params(**svc_rs.best_params_)
scores = cross_validate(svc, test_X, test_y, cv=sss, scoring="accuracy", return_train_score=True)
svc_score_train = np.mean(scores["train_score"])
svc_score_test = np.mean(scores["test_score"])
print("SVC Training Accuracy: ", svc_score_train)
print("SVC Validation Accuracy: ", svc_score_test)

svc.set_params(**svc_rs.best_params_)
svc.fit(train_X, train_y)
plot = plot_confusion_matrix(svc, train_X, train_y, cmap="Blues")
plt.title('Confusion Matrix - SVC')

#SVC ENSEMBLE BEST PARAMS/SCORE:  {'predictor__C': 0.412946203459679, 'predictor__degree': 2, 'predictor__gamma': 'scale', 'predictor__kernel': 'poly', 'predictor__tol': 0.08807192804961164, 'preprocessor__num__cs_per_l_away__insert': True, 'preprocessor__num__cs_per_w_home__insert': False, 'preprocessor__num__gls_per_l_away__insert': True, 'preprocessor__num__gls_per_sot_per_w_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_home__insert': True, 'preprocessor__num__gls_per_w_home__insert': True, 'preprocessor__num__scaler__transformer': MinMaxScaler(), 'preprocessor__num__sot_per_l_away__insert': True, 'preprocessor__num__sot_per_w_home__insert': False} 0.5842105263157895
#SVC Training Accuracy:  0.637719298245614
#SVC Validation Accuracy:  0.5842105263157895

#SVC ENSEMBLE BEST PARAMS/SCORE:  {'predictor__C': 0.3889277238201591, 'predictor__degree': 2, 'predictor__gamma': 'scale', 'predictor__kernel': 'poly', 'predictor__tol': 0.00013347685922033482, 'preprocessor__num__cs_per_l_away__insert': True, 'preprocessor__num__cs_per_w_home__insert': False, 'preprocessor__num__gls_per_l_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_away__insert': False, 'preprocessor__num__gls_per_sot_per_w_home__insert': True, 'preprocessor__num__gls_per_w_home__insert': True, 'preprocessor__num__scaler__transformer': MinMaxScaler(), 'preprocessor__num__sot_per_l_away__insert': True, 'preprocessor__num__sot_per_w_home__insert': False} 0.5842105263157895
#SVC Training Accuracy:  0.7157894736842105
#SVC Validation Accuracy:  0.5894736842105263


print("--------")
# -- Polynomial Regression --
poly = Pipeline([
    ("preprocessor", preprocessor),
    ("poly", PolynomialFeatures()),
    ("predictor", LogisticRegression(max_iter=2000, n_jobs=-1))])

# Dictionary for the hyperparameters, in this case C for the amount of regularisation,
# as well as the new features. I also included 3 scalers.
poly_param_grid = {"predictor__C": [0.0020096223623516747],
                   "predictor__penalty": ["l2"],
                   "poly__degree": [2],
                   "poly__interaction_only": [False],
                   "poly__include_bias": [False],
                   "preprocessor__num__cs_per_w_home__insert": [True],
                   "preprocessor__num__cs_per_l_away__insert": [False],
                   "preprocessor__num__gls_per_w_home__insert": [False],
                   "preprocessor__num__gls_per_l_away__insert": [True],
                   "preprocessor__num__gls_per_sot_per_w_home__insert": [False],
                   "preprocessor__num__gls_per_sot_per_w_away__insert": [False],
                   "preprocessor__num__sot_per_w_home__insert": [True],
                   "preprocessor__num__sot_per_l_away__insert": [False],
                   "preprocessor__num__scaler__transformer": [RobustScaler()],
                   }


poly_param_dist = {"predictor__C": loguniform(1e-5, 1),
                   "poly__degree": list(range(1, 30)),
                   "poly__interaction_only": [False, True],
                   "poly__include_bias": [True, False],
                   "preprocessor__num__cs_per_w_home__insert": [True, False],
                   "preprocessor__num__cs_per_l_away__insert": [True, False],
                   "preprocessor__num__gls_per_w_home__insert": [True, False],
                   "preprocessor__num__gls_per_l_away__insert": [True, False],
                   "preprocessor__num__gls_per_sot_per_w_home__insert": [True, False],
                   "preprocessor__num__gls_per_sot_per_w_away__insert": [True, False],
                   "preprocessor__num__sot_per_w_home__insert": [True, False],
                   "preprocessor__num__sot_per_l_away__insert": [True, False],
                   "preprocessor__num__scaler__transformer": [StandardScaler(), RobustScaler(), MinMaxScaler()]
                   }

# This dataset is quiet large, so holdout has been used.
n_iter_search = 200
poly_rs = RandomizedSearchCV(poly, param_distributions=poly_param_dist, n_iter=n_iter_search, scoring="accuracy", cv=sss)
#poly_gs = GridSearchCV(poly, poly_param_grid, scoring="accuracy", cv=sss)


poly_rs.fit(train_X, train_y)

print("POLYNOMIAL REGRESSION BEST PARAMS/SCORE: ", poly_rs.best_params_, poly_rs.best_score_)
# Checking for under/overfitting
poly.set_params(**poly_rs.best_params_)
scores = cross_validate(poly, test_X, test_y, cv=sss, scoring="accuracy", return_train_score=True)
poly_score_train = np.mean(scores["train_score"])
poly_score_test = np.mean(scores["test_score"])
print("POLYNOMIAL Training Accuracy: ", poly_score_train)
print("POLYNOMIAL Validation Accuracy: ", poly_score_test)

poly.set_params(**poly_rs.best_params_)
poly.fit(train_X, train_y)
plot = plot_confusion_matrix(poly, train_X, train_y, cmap="Blues")
plt.title('Confusion Matrix - Polynomial Regression')
'''
'''
# --- Plot --- 
names = ["kNN", "Logistic_Reg", "Decision_Tree", "Random_Forest", "AdaBoost", "SVC", "Polynomial_Reg"]
values_train = [0.5614035087719298, logistic_score_train, tree_score_train, forest_score_train, ada_score_train, svc_score_train, poly_score_train]
values_test = [0.5868421052631579, logistic_score_test, tree_score_test, forest_score_test, ada_score_test, svc_score_test, poly_score_test]

plt.figure(figsize=(9, 4))
plt.subplot(134)
plt.plot(names, values_train, label="Training Accuracy")
plt.plot(names, values_test, label="Validation Accuracy")
plt.legend(loc="upper left")
plt.ylim(0.40, 0.70)

plt.ylabel("Accuracy")
'''

'''
# --- Column Chart Comparing Model Accuracies ---
labels = ["kNN", "Logistic_Reg", "Decision_Tree", "Random_Forest", "AdaBoost", "SVC"]
values_train = [56.1, 59.4, 57.4, 64.9, 60.5, 63.7]
values_test = [58.6, 59.2, 57.8, 58.1, 57.3, 58.4]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values_train, width, label='Training Accuracy')
rects2 = ax.bar(x + width/2, values_test, width, label='Validation Accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy of the Attempted Models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
'''


#plt.show()

logistic.set_params(**logistic_gs.best_params_)
logistic.fit(train_X, train_y)
accuracy_score(test_y, logistic.predict(test_X))


#dump(logistic, "model/logistic_fyp.joblib")


#print(test_df.loc[[20]])
