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
train_df = pd.read_csv("../datasets/odds/bookie_odds.csv")
test_df = pd.read_csv("../datasets/odds/bookie_odds_18.csv")

train_df = train_df.sample(frac=1, random_state=2)
train_df.reset_index(drop=True, inplace=True)
test_df = test_df.sample(frac=1, random_state=2)
test_df.reset_index(drop=True, inplace=True)

train_df = train_df.drop(["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
                          "HTR", "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY",
                          "AY", "HR", "AR", 'LBA', 'LBD', 'LBH'], axis=1)
test_df = test_df.drop(["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
                        "HTR", "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY",
                        "AY", "HR", "AR"], axis=1)

#print(train_df.describe(include="all"))

train_df = train_df.sample(frac=1, random_state=2)
train_df.reset_index(drop=True, inplace=True)
test_df = test_df.sample(frac=1, random_state=2)
test_df.reset_index(drop=True, inplace=True)

features = train_df.columns.tolist()
del features[0]

train_X = train_df[features]
test_X = test_df[features]
train_X = train_X.fillna(0)
test_X = test_X .fillna(0)

train_y = train_df["FTR"]
test_y = test_df["FTR"]


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

'''
# Majority-class Classifier
maj = DummyClassifier()
maj.fit(train_X, train_y)
print("MAJORITY CLASSIFIER -- ", accuracy_score(test_y, maj.predict(test_X)))
'''
# Using holdout with stratification
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.66, random_state=2)   # Split for validation set, 50-25-25 overall

# print(np.any(np.isnan(train_X))) # Was true so there's NaNs
# print(np.all(np.isfinite(train_X)))
# print(train_X.describe(include="all"))

'''
print("--------")
# -- kNN --

knn = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsClassifier(n_jobs=-1))])

# Dictionary for the hyperparameters, as well as the new features. I also included 3 scalers.
knn_param_grid = {"predictor__n_neighbors": [32],
                  "predictor__weights": ["uniform"],
                  "preprocessor__num__scaler__transformer": [StandardScaler()],
                  }


# This dataset is quiet large, so holdout has been used.
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

# KNN Training Accuracy:  0.5438829787234043
# KNN Validation Accuracy:  0.5360824742268041

print("--------")
# -- Logistic Regression --
logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", LogisticRegression(max_iter=500, n_jobs=-1))])  # Regularisation by default is l2, which I'm fine with.

# Dictionary for the hyperparameters, in this case C for the amount of regularisation,
# as well as the new features. I also included 3 scalers.
logistic_param_grid = {"predictor__C": [0.3212263075918594],
                       "predictor__penalty": ["l2"],
                       "preprocessor__num__scaler__transformer": [RobustScaler()],
                       }


logistic_param_dist = {"predictor__C": loguniform(1e-5, 1),
                       "preprocessor__num__scaler__transformer": [StandardScaler(), RobustScaler(), MinMaxScaler()]
                       }

# This dataset is quiet large, so holdout has been used.
#n_iter_search = 500
#logistic_rs = RandomizedSearchCV(logistic, param_distributions=logistic_param_dist, n_iter=n_iter_search, scoring="accuracy", cv=sss)
logistic_gs = GridSearchCV(logistic, logistic_param_grid, scoring="accuracy", cv=sss)


logistic_gs.fit(train_X, train_y)

print("LOGISTIC REGRESSION BEST PARAMS/SCORE: ", logistic_gs.best_params_, logistic_gs.best_score_)
# Checking for under/overfitting
logistic.set_params(**logistic_gs.best_params_)
scores = cross_validate(logistic, train_X, train_y, cv=sss, scoring="accuracy", return_train_score=True)
logistic_score_train = np.mean(scores["train_score"])
logistic_score_test = np.mean(scores["test_score"])
print("LOGISTIC Training Accuracy: ", logistic_score_train)
print("LOGISTIC Validation Accuracy: ", logistic_score_test)

logistic.set_params(**logistic_gs.best_params_)
logistic.fit(train_X, train_y)
plot = plot_confusion_matrix(logistic, test_X, test_y, cmap="Blues")
plt.title('Confusion Matrix - Logistic Regression')

# LOGISTIC Training Accuracy:  0.5518617021276596
# LOGISTIC Validation Accuracy:  0.5515463917525774
'''

# --- Column Chart Comparing Model Accuracies ---
labels = ["Logistic Reg\n(Created Dataset)", "KNN\n(Bookies Dataset)", "Logistic_Reg\n(Bookies Dataset)"]
values_train = [60, 54.3, 55.1]
values_test = [60, 53.6, 55.1]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, values_train, width, label='Training Accuracy')
rects2 = ax.bar(x + width/2, values_test, width, label='Validation Accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy of the Different Models')
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


plt.show()



