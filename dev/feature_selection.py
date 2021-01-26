import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix
from seaborn import scatterplot
from seaborn import lmplot, stripplot
import matplotlib.pyplot as plt

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
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

# Read in the csv file to a DataFrame
train_df = pd.read_csv("../datasets/prem_final_datasets/prem_training_set.csv")
#test_df = pd.read_csv("../datasets/prem_final_datasets/prem_test_set.csv")

train_df = train_df.sample(frac=1, random_state=2)
train_df.reset_index(drop=True, inplace=True)
#test_df = test_df.sample(frac=1, random_state=2)
#test_df.reset_index(drop=True, inplace=True)


#print(train_df.shape)

#print(train_df.head())

train_df = train_df.drop(["Date", "Home", "Score", "Away", "Venue", "MP_Home", "MP_Away"], axis=1)
#test_df = test_df.drop(["Date", "Home", "Score", "Away"], axis=1)

#print(train_df.describe(include="all"))

# Columns with tiny std:
#   GlsPer90_2_Home = 0.051949
#   AstPer90_2_Home = 0.052357
#   GlsPer90_3_Home = 0.055126
#   AstPer90_3_Home = 0.050305
#   Same for all 10 outfielders.
#
#   While these values will almost always be minuscule, they are also somewhat redundant as they are simply
#       the goals divided by 90. Taking a feature and multiplying/dividing it by a scalar is not useful as
#       this just results in new features that are linearly correlated with existing features.
#       As such, all of the per 90 minute features should be removed.

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

# Squad Stats: CrdR_Home, CrdR_Away, Sh_Away, SoT%_Away, G/Sh_Away,
# Player Stats: MP_2_Home, Min_2_Home, MP_4_Home, Min_4_Home, MP_5_Home, Min_5_Home, MP_6_Home, Min_6_Home,
#               MP_8_Home, Min_8_Home, MP_11_Home, Min_11_Home, MP_2_Away, Min_2_Away,

redundant_features = ["CrdR_Home", "D_Away", "CrdR_Away", "Sh_Away", "MP_2_Home", "Min_2_Home", "MP_4_Home",
                      "Min_4_Home", "MP_5_Home", "Min_5_Home", "MP_6_Home", "Min_6_Home", "MP_8_Home", "Min_8_Home",
                      "MP_11_Home", "Min_11_Home", "MP_2_Away", "Min_2_Away"]

train_df = train_df.drop(redundant_features, axis=1)


#test_df = test_df.drop(per_90_features, axis=1)
#print(train_df.describe(include="all"))

#print(train_df.shape)

# Now there's 125 features instead of 176


# -----DATA EXPLORATION-----
copy_df = train_df.copy()   # Copy the dataset

'''
def encode_and_bind(original_dataframe, feature_to_encode):
    # Function to one-hot encode a feature and remove the initial feature
    # Provided by user Cybernetic
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)

    return res

copy_one_hot = encode_and_bind(copy_df, "Result")   # One-hot encode the Results feature in a new dataframe
'''

label_encoder = LabelEncoder()
label_encoder.fit(train_df["Result"])
copy_df["Result"] = label_encoder.transform(copy_df["Result"])

#print(copy_df.corr())

#print(copy_one_hot.head())

#print(copy_one_hot.corr())
# This may not be the best method for feature engineering, but it will give a good indication of what features
#   may prove to useful. For example if a feature is highly correlated with a Loss, it will certainly be negatively
#   correlated with a Win.
# A better method for checking multi-class correlations would be to use a scatter matrix.

#m = scatter_matrix(copy_df, figsize=(15, 15))
#plt.show()


#plot = lmplot(x="Gls", y="W", hue="Result", data=copy_df, fit_reg=False)
#plot = lmplot(x="Poss", y="GA", hue="Result", data=copy_df, fit_reg=False)
#plot = scatterplot("Gls", "Result_W", data=copy_one_hot)

# These plots reinforce the 60% accuracy seen earlier in the base models. There's a clear lack of linear separability

# Scatter plot isn't the most useful for multiclass
#plot = scatterplot("Gls", "Result", data=copy_df)

# Stripplot conveys the same message as the lmplots, just a bit more readable
#plot = stripplot("Result", "W", data=copy_df, jitter=0.2)

#plt.show()

'''
# All Bad
#copy_df["GlsPerSCA"] = copy_df["Gls"] / copy_df["SCA"]
#copy_df["sqrt_GlsPerSCA"] = np.sqrt(copy_df["Gls"] / copy_df["SCA"])
#copy_df["CSPerSoTA"] = copy_df["CS"] / copy_df["SoTA"]
#copy_df["sqrt_CSPerSoTA"] = np.sqrt(copy_df["CS"] / copy_df["SoTA"])
copy_df["sqrt_W_Home"] = np.sqrt(copy_df["W_Home"])
copy_df["sqrt_W_Away"] = np.sqrt(copy_df["W_Away"])
copy_df["sqrt_Pts_Home"] = np.sqrt(copy_df["Pts_Home"])
copy_df["sqrt_Pts_Away"] = np.sqrt(copy_df["Pts_Away"])
copy_df["sqrt_Gls_Home"] = np.sqrt(copy_df["Gls_Home"])
copy_df["sqrt_Gls_Away"] = np.sqrt(copy_df["Gls_Away"])
copy_df["sqrt_Ast_Home"] = np.sqrt(copy_df["Ast_Home"])
copy_df["sqrt_Ast_Away"] = np.sqrt(copy_df["Ast_Away"])
#copy_df["sqrt_GA_Home"] = np.sqrt(copy_df["GA_Home"])
copy_df["sqrt_CS_Home"] = np.sqrt(copy_df["CS_Home"])
copy_df["sqrt_CS_Away"] = np.sqrt(copy_df["CS_Away"])
'''
#print(copy_df.corr())
#print(copy_one_hot.corr())
#print(copy_df.describe(include="all"))

#print(copy_df.ewm(span=12).mean())
#copy_df["ewm_W_Home"] = copy_df["W_Home"].ewm(span=2).mean()
#copy_df["ewm_W_Away"] = copy_df["W_Away"].ewm(span=2).mean()
#copy_df["ewm_W_CS_Away"] = (copy_df["CS_Away"]/copy_df["W_Away"]).ewm(span=2).mean()
#copy_df["ewm_CS_Home"] = copy_df["CS_Home"].ewm(span=2).mean()
#copy_df["ewm_CS_Away"] = copy_df["CS_Away"].ewm(span=2).mean()

copy_df["CS_per_W_Home"] = copy_df["CS_Home"]/copy_df["W_Home"]
copy_df["CS_per_L_Away"] = copy_df["CS_Away"]/copy_df["L_Away"]

copy_df["Gls_per_W_Home"] = copy_df["Gls_Home"] / copy_df["W_Home"]
#copy_df["Gls_per_W_Away"] = copy_df["Gls_Away"] / copy_df["W_Away"]
#copy_df["Gls_per_L_Home"] = copy_df["Gls_Home"] / copy_df["L_Home"]
copy_df["Gls_per_L_Away"] = copy_df["Gls_Away"] / copy_df["L_Away"]

copy_df["Gls_per_SoT_per_W_Home"] = copy_df["G/SoT_Home"] / copy_df["W_Home"]
copy_df["Gls_per_SoT_per_W_Away"] = copy_df["G/SoT_Away"] / copy_df["W_Away"]
#copy_df["Gls_per_SoT_per_L_Home"] = copy_df["G/SoT_Home"] / copy_df["L_Home"]
#copy_df["Gls_per_SoT_per_L_Away"] = copy_df["G/SoT_Away"] / copy_df["L_Away"]

copy_df["SoT_per_W_Home"] = copy_df["SoT_Home"] / copy_df["W_Home"]
#copy_df["SoT_per_W_Away"] = copy_df["SoT_Away"] / copy_df["W_Away"]
#copy_df["SoT_per_L_Home"] = copy_df["SoT_Home"] / copy_df["L_Home"]
copy_df["SoT_per_L_Away"] = copy_df["SoT_Away"] / copy_df["L_Away"]

print(copy_df.corr())
