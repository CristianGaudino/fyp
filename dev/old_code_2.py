import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

from joblib import dump, load


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("../datasets/season-1819_csv.csv")

# Shuffle the dataset
df = df.sample(frac=1, random_state=2)
df.reset_index(drop=True, inplace=True)

removal_list = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "Referee", "Div", "Date", "Time", "FTHG", "FTAG", "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]

with open("../datasets/season-1819_csv.csv") as f:
    first_line = f.readline().strip()
all_features = first_line.split(",")

features = [x for x in all_features if x not in removal_list]

df = df[features]   # Remove redundant features
print(df.shape)
#print(df.columns)
print(df.dtypes)
print(df.head())
print(df.describe(include="all"))

rest_of_df, test_df = train_test_split(df, train_size=0.8, random_state=2)  # Split test set, 20% of dataset

# Target values, encoded and converted to a 1D numpy array
label_encoder = LabelEncoder()
rest_y = label_encoder.fit_transform(rest_of_df["FTR"])

copy_df = rest_of_df.copy()
print(copy_df.corr())


