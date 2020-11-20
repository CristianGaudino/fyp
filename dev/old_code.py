import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error


from joblib import dump, load

df = pd.read_csv("../datasets/dataset_prem_2019.csv")
df = df.replace("H", 0)
df = df.replace("D", 1)
df = df.replace("A", 2)

# Shuffle the dataset
df = df.sample(frac=1, random_state=2)
df.reset_index(drop=True, inplace=True)

removal_list = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "Referee", "Div", "Date", "Time", "FTHG", "FTAG", "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]

with open("../datasets/dataset_prem_2019.csv") as f:
    first_line = f.readline().strip()
all_features = first_line.split(",")

features = [x for x in all_features if x not in removal_list]
'''
X = df[features]    # Extract the features but leave as a DataFrame
print(X.head())
print(X.shape)
y = X["FTR"].values    # Target values, converted to a 1D numpy array
print(X.describe(include="all"))
'''
# print(df.head())
# print(df.describe(include="all"))
# print(df.dtypes)





# Create a preprocessor
preprocessor = ColumnTransformer([
        ("scaler", StandardScaler(), features)],
        remainder="passthrough")

rest_of_df, test_df = train_test_split(df, train_size=0.8, random_state=2)  # Split test set, 20% of dataset
# Extract features, leave as a dataframe
rest_of_X = rest_of_df[features]
test_X = test_df[features]
# Target values
rest_of_y = rest_of_df["FTR"].values
test_y = test_df["FTR"].values


# Object to shuffle and split rest of data. Training and validation set.
ss = ShuffleSplit(n_splits=10, train_size=0.75, random_state=2)

# Create a pipeline that combines the preprocessor with 3NN
knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsRegressor(n_neighbors=1))])

'''
# HYPER PARAMETER SELECTION, 1 was best neighbors after narrowing down and using kfold over holdout
# Dictionary of hyper-parameters to try
param_grid = {"predictor__n_neighbors": [1, 2, 3, 4, 5, 6, 7]}
# Grid search object which will find the best hyper-parameter based on validation error
gs = GridSearchCV(knn_model, param_grid, scoring="neg_mean_absolute_error", cv=ss)
# Run grid search by calling fit
gs.fit(rest_of_X, rest_of_y)

print(gs.best_params_, gs.best_score_)
'''


# Testing kNN on training and validation set
knn = np.mean(cross_val_score(knn_model, rest_of_X, rest_of_y, scoring="neg_mean_absolute_error", cv=ss))
#print("1NN: ", knn)

# Create a pipeline that combines the preprocessor with the linear model
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", LinearRegression())])

# Testing linear on training and validation set
linear = np.mean(cross_val_score(linear_model, rest_of_X, rest_of_y, scoring="neg_mean_absolute_error", cv=ss))
#print("Linear: ", linear)


# QUADRATIC
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
    ("predictor", LinearRegression())
])

# CHECKING THE FITTING
#scores = cross_validate(poly_model, rest_of_X, rest_of_y, cv=ss, scoring="neg_mean_absolute_error", return_train_score=True)
#print("Training Error: ", np.mean(np.abs(scores["train_score"])))
#print("Validation Error: ", np.mean(np.abs(scores["test_score"])))
'''
# HYPER-PARAMETER SELECTION.
# Dictionary of hyper-parameters to try
param_grid = {"predictor__degrees": [1, 3, 5]}
# Grid search object which will find the best hyper-parameter based on validation error
gs = GridSearchCV(quadratic_model, param_grid, scoring="neg_mean_absolute_error", cv=ss)
# Run grid search by calling fit
gs.fit(rest_of_X, rest_of_y)

print(gs.best_params_, gs.best_score_)
'''
#quadratic = np.mean(cross_val_score(quadratic_model, rest_of_X, rest_of_y, scoring="neg_mean_absolute_error", cv=ss))
#print("Quadratic: ", quadratic)

# Last thing to do once I have a good model is to do error estimation on the training set.
# Then train on the whole set and save the model



# 3 degrees, training error 1.6, val error 0.3
# 2 degrees, training 4, val 0.03
# 1 degree, training 1.65, val 2


scaler = StandardScaler()
rest_of_X = scaler.fit_transform(rest_of_X)
train_df, valid_df = train_test_split(rest_of_X, train_size=0.75, random_state=2)
train_y = train_df["FTR"].values
valid_y = valid_df["FTR"].values

ridge = Ridge(alpha=0)
ridge.fit(train_df, train_y)

y_predicted_ridge = ridge.predict(valid_df)
mse_ridge = mean_squared_error(y_predicted_ridge, valid_y)
print(mse_ridge)
print(ridge.intercept_)
print(ridge.coef_)
