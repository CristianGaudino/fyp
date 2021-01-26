# Cristiano Gaudino - 117434292
## CS4618 Assignment


```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```


```python
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree

from seaborn import lmplot, stripplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```


```python
# Use pandas to read the CSV file into a DataFrame
df = pd.read_csv("../datasets/dataset_shopping.csv")
# Shuffle the dataset
df = df.sample(frac=1, random_state=2)
df.reset_index(drop=True, inplace=True)
```

# Read in and Check Data


```python
df.describe(include="all")
```

**Observations:**
* Max prod_num_pages value is 705. That's 705 products viewed in one session, investigate this.
* Max value for prod_duration is 64k~. That would be >17 hours spent on one session, no doubt an error.
    * In comparison, max duration for admin pages was an hour, with max duration for info pages being lower again. 
* Instructions said visitor can be new visitor, returning visitor or other. Check some of the rows that contain other.
* Only 10 months? Check these.
* Why are os, browser and region integers? I guess it doesn't matter, they would just be encoded later. Though it may be worth checking some of the higher values.
* Check some of the higher page_value values, though it's entirely possible for value to be high, the mean is only 6 so it might be worth checking.


```python
df.shape
```


```python
df.dtypes
```

**Remarks:**
* Weekend and Purchase are booleans. Does this mean that they'll have some special functionality when it comes to feature engineering? In comparison, in the previous lab regarding alchohol consumption, the target was a String.


```python
df.head()
```


```python
df[df["prod_num_pages"] >= 500]
```

**Remarks:**
* Not sure what to make here, 12 hours spent on product pages in one session with 705 product pages visited, the fact that a purchase wasn't made makes me suspect a bot or something. Will remove this entry.
* Also note that one user visited 518 product pages, but the average page value was 0. Definitely an error here, will check other records with a page value of 0. 
    * *Note that at the time I assumed page_value was an average of all page values in the session, I know now what it means and I've decided not to check it but I'll leave this remark in anyway.*
* The other rows here also have a high prod_num_pages value, but the prod_duration is some bit believable in comparison to the 705 prod_num_pages entry. 


```python
df[df["prod_duration"] >= 30000]
```

**Remarks:**
* The entry here with 43 thousand seconds on prod_duration was discussed above.
* The other entry seems like an error or a bot, as stated previously this is a 17 hour session. This will be removed.


```python
df = (df[df["prod_duration"] < 30000]).copy() # Removing the 2 entries discussed above.
```

**While investigating the entries with a page_value of 0, I noticed that some entries that had no product pages visited had purchase set to True, I will check this now.**


```python
no_prod_pages = df[df["prod_num_pages"] == 0]
no_prod_pages[no_prod_pages["purchase"] == True] # No product pages viewed in the session and a purchase made
```

**Remarks:** 
* A few interesting things to note here, in some cases above, a user makes a purchase without viewing any product pages in the current session. This is plausible as the user may be returning to their cart to make a purchase, this is backed up by the fact that in all of these cases the user only visits 1 or 2 admin pages and no other pages, indicating login and checkout. However, there are some new visitors that make a purchase in the session with no product pages viewed, this seems incorrect to me and I will remove these entries.
* *The above observation is incorrect (but I've left it in regardless), I have realised since that the visitor entry must refer to the current page itself, whereas before I was assuming it referred to the website the page is on. I won't be removing these entries as it must be the case that the new_visitor entries are using the checkout for the first time.*


```python
# Note that here I started at a value of 100, which is when i noticed the below remark,
# at which point I increased the value.
df[df["page_value"] >= 250] 
```

**Remark: There seems to be a really high correlation here between high page values and purchase being true. While it raised suspicion earlier, I don't see any reason why a page_value can't be this high.**


```python
# Checking visitors set to other
df["visitor"].unique()
```


```python
other_visitor = df[df["visitor"] == "Other"] # Check other vistors and purchase equals true
other_visitor[other_visitor["purchase"] == True]
```

**Remarks: I had checked entries where visitor was Other, and it seemed like all entries did not result in a purchase. But after checking above, there doesn't seem to be anything wrong here.**


```python
df["month"].unique()
```

**Remarks: Jan and Apr are not included, not sure if this matters or not.**


```python
df[df["browser"] >= 10]
```

**Remark: Doesn't seem to be anything weird here**

**Later on I realised that some columns such as "browser", shouldn't be stored as an integer. The model may assume that a browser of value 10 is more closely related to a browser of value 11 than one of value 3. This is incorrect, and as such I decided to store them as a string instead, this way it can be one-hot encoded later.**


```python
df['operating_system'] = df['operating_system'].astype(str)
df['browser'] = df['browser'].astype(str)
df['region'] = df['region'].astype(str)
df['referrer'] = df['referrer'].astype(str)
```


```python
df.dtypes
```


```python
df.describe(include="all")
```


```python
df.reset_index(drop=True, inplace=True)
```

# Create a Test Set


```python
# Using 20% for the test set
rest_of_df, test_df = train_test_split(df, train_size=0.8, stratify=df["purchase"], random_state=2)
```

# Dataset Exploration


```python
copy_df = rest_of_df.copy()
```


```python
copy_df.corr()
```

**Remarks:**
* Kind of unexpected here but booleans seem to work with correlation. This should make feature engineering a lot clearer.
* As expected, page value has a high correlation with the target.
* Exit rate is the next highest correlation, however it's still only -0.2.
* Referrer seems redundant at -0.006, but it may be useful in feature engineering.


```python
plot = lmplot(x="exit_rate", y="page_value", hue="purchase", data=copy_df, fit_reg=False)
```


```python
plot = stripplot("purchase", "exit_rate", data=copy_df, jitter=0.2)
```

**In general, pages with a high exit_rate will never result in a purchase. But a low exit rate can be either.**

**Results of trying different features:**
* square root of exit_rate = -0.24
* exit_rate squared = -0.17
* page_value squared = 0.22
* square root of page_value = 0.62
* square root of exit_rate over square root of page_value = -0.24
    * Went back to this and modified it slightly to be page_value over exit_rate sqrt which gave 0.38.
    * This is pretty similar to what I used for page_value and special_day but nonetheless, I then tried square root of (page_value over exit_rate) which gave 0.54
* exit_rate over page_value = -0.26
* page_value over special_day = 0.59 (tried special_day/page_value and it was terrible)
    * Then tried page_value over sqrt.special_day which gave 0.66
    * I then tried square root of (page_value over special_day) which gave 0.70
* I also tried a variety of features, such as averages,using admin, info and prod pages/duration but couldn't get a good result.
* Finally I tried checking online for anything that may point towards a new feature, but most of the data was centered around discounts and how long an item has been left in a user's cart. At this point I decided to just go with the 3 features below.



```python
copy_df.dtypes
```


```python
copy_df.isnull().values.any()
```


```python
copy_df["page_value_sqrt"] = np.sqrt(copy_df["page_value"])
```


```python
copy_df["sqrt_page_value_over_special_day"] = np.sqrt(copy_df["page_value"] / copy_df["special_day"])
```


```python
copy_df["sqrt_page_value_over_exit_rate"] = np.sqrt(copy_df["page_value"] / copy_df["exit_rate"])
```

### Investigating the page_value correlations a bit more:


```python
plot = stripplot("purchase", "page_value_sqrt", data=copy_df, jitter=0.2)
```


```python
plot = stripplot("purchase", "sqrt_page_value_over_special_day", data=copy_df, jitter=0.2)
```

**Remarks:** 
* Interesting that if square root (page value over special) is 0, then purchase will always be false. The same also seems to hold for page_value over square root of special_day, but this is not true for square root of page_value.

**Note:** Later on during model selection I was having issues with infinite and NaN values on the added features. As such, I decided to investigate why here. However, after some troubleshooting, I realised that my solution to this issue was incorrect, and I decided to just replace all infinite and NaN values with 0. While I would have like to do the below solution, I was unable to get it working. As far as I can tell the fillna method to replace NaN's with the mean simply wasn't doing anything.

So the next few cells are no longer relevant, but I decided to leave them in anyway to show the testing I did to try solve the issue of infinite and NaN values.

**Handling NaN and infinite values.** After checking the values of the below copy_df.head(), I came up with the following:
* page_value = 0 divided by special_day = 0 results in NaN.
    * Will replace NaN values by 0
* A page_value >0 divided by special_day = 0 results in infinite.
    * Decided to replace infinite values by the mean. I did this by first replacing them with NaN and then replacing NaN values with the mean.
* page_value = 0 divided by special_day >0 results in 0.


```python
copy_df.isnull().values.any() # Check if there are NaN values present
```


```python
copy_df.head()
```


```python
copy_df = copy_df.replace( [ np.nan ], 0 ) # First replace the NaNs with 0
```


```python
# Replace infinite values with NaNs, this will allow me to set infinite values to the mean
copy_df = copy_df.replace( [ np.inf, -np.inf ], np.nan )
```


```python
copy_df.mean() # Checking if this works before I use it.
```


```python
 copy_df.fillna(copy_df.mean()) # Replace any NaNs with the mean.
```


```python
# At this point i realised there was something wrong, and as stated above, after spending time trying to fix it,
# I eventually abandoned the idea of having infinite values set to the mean.
copy_df.isnull().values.any() 
```

# Model Selection


```python
features = rest_of_df.columns.tolist() # pandas method to get all column names
features = features[:-1] # remove the target column from the end of the features list
```


```python
# Extract the features but leave as a DataFrame
rest_of_X = rest_of_df[features]
test_X = test_df[features]

# Target values, encoded and converted to a 1D numpy array
label_encoder = LabelEncoder()
label_encoder.fit(df["purchase"])
rest_of_y = label_encoder.transform(rest_of_df["purchase"])
test_y = label_encoder.transform(test_df["purchase"])

```


```python
class InsertPageValueSqrt(BaseEstimator, TransformerMixin):

    def __init__(self, insert=True):
        self.insert = insert
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.insert:
            X["page_value_sqrt"] = np.sqrt(X["page_value"])
        return X

class InsertSqrtPageValueOverSpecial(BaseEstimator, TransformerMixin):

    def __init__(self, insert=True):
        self.insert = insert
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.insert:
            X["sqrt_page_value_over_special_day"] = np.sqrt(X["page_value"] / X["special_day"])
            # Reasoning for doing these replacements was discussed above.
            X = X.replace( [ np.inf, -np.inf ], np.nan )
            X = X.replace( [ np.nan ], 0 ) 
        return X

class InsertSqrtPageValueOverExit(BaseEstimator, TransformerMixin):

    def __init__(self, insert=True):
        self.insert = insert
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.insert:
            X["sqrt_page_value_over_exit"] = np.sqrt(X["page_value"] / X["exit_rate"])
            # Reasoning for doing these replacements was discussed above.
            X = X.replace( [ np.inf, -np.inf ], np.nan ) 
            X = X.replace( [ np.nan ], 0 )
        return X
```


```python
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
```

### Preprocessor


```python
numeric_features = ["admin_num_pages", "admin_duration", "info_num_pages", "info_duration", 
                    "prod_num_pages", "prod_duration", "bounce_rate", "exit_rate", "page_value", 
                    "special_day"]
nominal_features = ["month","operating_system", "browser", "region", "referrer", "visitor", "weekend"]
preprocessor = ColumnTransformer([
        ("num", Pipeline([("page_value_sqrt", InsertPageValueSqrt()),
                          ("sqrt_page_value_over_special_day", InsertSqrtPageValueOverSpecial()),
                          ("sqrt_page_value_over_exit", InsertSqrtPageValueOverExit()),
                          ("scaler", TransformerFromHyperP())]),
                     numeric_features),
        ("nom", Pipeline([("binarizer", OneHotEncoder(handle_unknown="ignore"))]),
                     nominal_features)],
        remainder="passthrough")
```


```python
# Majority-class Classifier
maj = DummyClassifier()
maj.fit(rest_of_X, rest_of_y)
accuracy_score(test_y, maj.predict(test_X))
```

**Remarks: ~74% accuracy is the baseline.**


```python
# The dataset is very large, as such holdout should be used over kfold
ss = ShuffleSplit(n_splits = 1, train_size = 0.8, random_state = 2)
# Using holdout with stratification
sss = StratifiedShuffleSplit(n_splits = 1, train_size = 0.8, random_state = 2)
```

### kNN


```python
# Starting with kNN
knn = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", KNeighborsClassifier())])

# Dictionary for the hyperparameters, as well as the new features. I also included 3 scalers.
knn_param_grid = {"predictor__n_neighbors": [14, 15, 16],
                  "preprocessor__num__page_value_sqrt__insert": [False],
                  "preprocessor__num__sqrt_page_value_over_special_day__insert": [True],
                  "preprocessor__num__sqrt_page_value_over_exit__insert": [False],
                  "preprocessor__num__scaler__transformer": [RobustScaler()],
                 }
# This dataset is quiet large, so holdout has been used.
knn_gs = GridSearchCV(knn, knn_param_grid, scoring="accuracy", cv=sss)

knn_gs.fit(rest_of_X, rest_of_y)

knn_gs.best_params_, knn_gs.best_score_
```

**Remarks:**

For time's sake I set the dictionary to reflect these results:
* Neighbours:
    * Initially tried with neighbours 1, 5, 10. 10 was the best here so increased to 10, 15, 20.
    * 15 was the best here so I then tried 13, 14, 15, 16, 17. 15 was still the best.
* Scaler:
    * In the grid search dictionary I had transformer set to StandardScaler(), MinMaxScaler(), RobustScaler(). RobustScaler() was best so I removed the other 2 from the dictionary.
* Additional Features:
    * page_value_sqrt: Not used
    * sqrt_page_value_over_exit: Not used
    * sqrt_page_value_over_special_day: Used

**Result:** 89%


```python
# Checking for under/overfitting
knn.set_params(**knn_gs.best_params_)
scores = cross_validate(knn, rest_of_X, rest_of_y, cv=sss,
                         scoring="accuracy", return_train_score=True)
print("Training Accuracy: ", np.mean(scores["train_score"]))
print("Validation Accuracy: ", np.mean(scores["test_score"]))
```

**Remarks:** No under/overfitting here.

### Logistic Regression


```python
# Logistic Regression
logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", LogisticRegression(max_iter=2500))]) # Regularisation by default is l2, which I'm fine with.

# Dictionary for the hyperparameters, in this case C for the amount of regularisation, 
# as well as the new features. I also included 3 scalers.
logistic_param_grid = {"predictor__C": [0.7],
                       "preprocessor__num__page_value_sqrt__insert": [True],
                       "preprocessor__num__sqrt_page_value_over_special_day__insert": [False],
                       "preprocessor__num__sqrt_page_value_over_exit__insert": [True],
                       "preprocessor__num__scaler__transformer": [RobustScaler()],
                 }
# This dataset is quiet large, so holdout has been used.
logistic_gs = GridSearchCV(logistic, logistic_param_grid, scoring="accuracy", cv=sss)

logistic_gs.fit(rest_of_X, rest_of_y)

logistic_gs.best_params_, logistic_gs.best_score_
```

**Remarks:**
* Note the max_iter parameter used in the pipeline. I was having some issues with logistic regression here but increasing the max iterations helped with the problem. But in the interest of saving time, I have reduced it and set some of the grid search values to the best params which I'll go through now.

For time's sake I set the dictionary to reflect these results:

* Grid Search:
    * C:
        * Originally set to 0.2, 0.3, 0.4, 0.5. 0.5 was best.
        * Updated to try 0.5, 0.6, 0.7, 0.8. 0.7 was best so I stopped here.
    * Features:
        * page_value_sqrt: Used
        * sqrt_page_value_over_exit: Used
        * sqrt_page_value_over_special_day: Not Used
    * Scaling:
        * In the grid search dictionary I had transformer set to StandardScaler(), MinMaxScaler(), RobustScaler(). RobustScaler() was best so I removed the other 2 from the dictionary

**Result:** Just below 90%.


```python
# Checking for under/overfitting
logistic.set_params(**logistic_gs.best_params_)
scores = cross_validate(logistic, rest_of_X, rest_of_y, cv=sss,
                         scoring="accuracy", return_train_score=True)
print("Training Accuracy: ", np.mean(scores["train_score"]))
print("Validation Accuracy: ", np.mean(scores["test_score"]))
```

**Remarks:** No under/overfitting here

**Regarding using multinomial:** I considered trying out some models for this, but during one of the lectures a student asked if there was any need to use multinomial for a binary target, to which it was stated that the result given is basically the same. As such I decided not try these.

### Decision Tree


```python
# Decision Tree
# Preprocessor is needed as we need the new features and we need to one-hot encode nominal features.
tree = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", DecisionTreeClassifier())])

# Dictionary for the hyperparameters, in this case max depth. While it's not needed for 
# decision trees, the same preprocessor is used here, so there's scalers for the transformer.
tree_param_grid = {"predictor__max_depth": [3, 4],
                   "preprocessor__num__page_value_sqrt__insert": [True],
                   "preprocessor__num__sqrt_page_value_over_special_day__insert": [True],
                   "preprocessor__num__sqrt_page_value_over_exit__insert": [False],
                   "preprocessor__num__scaler__transformer": [MinMaxScaler()],
                 }
# This dataset is quiet large, so holdout has been used.
tree_gs = GridSearchCV(tree, tree_param_grid, scoring="accuracy", cv=sss)

tree_gs.fit(rest_of_X, rest_of_y)

tree_gs.best_params_, tree_gs.best_score_
```

**Remarks:**

For time's sake I set the dictionary to reflect these results:
* Max Depth:
    * Initially tried 1,2, 3. 3 was the best here so set to 3, 5, 7.
    * 5 was the best here so I then tried 4, 5, 6. 4 was best here.
    * Finally set to 3, 4. 4 was best.
* Scaler:
    * The data doesn't need to be scaled for decision trees, but because of the preprocessor I had to include a scaler for transform. For the sake of it, I tried all 3 and MinMaxScaler() was chosen.
* Additional Features:
    * page_value_sqrt: Used
    * sqrt_page_value_over_exit: Not used
    * sqrt_page_value_over_special_day: Used

**Result:** 90%, slightly better than logistic regression.


```python
# Checking for under/overfitting
tree.set_params(**tree_gs.best_params_)
scores = cross_validate(tree, rest_of_X, rest_of_y, cv=sss,
                         scoring="accuracy", return_train_score=True)
print("Training Accuracy: ", np.mean(scores["train_score"]))
print("Validation Accuracy: ", np.mean(scores["test_score"]))
```

**Remarks:** No over/underfitting here.

### Random Forest


```python
# Random Forest
# Preprocessor is needed as we need the new features and we need to one-hot encode nominal features.
forest = Pipeline([
    ("preprocessor", preprocessor),
    ("predictor", RandomForestClassifier())])

# Dictionary for the hyperparameters, in this case n_estimators (the number of trees used, default being 100), 
# max_depth and random_state. 
# As with decision trees, a scaler isn't needed, but the same preprocessor is used.
forest_param_grid = {"predictor__max_depth": [5, 6, 7],
                     "predictor__n_estimators": [34, 35, 36],
                     "predictor__random_state": [1, 2, 3],
                     "preprocessor__num__page_value_sqrt__insert": [True],
                     "preprocessor__num__sqrt_page_value_over_special_day__insert": [False],
                     "preprocessor__num__sqrt_page_value_over_exit__insert": [True],
                     "preprocessor__num__scaler__transformer": [MinMaxScaler()],
                 }
# This dataset is quiet large, so holdout has been used.
forest_gs = GridSearchCV(forest, forest_param_grid, scoring="accuracy", cv=sss)

forest_gs.fit(rest_of_X, rest_of_y)

forest_gs.best_params_, forest_gs.best_score_
```

**Remarks:**

* The grid search was very misleading here, it seemed to just go for the max value every time, which of course would lead to huge overfitting. 

* I tried so many combinations here that there's no point listing them all like I did before, the main problem I encountered here was that grid search seemed happy to overfit as much as possible, eventually I to limit the hyperparameters myself.

* Some of the last few results before the final hyperparameters were found were:
    * Max depth 6, n_estimators 30 (I used 20, 30, 70, 100 to find this), and random state 1. This was overfitting by about .3% which is good, but I tried a few more before settling for this.
    * Max depth 6, n_estimators 35 and random state 1 gave a better result again, and wasn't overfitting at all which is great. Here I used quite large steps again for the hyperparameters so I thought to try focusing them to see if I could improve by even a small amount.
    * The final values I used for grid search are displayed in the code above (Note that I set the features to the best value for better running time), the result from the bullet point above was still the best.

* Scaler:
    * The data doesn't need to be scaled for decision trees, but because of the preprocessor I had to include a scaler for transform. To save time I just used MinMaxScaler(), as that was best for the Decision Tree.

* Additional Features:
    * I set the dictionary to reflect these results to save time. Note that I set these once I found the best values for max_depth, n_estimators and random_state.
    * page_value_sqrt: Used
    * sqrt_page_value_over_exit: Not used
    * sqrt_page_value_over_special_day: Used

**Result:** 91%, The best model by rougly 1%


```python
# Checking for under/overfitting
forest.set_params(**forest_gs.best_params_)
scores = cross_validate(forest, rest_of_X, rest_of_y, cv=sss,
                         scoring="accuracy", return_train_score=True)
print("Training Accuracy: ", np.mean(scores["train_score"]))
print("Validation Accuracy: ", np.mean(scores["test_score"]))
```

**Remarks:** No over/underfitting here.

## Training on Test Set and Saving

To me, Random Forest seems to be the best, it had the highest accuracy on the training set and it isn't over or underfitting.


```python
# Final steps to do, train the model on the training and validation set and then test on the test set
forest.set_params(**forest_gs.best_params_)
forest.fit(rest_of_X, rest_of_y)
accuracy_score(test_y, forest.predict(test_X))
```
