# libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# create dataframe from csv
df = pd.read_csv('data_for_exercise.csv')

# set target and predictor columns
target = ['attendance'] 
predictors = list(set(list(df.columns))-set(target)-set(['Unnamed: 0']))

# create training and test sets
X = df[predictors].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_r2 = r2_score(y_test, lin_reg.predict(X_test))
print("Linear (All IVs): " + str(lin_reg_r2))

# ridge regression
rid_reg = Ridge(alpha=0.01)
rid_reg.fit(X_train, y_train)
rid_reg_r2 = r2_score(y_test, rid_reg.predict(X_test))
print("Ridge (All IVs): " + str(rid_reg_r2))

# lasso regression
las_reg = Lasso(alpha=0.01)
las_reg.fit(X_train, y_train)
las_reg_r2 = r2_score(y_test, las_reg.predict(X_test))
print("Lasso (All IVs): " + str(las_reg_r2))

print("\n")

# determine difference in R2 without each IV
for p in list(set(list(df.columns))-set(target)-set(['Unnamed: 0'])):
    print(p)
    predictors.remove(p)
    
    # create training and test sets
    X = df[predictors].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    new_lin_reg_r2 = r2_score(y_test, lin_reg.predict(X_test))
    print("Linear: " + str(lin_reg_r2 - new_lin_reg_r2))

    # ridge regression
    rid_reg = Ridge(alpha=0.01)
    rid_reg.fit(X_train, y_train)
    new_rid_reg_r2 = r2_score(y_test, rid_reg.predict(X_test))
    print("Ridge: " + str(rid_reg_r2 - new_rid_reg_r2))

    # lasso regression
    las_reg = Lasso(alpha=0.01)
    las_reg.fit(X_train, y_train)
    new_las_reg_r2 = r2_score(y_test, las_reg.predict(X_test))
    print("Lasso: " + str(las_reg_r2 - new_las_reg_r2))

    print("\n")
    predictors = list(set(list(df.columns))-set(target)-set(['Unnamed: 0']))

