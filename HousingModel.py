import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Housing.csv")

y = df["price"]
x = df.drop("price",axis=1)
print(df)

categorical_cols = x.select_dtypes(include=["object"]).columns
numerical_cols = x.select_dtypes(include=["number"]).columns

preprocessor = ColumnTransformer(
    transformers=[ ('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical_cols),
                   ('num','passthrough',numerical_cols)

    ]
)

x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape , x_test.shape , y_train.shape , y_test.shape)

lr = Pipeline([
    ('preprocess',preprocessor),
    ('scale',StandardScaler()),
    ('model',LinearRegression())
])

lr.fit(x_train,y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)


lr_results = pd.DataFrame(
    [["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]],
    columns=["Model", "Train MSE", "Train R2", "Test MSE", "Test R2"],
)

rf = Pipeline([
    ('preprocess',preprocessor),
    ('scale',StandardScaler()),
    ('model',RandomForestRegressor(max_depth=4,random_state=42))

])

rf.fit(x_train,y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

rf_train_mse = mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2 =r2_score(y_train,y_rf_train_pred)

rf_test_mse =mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2 = r2_score(y_test,y_rf_test_pred)

rf_results = pd.DataFrame(
    [["RandomForest Regressor", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]],
    columns=["Model", "Train MSE", "Train R2", "Test MSE", "Test R2"],
)

results = pd.concat([lr_results, rf_results], ignore_index=True)
print(results.to_string(index=False))

