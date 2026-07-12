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

df = pd.read_csv("Housing.csv")

y = df["price"]
x = df.drop("price",axis=1)
print(df)

categorical_cols = x.select_dtypes(include=["object", "str"]).columns
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
