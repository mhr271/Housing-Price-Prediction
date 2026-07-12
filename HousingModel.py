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
