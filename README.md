# Housing Price Prediction
This model predicts **house prices** based on property features, using a housing dataset containing details like area, number of bedrooms/bathrooms, and various amenities.
## Overview
Two regression models are trained and compared:
- **Linear Regression** — a baseline model that assumes a linear relationship between the house features and price.
- **Random Forest Regressor** — an ensemble tree-based model capable of capturing more complex, non-linear patterns in the data.
Both models are evaluated and compared using **Mean Squared Error (MSE)** and **R² Score** on both the training and test sets.
## Dataset
This project uses `Housing.csv`, which contains the following features for each property:
| Feature | Description |
|---|---|
| `price` | Sale price of the house (target variable) |
| `area` | Area of the house (sq. ft.) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether the house faces a main road (yes/no) |
| `guestroom` | Whether the house has a guest room (yes/no) |
| `basement` | Whether the house has a basement (yes/no) |
| `hotwaterheating` | Whether the house has hot water heating (yes/no) |
| `airconditioning` | Whether the house has air conditioning (yes/no) |
| `parking` | Number of parking spaces |
| `prefarea` | Whether the house is in a preferred area (yes/no) |
| `furnishingstatus` | Furnishing status (furnished / semi-furnished / unfurnished) |
545 rows total — 436 used for training, 109 held out for testing (80/20 split, `random_state=42`).
## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
## Results
| Model | Train MSE | Train R² | Test MSE | Test R² |
|---|---|---|---|---|
| Linear Regression | 9.68e+11 | 0.686 | 1.75e+12 | **0.653** |
| Random Forest Regressor (max_depth=4) | 8.51e+11 | 0.724 | 2.31e+12 | **0.544** |
![Predicted vs Actual Price](housing_predictions.png)
**Linear Regression outperforms Random Forest on the test set**, despite Random Forest fitting the training data slightly better. The gap between Random Forest's train R² (0.724) and test R² (0.544) is a sign of mild overfitting relative to Linear Regression, which generalizes more consistently.
Both models tend to compress the price range — overestimating cheaper houses and underestimating more expensive ones, visible as points drifting away from the red diagonal (perfect prediction) at the high end of the price scale.
## Notes
- Since `price` is a continuous value, this is a **regression** problem — Linear Regression and Random Forest Regressor are both appropriate. (Logistic Regression is not used here, as it's meant for classification, not predicting continuous prices.)
- `RandomForestRegressor` is intentionally limited to `max_depth=4` here. A deeper forest or more estimators may close the gap with Linear Regression, but risks further overfitting on a dataset of this size.
- Feel free to experiment with different hyperparameters, log-transforming `price` (housing prices are typically right-skewed), other models (e.g., Gradient Boosting, XGBoost), or additional feature engineering.
