from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🚀 Load cleaned dataset
df = pd.read_csv("cleaned_craigslist_cars.csv")

year = datetime.now().year

# 🚀 Feature Engineering
df["Car_Age"] = year - df["Year"]
df["Mileage_per_Year"] = df["Mileage"] / (df["Car_Age"] + 1)  # Avoid division by zero

# 🚀 Drop unnecessary columns
df.drop(columns=["Year"], inplace=True)

# 🚀 Define features (X) and target variable (y)
X = df.drop(columns=["Price"])  # Features
y = df["Price"]  # Target variable

# 🚀 Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=42)

# 🚀 Feature Scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🚀 Train RandomForest with Hyperparameter Tuning
rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# 🚀 Train XGBoost
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)


gradient = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gradient.fit(X_train, y_train)

# Evaluate Models
models = {"RandomForest": best_rf, "XGBoost": xgb, "GradientBooster": gradient}

for name, decision in models.items():
    y_pred = decision.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 {name} Performance:")
    print(f"   MAE: {mae:.4f}")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²: {r2:.4f}\n")

# # Evaluate Models
# models = {"RandomForest": best_rf,"LinearRegression": lr, "XGBoost": xgb, "GradientBooster": gradient, "DecisionTree": decision, "AdaBooster": ada}
#
# for name, decision in models.items():
#     y_pred = decision.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#
#     print(f"📊 {name} Performance:")
#     print(f"   MAE: {mae:.4f}")
#     print(f"   MSE: {mse:.4f}")
#     print(f"   R²: {r2:.4f}\n")
