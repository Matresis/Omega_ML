import pandas as pd
import pickle as pc
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/cleaned_craigslist_cars.csv")

# Define features (X) and target variable (y)
X = df.drop(columns=["Price"])  # Features
y = df["Price"]  # Target variable

# Check for NaN or infinite values
print("NaN values in X:", X.isna().sum())
print("Infinite values in X:", np.isinf(X).sum())

# Replace NaN values with median
X = X.fillna(X.median())

# Replace infinite values with a large number
X.replace([np.inf, -np.inf], 1e10, inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Save the feature order after preprocessing
with open("models/feature_order.pkl", "wb") as f:
    pc.dump(X.columns.tolist(), f)

# Train Random Forest with Hyperparameter Tuning
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# 🎯 XGBoost
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

# 🎯 Gradient Boosting
gradient = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gradient.fit(X_train, y_train)

# 🎯 Decision Tree
decision = DecisionTreeRegressor(max_depth=10, random_state=42)
decision.fit(X_train, y_train)

# 🎯 AdaBoost
ada = AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
ada.fit(X_train, y_train)

# 🎯 Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# 🚀 Evaluate Models
models = {
    "RandomForest": best_rf,
    "XGBoost": xgb,
    "GradientBoosting": gradient,
    "DecisionTree": decision,
    "AdaBoost": ada,
    "LinearRegression": lr
}

print("\n📊 Model Performance:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"🔹 {name}:")
    print(f"   MAE: {mae:.2f}")
    print(f"   MSE: {mse:.2f}")
    print(f"   R² Score: {r2:.4f}\n")

# 🚀 Save Models
for name, model in models.items():
    filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
    pc.dump(model, open(filename, 'wb'))
    print(f"✅ Model saved: {filename}")

print("\n🎯 All models trained and exported successfully!")
