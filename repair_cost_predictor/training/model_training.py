import pandas as pd
import pickle as pc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, \
    mean_absolute_error
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/cleaned_craigslist_cars_repair.csv")

# Define features (X) and target variables for both repair need and repair cost
X = df.drop(columns=["Repair Cost"])  # Features
y = df["Repair Cost"]  # Target for regression (repair cost)

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

# Best Model for Regression: XGBoostRegressor
xgb_regressor = XGBRegressor(random_state=42, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=1.0)
xgb_regressor.fit(X_train, y_train)

# 🎯 Evaluate Regression Model (Repair Cost)
print("\n📊 Repair Cost Model Performance:")
y_pred = xgb_regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"🔹 XGBoostRegressor:")
print(f"   Mean Absolute Error: {mae:.4f}")
print(f"   Mean Squared Error: {mse:.4f}")
print(f"   Predicted Repair Costs: {y_pred[:5]}\n")

pc.dump(xgb_regressor, open("models/repair_cost_model.pkl", 'wb'))
print(f"✅ XGBoost model for repair cost saved: models/xgboost_repair_cost_model.pkl")

print("\n🎯 All models trained and exported successfully!")