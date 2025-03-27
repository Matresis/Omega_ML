import pandas as pd
import pickle as pc
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Load the trained model
xgb_model = pc.load(open("models/gradientboosting_model.pkl", 'rb'))

# Load the encoding mappings
with open("models/brand_encoding.pkl", "rb") as f:
    brand_encoding = pc.load(f)

# Load the feature order
with open("models/feature_order.pkl", "rb") as f:
    expected_columns = pc.load(f)

# Load the standard scaler used in training
with open("models/scaler.pkl", "rb") as f:
    scaler = pc.load(f)

# Example input data
new_data = {
    "Brand": "Ford",
    "Year": 2023,
    "Mileage": 10000,
    "Transmission": "automatic",
    "Body Type": "sedan",
    "Condition": "like new",
    "Cylinders": 8,
    "Fuel Type": "gas",
    "Title Status": "clean"
}

# Convert input to DataFrame
df_input = pd.DataFrame([new_data])

# Feature Engineering
current_year = datetime.now().year
df_input["Car_Age"] = current_year - df_input["Year"]
df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

# Encode brand using the saved encoding
df_input["Brand_Encoded"] = df_input["Brand"].map(brand_encoding).fillna(0)

# Drop 'Year' column as it's no longer needed
df_input.drop(columns=["Year", "Brand"], inplace=True)

# One-Hot Encoding for categorical variables
categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
df_input = pd.get_dummies(df_input, columns=categorical_columns)

# Ensure all expected columns exist (fill missing ones with 0)
for col in expected_columns:
    if col not in df_input.columns:
        df_input[col] = 0

# Reorder features to match the model's training data
df_input = df_input[expected_columns]

# ⚠️ Standardize numeric features (excluding "Price")
numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
df_input[numeric_features] = scaler.transform(df_input[numeric_features])

# 🚀 Convert to NumPy array to avoid sklearn warning
df_input = df_input.values

# 🔥 Make the prediction
predicted_price = xgb_model.predict(df_input)

# 🎯 Output the result
print(f"The predicted price of the car is: ${predicted_price[0]:,.2f}")
