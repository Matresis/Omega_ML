import pandas as pd
import pickle as pc
from datetime import datetime

# Load the trained model
model = pc.load(open("models/gradient_boosting_model.pkl", 'rb'))

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
    "Year": 2010,
    "Mileage": 50000,
    "Transmission": "automatic",
    "Body Type": "sedan",
    "Condition": "like new",
    "Cylinders": 6,
    "Fuel Type": "gas",
    "Title Status": "clean"
}

# Convert input to DataFrame
df_input = pd.DataFrame([new_data])

# Feature Engineering
df_input["Car_Age"] = datetime.now().year - df_input["Year"]
df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

# Encode brand
df_input["Brand_Encoded"] = df_input["Brand"].map(brand_encoding).fillna(0)

# Drop unused columns
df_input.drop(columns=["Year", "Brand"], inplace=True)

# One-Hot Encoding
categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
df_input = pd.get_dummies(df_input, columns=categorical_columns)

# Ensure all expected columns exist
for col in expected_columns:
    if col not in df_input.columns:
        df_input[col] = 0

# Reorder features
df_input = df_input[expected_columns]

# Standardize numeric features
numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
df_input[numeric_features] = scaler.transform(df_input[numeric_features])

# Predict
predicted_price = model.predict(df_input.values)

# 🎯 Output the result
print(f"The predicted price of the car is: ${predicted_price[0]:,.2f}")
