import pandas as pd
import pickle as pc
import numpy as np
from datetime import datetime

# Load trained model
model = pc.load(open("models/risk_model.pkl", 'rb'))

# Load encoding mappings
with open("models/brand_encoding.pkl", "rb") as f:
    brand_encoding = pc.load(f)

# Load feature order and scaler
with open("models/feature_order.pkl", "rb") as f:
    expected_columns = pc.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pc.load(f)

# Load risk labels
with open("models/risk_label_map.pkl", "rb") as f:
    risk_labels = pc.load(f)

# Example input
new_data = {
    "Brand": "Ford",
    "Year": 2023,
    "Mileage": 50000,
    "Transmission": "automatic",
    "Body Type": "pickup",
    "Condition": "salvage",
    "Cylinders": 6,
    "Fuel Type": "gas",
    "Title Status": "salvage",
    "Price": 0
}

# Convert input to DataFrame
df_input = pd.DataFrame([new_data])

# Feature Engineering
current_year = datetime.now().year
df_input["Car_Age"] = current_year - df_input["Year"]
df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

# Encode categorical values
df_input["Brand"] = df_input["Brand"].str.title().str.strip()
df_input["Brand_Encoded"] = df_input["Brand"].map(brand_encoding).fillna(0)

df_input.drop(columns=["Year", "Brand"], inplace=True)

# Risk Mapping
condition_map = {"new": 0, "like new": 1, "excellent": 2, "good": 3, "fair": 4, "salvage": 5}
title_risk_map = {"clean": 0, "rebuilt": 2, "salvage": 3}
body_risk_map = {"sedan": 0, "suv": 1, "coupe": 1, "hatchback": 1, "van": 2, "pickup": 2, "truck": 3}
fuel_risk_map = {"gas": 1, "diesel": 2, "hybrid": 1, "electric": 0}
transmission_risk_map = {"automatic": 1, "manual": 2}

df_input["Condition_Risk"] = df_input["Condition"].map(condition_map).fillna(3)
df_input["Title_Risk"] = df_input["Title Status"].map(title_risk_map).fillna(1)
df_input["Body_Risk"] = df_input["Body Type"].map(body_risk_map).fillna(2)
df_input["Fuel_Risk"] = df_input["Fuel Type"].map(fuel_risk_map).fillna(2)
df_input["Transmission_Risk"] = df_input["Transmission"].map(transmission_risk_map).fillna(2)

# One-Hot Encoding for categorical variables
categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
df_input = pd.get_dummies(df_input, columns=categorical_columns)

# Ensure all expected columns exist
for col in expected_columns:
    if col not in df_input.columns:
        df_input[col] = 0

# Ensure column order matches training
df_input = df_input[expected_columns]

# Standardize numeric features
numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
df_input[numeric_features] = scaler.transform(df_input[numeric_features])

# Convert to NumPy array
df_input = df_input.values

# Make the prediction
predicted_risk = model.predict(df_input)

# Convert numeric risk to descriptive label
predicted_risk_label = risk_labels[int(predicted_risk[0])]

# Output the result
print(f"🚗 The predicted risk level of the car is: **{predicted_risk_label}**")
