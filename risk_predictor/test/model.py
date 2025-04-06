import pandas as pd
import pickle as pc
import numpy as np

# Load trained AI model
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
    "Year": 2000,
    "Mileage": 150000,
    "Transmission": "automatic",
    "Body Type": "pickup",
    "Condition": "salvage",
    "Cylinders": 6,
    "Fuel Type": "gas",
    "Title Status": "salvage",
    "Price": 10000
}

# Convert input to DataFrame
df_input = pd.DataFrame([new_data])

# Normalize text data
df_input["Brand"] = df_input["Brand"].str.title().str.strip()
df_input["Condition"] = df_input["Condition"].str.lower().replace("like new", "excellent")
df_input["Fuel Type"] = df_input["Fuel Type"].str.lower().str.strip()
df_input["Transmission"] = df_input["Transmission"].str.lower().str.strip()
df_input["Body Type"] = df_input["Body Type"].str.lower().str.strip()
df_input["Title Status"] = df_input["Title Status"].str.lower().str.strip()

# Convert numerical values
df_input["Price"] = pd.to_numeric(df_input["Price"], errors="coerce").fillna(0)
df_input["Mileage"] = pd.to_numeric(df_input["Mileage"], errors="coerce").fillna(df_input["Mileage"].median())
df_input["Cylinders"] = pd.to_numeric(df_input["Cylinders"], errors="coerce").fillna(df_input["Cylinders"].median())

# Handle missing values in categorical columns
default_mappings = {
    "Transmission": "automatic",
    "Body Type": "sedan",
    "Condition": "good",
    "Fuel Type": "gas",
    "Title Status": "clean"
}
for col, default_value in default_mappings.items():
    df_input[col] = df_input[col].fillna(default_value)

# Feature Engineering
CURRENT_YEAR = 2025
df_input["Car_Age"] = CURRENT_YEAR - df_input["Year"]

# Encode Brand using precomputed mapping
df_input["Brand_Encoded"] = df_input["Brand"].map(brand_encoding).fillna(0)

# Drop redundant columns
df_input.drop(columns=["Year", "Brand"], inplace=True, errors="ignore")

# One-Hot Encoding for categorical variables
categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
df_input = pd.get_dummies(df_input, columns=categorical_columns)

# Ensure all expected columns exist
for col in expected_columns:
    if col not in df_input.columns:
        df_input[col] = 0  # Add missing columns with 0

# Standardize numeric features
numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
df_input[numeric_features] = scaler.transform(df_input[numeric_features])

# Ensure column order matches training
df_input = df_input[expected_columns]

# Debugging encoding process
print("Brand Encoding:", df_input["Brand_Encoded"].head())
print("Condition Risk:", df_input.get("Condition_Risk", "Column not found").head())  # Safe access
print("Title Status Risk:", df_input.get("Title_Risk", "Column not found").head())  # Safe access

# Debugging the processed input features
print("Processed Input Data:\n", df_input.head())

# Convert to NumPy array
df_input_np = df_input.values  # Store separately to avoid overwriting DataFrame

# Make the prediction
predicted_risk = model.predict(df_input_np)

# Reverse the dictionary to use numbers as keys
risk_labels = {v: k for k, v in risk_labels.items()}

# Debugging prediction and labels
print(risk_labels)  # Debugging: should print {0: 'Very Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
print(f"Predicted numeric risk value: {int(predicted_risk[0])}")  # Debugging

# Convert numeric risk to descriptive label
predicted_risk_label = risk_labels.get(int(predicted_risk[0]), "Unknown Risk Level")

# Output the result
print(f"🚗 The AI-predicted risk level of the car is: {predicted_risk_label}")
