import pandas as pd
import pickle as pc
from datetime import datetime

# Load the trained model
model = pc.load(open("models/randomforest_model.pkl", 'rb'))  # Use the desired model

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
    "Year": 2020,
    "Mileage": 50000,
    "Transmission": "automatic",
    "Body Type": "sedan",
    "Condition": "excellent",
    "Cylinders": 6,
    "Fuel Type": "gas",
    "Title Status": "clean",
    "Price": 17596
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

# Add missing expected columns
for col in expected_columns:
    if col not in df_input.columns:
        df_input[col] = 0

# Remove extra columns not seen during training
df_input = df_input[expected_columns]

# Standardize numeric features
numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
df_input[numeric_features] = scaler.transform(df_input[numeric_features])

# Predict
predicted_repair_needed = model.predict(df_input.values)

# 🎯 Output the result
if predicted_repair_needed[0] == 1:
    print("This car likely needs repairs.")
else:
    print("This car does not likely need repairs.")
