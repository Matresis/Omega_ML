import pandas as pd
import pickle as pc
from datetime import datetime

# Load the trained model
xgb_model = pc.load(open("models/gradientboosting_model.pkl", 'rb'))

# Load the encoding mappings
with open("models/brand_encoding.pkl", "rb") as f:
    brand_encoding = pc.load(f)

# Example input data
new_data = {
    "Brand": "Ford",
    "Year": 2018,
    "Mileage": 67772,
    "Transmission": "automatic",
    "Body Type": "pickup",
    "Condition": "like new",
    "Cylinders": 6,
    "Fuel Type": "gas",
    "Title Status": "clean"
}

# Convert input to DataFrame
df_input = pd.DataFrame([new_data])

# Feature Engineering
current_year = datetime.now().year
df_input["Car_Age"] = current_year - df_input["Year"]
df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

# Encoding categorical features
print("Encoding brand:", df_input["Brand"].iloc[0])
df_input["Brand_Encoded"] = df_input["Brand"].map(brand_encoding).fillna(0)

# Check the encoding value
print(f"Brand Encoding: {df_input['Brand_Encoded']}")

# Drop 'Year' column as it’s no longer needed
df_input.drop(columns=["Year"], inplace=True)

# One-Hot Encoding for categorical variables
categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
df_input = pd.get_dummies(df_input, columns=categorical_columns)

# Load training feature names (ensure correct order)
with open("models/feature_order.pkl", "rb") as f:
    expected_columns = pc.load(f)

# Ensure all expected columns exist (fill missing ones with 0)
for col in expected_columns:
    if col not in df_input.columns:
        df_input[col] = 0

# 🔥 Explicitly reorder `df_input` to match the training set
df_input = df_input[expected_columns]

# 🚀 Make the prediction
predicted_price = xgb_model.predict(df_input)

# 🎯 Output the result
print(f"The predicted price of the car is: ${predicted_price[0]:,.2f}")
