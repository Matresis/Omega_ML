import pandas as pd
import pickle as pc
from datetime import datetime

# Load the scaler and the encoding dictionaries
scaler = pc.load(open("models/scaler.pkl", 'rb'))
with open("models/brand_encoding.pkl", "rb") as f:
    brand_encoding = pc.load(f)

with open("models/model_encoding.pkl", "rb") as f:
    model_encoding = pc.load(f)

# Example input data (replace with actual new data)
new_data = {
    "Brand": "Ford",
    "Model": "f-150",
    "Year": 2018,
    "Mileage": 67772,
    "Transmission": "automatic",
    "Body Type": "pickup",
    "Condition": "like new",
    "Cylinders": 6,
    "Fuel Type": "gas",
    "Title Status": "clean"
}

# Convert input data to DataFrame
df_input = pd.DataFrame([new_data])

# Feature Engineering
year = datetime.now().year
df_input["Car_Age"] = year - df_input["Year"]
df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

# Encoding for 'Brand' and 'Model' columns
df_input["Brand_Encoded"] = df_input["Brand"].map(brand_encoding).fillna(0)
df_input["Model_Encoded"] = df_input["Model"].map(model_encoding).fillna(0)

# Drop 'Year' as we now have 'Car_Age'
df_input.drop(columns=["Year"], inplace=True)

# One-Hot Encoding for categorical features (including missing categories)
categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
df_input = pd.get_dummies(df_input, columns=categorical_columns, drop_first=True)

print("Columns after encoding:", df_input.columns.tolist())

# List of expected columns after encoding
expected_columns = ['Mileage', 'Cylinders', 'Brand_Encoded', 'Model_Encoded', 'Fuel Type_diesel', 'Fuel Type_electric',
                    'Fuel Type_gas', 'Fuel Type_hybrid', 'Fuel Type_other', 'Transmission_automatic', 'Transmission_manual',
                    'Body Type_bus', 'Body Type_convertible', 'Body Type_coupe', 'Body Type_hatchback', 'Body Type_minivan',
                    'Body Type_offroad', 'Body Type_other', 'Body Type_pickup', 'Body Type_sedan', 'Body Type_suv', 'Body Type_truck',
                    'Body Type_van', 'Body Type_wagon', 'Condition_excellent', 'Condition_fair', 'Condition_good', 'Condition_new',
                    'Condition_salvage', 'Title Status_clean', 'Title Status_lien', 'Title Status_missing', 'Title Status_parts only',
                    'Title Status_rebuilt', 'Title Status_salvage', 'Car_Age', 'Mileage_per_Year']

# Add missing columns with 0 if they're not present in the new data
for col in expected_columns:
    if col not in df_input.columns:
        df_input[col] = 0

# Reorder columns to match the expected order (important for scaling)
df_input = df_input[expected_columns]

# Feature Scaling (Standardization)
scaled_features = ["Mileage", "Cylinders", "Brand_Encoded", "Model_Encoded", "Car_Age", "Mileage_per_Year"] + expected_columns[6:]

# Ensure the scaler is applied to the correct columns
df_input[scaled_features] = scaler.transform(df_input[scaled_features])

# Prepare the features for prediction (drop original categorical columns)
X_new = df_input.drop(columns=["Brand", "Model", "Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"])

# Load the trained XGBoost model
xgb_model = pc.load(open("models/xgboost_model.pkl", 'rb'))

# Make a prediction
predicted_price = xgb_model.predict(X_new)

# Print the predicted price
print(f"The predicted price of the car is: ${predicted_price[0]:,.2f}")
