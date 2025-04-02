import pandas as pd
import pickle as pc
from datetime import datetime

# Load the trained risk prediction model
model = pc.load(open("models/risk_model.pkl", "rb"))

# Load the encoding mapping for Brand (if used)
with open("models/brand_encoding.pkl", "rb") as f:
    brand_encoding = pc.load(f)

# Load the expected feature order saved during training
with open("models/feature_order.pkl", "rb") as f:
    expected_columns = pc.load(f)  # Expected to be a list of 45 feature names

print("Expected feature order ({} features):".format(len(expected_columns)))
print(expected_columns)

# Example input data provided by the user, including the Price
new_data = {
    "Brand": "Ford",
    "Year": 2018,
    "Mileage": 50000,
    "Price": 16000,  # User-entered price
    "Transmission": "automatic",
    "Body Type": "pickup",
    "Condition": "like new",
    "Cylinders": 6,
    "Fuel Type": "gas",
    "Title Status": "clean"
}

# Convert input data to DataFrame
df_input = pd.DataFrame([new_data])

# Feature Engineering: calculate Car_Age and Mileage_per_Year
current_year = datetime.now().year
df_input["Car_Age"] = current_year - df_input["Year"]
df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

# Encode 'Brand' using the saved encoding mapping (convert to lowercase for consistency)
df_input["Brand"] = df_input["Brand"].str.lower()
df_input["Brand_Encoded"] = df_input["Brand"].map(brand_encoding).fillna(0)

# Drop unnecessary columns (drop 'Year' and 'Brand'; keep 'Price')
df_input.drop(columns=["Year", "Brand"], inplace=True)

# One-Hot Encoding for remaining categorical variables
categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
df_input = pd.get_dummies(df_input, columns=categorical_columns)

# Debug: Print columns before reindexing
print("Columns before reindexing:", df_input.columns.tolist())

# Ensure that all expected columns exist; add missing ones with 0.
df_input = df_input.reindex(columns=expected_columns, fill_value=0)

# Debug: Print final columns and shape
print("Columns after reindexing:", df_input.columns.tolist())
print("Final shape of input features:", df_input.shape)

# Convert to NumPy array if required by the model
X_new = df_input.values

# Make the risk prediction
predicted_risk = model.predict(X_new)[0]

# Map numerical risk to a meaningful label (assuming: 0=Low, 1=Medium, 2=High, 3=Very High)
risk_categories = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk", 3: "Very High Risk"}
risk_label = risk_categories.get(predicted_risk, "Unknown")

# Output the result
print(f"\n🚗 Predicted Risk Category: {risk_label}")
