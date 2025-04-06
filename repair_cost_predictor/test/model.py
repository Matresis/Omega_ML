import pandas as pd
import pickle as pc
from datetime import datetime


def generate_repair_reason(car):
    reasons = []

    mileage = car.get("Mileage", 0)
    age = datetime.now().year - car.get("Year", datetime.now().year)
    condition = car.get("Condition", "").lower()
    title = car.get("Title Status", "").lower()
    price = car.get("Price", 0)

    if mileage > 120_000:
        reasons.append("High mileage – possible engine wear, transmission issues, or brake replacement.")

    if age > 10:
        reasons.append("Older vehicle – may require rust removal, battery replacement, or timing belt service.")

    if "fair" in condition or "salvage" in condition:
        reasons.append("Low condition – interior/exterior damage, possible frame or suspension issues.")

    if "salvage" in title:
        reasons.append("Salvage title – vehicle may have been in an accident and require structural repairs.")

    if price < 5000 and mileage > 100_000:
        reasons.append("Low price with high mileage – likely maintenance overdue or hidden issues.")

    if not reasons:
        reasons.append("Minor maintenance (e.g., oil change, brake pads, fluid replacement).")

    return reasons

# Load the trained regression model for repair cost
model = pc.load(open("models/repair_cost_model.pkl", 'rb'))  # XGBoost model for repair cost

# Load the encoding mappings for brand
with open("models/brand_encoding.pkl", "rb") as f:
    brand_encoding = pc.load(f)

# Load the feature order for repair cost model
with open("models/feature_order.pkl", "rb") as f:
    expected_columns = pc.load(f)

# Load the standard scaler used in training
with open("models/scaler.pkl", "rb") as f:
    scaler = pc.load(f)

# Example input data for a car
new_data = {
    "Brand": "Ford",
    "Year": 2010,
    "Mileage": 50000,
    "Transmission": "automatic",
    "Body Type": "sedan",
    "Condition": "excellent",
    "Cylinders": 6,
    "Fuel Type": "gas",
    "Title Status": "clean",
    "Price": 9765
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


# Feature Engineering
CURRENT_YEAR = 2025
df_input["Car_Age"] = CURRENT_YEAR - df_input["Year"]

df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

# Encode brand
df_input["Brand_Encoded"] = df_input["Brand"].map(brand_encoding).fillna(0)

# Drop unused columns for repair need prediction
df_input.drop(columns=["Year", "Brand"], inplace=True)

# One-Hot Encoding for categorical columns
categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
df_input = pd.get_dummies(df_input, columns=categorical_columns)

# Ensure all expected columns exist for repair cost model
for col in expected_columns:
    if col not in df_input.columns:
        df_input[col] = 0

# Reorder features to match the expected order
df_input = df_input[expected_columns]

# Standardize numeric features for repair need model (RandomForestClassifier)
numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded", "Price"]
df_input[numeric_features] = scaler.transform(df_input[numeric_features])

# Predict repair cost (regression task)
predicted_repair_cost = model.predict(df_input.values)

# 🎯 Output the result for repair cost prediction
print(f"Estimated repair cost for this car: ${predicted_repair_cost[0]:,.2f}")

# 🛠 Repair reason explanation
repair_reasons = generate_repair_reason(new_data)
print("\n💡 Possible reasons for the repairs:")

for idx, reason in enumerate(repair_reasons, 1):
    print(f"{idx}. {reason}")