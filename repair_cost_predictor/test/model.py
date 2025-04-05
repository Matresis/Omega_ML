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
    "Mileage": 150000,
    "Transmission": "automatic",
    "Body Type": "sedan",
    "Condition": "excellent",
    "Cylinders": 6,
    "Fuel Type": "gas",
    "Title Status": "clean",
    "Price": 16000
}

# Convert input to DataFrame
df_input = pd.DataFrame([new_data])

# Feature Engineering
df_input["Car_Age"] = datetime.now().year - df_input["Year"]
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
numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
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