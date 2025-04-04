import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pc
from datetime import datetime

# Load dataset
df = pd.read_csv("data/craigslist_cars_to_clean.csv")

# Drop unnecessary columns
df.drop(columns=["VIN", "Model", "Link"], errors="ignore", inplace=True)

# Convert columns to appropriate data types
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")

# Handle missing or "Unknown" values
df["Mileage"].fillna(df["Mileage"].median(), inplace=True)
df["Price"].fillna(df["Price"].median(), inplace=True)
df["Year"].fillna(df["Year"].median(), inplace=True)
df["Cylinders"].fillna(df["Cylinders"].median(), inplace=True)

# Replace "Unknown" text with NaN to mark missing data
df.replace("Unknown", np.nan, inplace=True)

# Create a binary feature indicating if data is missing/unknown
df["Missing_Mileage"] = df["Mileage"].isna().astype(int)
df["Missing_Year"] = df["Year"].isna().astype(int)
df["Missing_Cylinders"] = df["Cylinders"].isna().astype(int)

# Normalize text data
df["Brand"] = df["Brand"].str.title().str.strip()
df["Condition"] = df["Condition"].str.lower().replace("like new", "excellent")
df["Fuel Type"] = df["Fuel Type"].str.lower()
df["Transmission"] = df["Transmission"].str.lower()
df["Body Type"] = df["Body Type"].str.lower()
df["Title Status"] = df["Title Status"].str.lower()

# Normalize text columns
for col in ["Condition", "Title Status", "Body Type", "Transmission", "Fuel Type"]:
    df[col] = df[col].str.strip().str.lower()

# Drop rows with missing 'Cylinders' or "unknown" transmission rows
df = df.dropna(subset=["Cylinders"])
df = df[df["Transmission"].isin(["manual", "automatic"])]
df = df[~df.apply(lambda row: row.astype(str).str.contains("unknown", case=False, na=False).any(), axis=1)]

# Calculate car age and add features like Mileage per Year
CURRENT_YEAR = 2025
df["Car_Age"] = CURRENT_YEAR - df["Year"]
df["Mileage_per_Year"] = df["Mileage"] / (df["Car_Age"] + 1)

# Remove 'Year' column (now redundant)
df.drop(columns=["Year"], inplace=True)

# Handle "Unknown" values more precisely: Encode with average price for brands
brand_avg_price = df.groupby("Brand")["Price"].transform("mean")
df["Brand_Encoded"] = brand_avg_price

# Handle outliers (IQR)
def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, "Price")
df = remove_outliers(df, "Mileage")

# Risk assessment mappings
condition_map = {
    "new": 0, "like new": 1, "excellent": 2,
    "good": 3, "fair": 4, "salvage": 5, "unknown": 3
}
title_risk_map = {"clean": 0, "rebuilt": 2, "salvage": 3}
body_risk_map = {"sedan": 0, "SUV": 1, "coupe": 1, "hatchback": 1, "van": 2, "pickup": 2, "truck": 3, "unknown": 2}
fuel_risk_map = {"gas": 1, "diesel": 2, "hybrid": 1, "electric": 0, "unknown": 2}
transmission_risk_map = {"automatic": 1, "manual": 2, "unknown": 2}

# Map categorical values to risk levels
df["Condition_Risk"] = df["Condition"].map(condition_map)
df["Title_Risk"] = df["Title Status"].map(title_risk_map).fillna(1)
df["Body_Risk"] = df["Body Type"].map(body_risk_map)
df["Fuel_Risk"] = df["Fuel Type"].map(fuel_risk_map)
df["Transmission_Risk"] = df["Transmission"].map(transmission_risk_map)

# Normalize price and mileage risk
df["Price_Risk"] = pd.qcut(df["Price"], q=3, labels=[2, 1, 0], duplicates="drop")  # Lower price = Higher risk
df["Mileage_Risk"] = pd.qcut(df["Mileage"], q=3, labels=[0, 1, 2], duplicates="drop")  # Higher mileage = Higher risk

# Normalize car age risk
df["Age_Risk"] = pd.qcut(df["Car_Age"], q=3, labels=[0, 1, 2], duplicates="drop")

# Compute total risk score
df["Total_Risk"] = (
    df["Condition_Risk"] + df["Title_Risk"] +
    df["Body_Risk"] + df["Fuel_Risk"] +
    df["Transmission_Risk"] + df["Price_Risk"].astype(int) +
    df["Mileage_Risk"].astype(int) + df["Age_Risk"].astype(int)
)

# Risk Category based on total risk score
df["Risk_Category"] = pd.cut(df["Total_Risk"], bins=[0, 5, 10, 15, 20], labels=["Low", "Medium", "High", "Very High"])

# One-hot encode categorical features
categorical_cols = ["Fuel Type", "Transmission", "Body Type", "Condition", "Title Status"]
df = pd.get_dummies(df, columns=categorical_cols)

# Drop irrelevant columns
df.drop(columns=["Brand"], errors="ignore", inplace=True)

# Feature Scaling (Standardize numeric fields)
scaler = StandardScaler()
scaled_cols = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Save the scaler
with open("models/scaler.pkl", "wb") as f:
    pc.dump(scaler, f)

# Save cleaned dataset, including the 'Total_Risk' column
df.to_csv("data/cleaned_risk_data_with_total_risk.csv", index=False)

# Completion message
print("✅ Risk data cleaning complete! Saved as cleaned_risk_data_with_total_risk.csv.")