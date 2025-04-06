import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle as pc
from datetime import datetime
import os

# Load the dataset
df = pd.read_csv("data/filtered_cars_by_brand.csv")

# Drop unnecessary columns
df.drop(columns=["VIN", "Model", "Link"], errors="ignore", inplace=True)

# Convert data types
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce").astype("Int64")

# Handle missing values
df[["Price", "Mileage", "Year"]] = df[["Price", "Mileage", "Year"]].fillna(df[["Price", "Mileage", "Year"]].median())
df.dropna(subset=["Cylinders"], inplace=True)

# Text standardization
df["Brand"] = df["Brand"].str.title().str.strip()
df["Condition"] = df["Condition"].str.lower().replace("like new", "excellent")
df["Fuel Type"] = df["Fuel Type"].str.lower()
df["Transmission"] = df["Transmission"].str.lower()
df["Body Type"] = df["Body Type"].str.lower()
df["Title Status"] = df["Title Status"].str.lower()

# Filter rows
df = df[df["Transmission"].isin(["manual", "automatic"])]
df = df[~df.apply(lambda row: row.astype(str).str.contains("unknown", case=False, na=False).any(), axis=1)]

# Remove outliers
def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

df = remove_outliers(df, "Price")
df = remove_outliers(df, "Mileage")

# Feature engineering
current_year = datetime.now().year
df["Car_Age"] = current_year - df["Year"]
df["Mileage_per_Year"] = df["Mileage"] / (df["Car_Age"] + 1)
df.drop(columns=["Year"], inplace=True)

# Estimate Repair Cost Heuristic (can be replaced by a more advanced formula later)
def estimate_repair_cost(row):
    if row["Condition"] in ["fair", "salvage"]:
        return row["Mileage"] * 0.05  # Example heuristic
    if row["Mileage"] > 150_000:
        return row["Mileage"] * 0.03
    return row["Mileage"] * 0.005

df["Repair Cost"] = df.apply(estimate_repair_cost, axis=1)
print("Estimate Repair Cost" + str(df["Repair Cost"]))

# Brand encoding (price-based)
df["Brand_Encoded"] = df.groupby("Brand")["Price"].transform("mean")
brand_price_mapping = df.groupby("Brand")["Price"].mean().to_dict()

with open("models/brand_encoding.pkl", "wb") as f:
    pc.dump(brand_price_mapping, f)

# One-hot encoding
categorical_cols = ["Fuel Type", "Transmission", "Body Type", "Condition", "Title Status"]
df = pd.get_dummies(df, columns=categorical_cols)

# Drop now irrelevant brand column
df.drop(columns=["Brand"], inplace=True)

# Feature scaling
scaler = StandardScaler()
scale_cols = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded", "Price"]
df[scale_cols] = scaler.fit_transform(df[scale_cols])

with open("models/scaler.pkl", "wb") as f:
    pc.dump(scaler, f)

# Save cleaned data
df.to_csv("data/cleaned_craigslist_cars_repair.csv", index=False)
print("✅ Data cleaning complete! Saved as cleaned_craigslist_cars_repair.csv.")
