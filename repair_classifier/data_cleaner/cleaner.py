import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle as pc
import os
from datetime import datetime

# Load the dataset
df = pd.read_csv("data/filtered_cars_by_brand.csv")

# Drop unnecessary columns (VIN, Model)
df.drop(columns=["VIN", "Model"], errors="ignore", inplace=True)

# Convert 'Price' and 'Mileage' to numeric (force errors to NaN)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")

# Convert 'Year' to integer
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

# Handle missing values
numeric_cols = ["Price", "Mileage", "Year"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Standardize text data
df["Brand"] = df["Brand"].str.title().str.strip()
df["Condition"] = df["Condition"].str.lower().replace("like new", "excellent")
df["Fuel Type"] = df["Fuel Type"].str.lower()
df["Transmission"] = df["Transmission"].str.lower()
df["Body Type"] = df["Body Type"].str.lower()
df["Title Status"] = df["Title Status"].str.lower()

# Convert 'Cylinders' to numeric if possible
df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce").astype("Int64")

# Remove rows with missing 'Cylinders' and remove 'unknown' transmission rows
df = df.dropna(subset=["Cylinders"])
df = df[df["Transmission"].isin(["manual", "automatic"])]
df = df[~df.apply(lambda row: row.astype(str).str.contains("unknown", case=False, na=False).any(), axis=1)]

# Remove outliers using IQR
def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, "Price")
df = remove_outliers(df, "Mileage")

# Feature Engineering: Add 'Car_Age' and 'Mileage_per_Year'
current_year = datetime.now().year
df["Car_Age"] = current_year - df["Year"]
df["Mileage_per_Year"] = df["Mileage"] / (df["Car_Age"] + 1)

# Drop 'Year' column as we now have 'Car_Age'
df.drop(columns=["Year"], inplace=True)

# Create the target variable 'Repair Needed'
# Assuming cars with high mileage, old age, or poor condition might need repairs
df['Repair Needed'] = ((df['Mileage'] > 100000) | (df['Car_Age'] > 10) | (df['Condition'] == 'fair')).astype(int)

# Encode 'Brand' by average price
brand_avg_price = df.groupby("Brand")["Price"].transform("mean")
df["Brand_Encoded"] = brand_avg_price

# Save the encoding mappings for 'Brand' for use in training
brand_price_mapping = df.groupby("Brand")["Price"].mean().to_dict()

with open("models/brand_encoding.pkl", "wb") as f:
    pc.dump(brand_price_mapping, f)

# One-hot encode categorical features
categorical_cols = ["Fuel Type", "Transmission", "Body Type", "Condition", "Title Status"]
df = pd.get_dummies(df, columns=categorical_cols)

# Drop irrelevant columns
df.drop(columns=["VIN", "Link", "Brand"], errors="ignore", inplace=True)

# Feature Scaling (Standardize the numeric columns)
scaler = StandardScaler()
scaled_cols = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Save the scaler
with open("models/scaler.pkl", "wb") as f:
    pc.dump(scaler, f)

# Save the cleaned dataset
df.to_csv("data/cleaned_craigslist_cars_repair.csv", index=False)

print("✅ Data cleaning complete! Saved as cleaned_craigslist_cars_repair.csv.")
