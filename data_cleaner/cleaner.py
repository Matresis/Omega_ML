import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("craigslist_cars_to_clean.csv")

df.drop(columns=["VIN"], errors="ignore", inplace=True)

# 🚀 Convert 'Price' and 'Mileage' to numeric, forcing errors to NaN
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")

# Convert 'Year' to integer (forcing errors to NaN)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

# 🚀 Handle missing values
numeric_cols = ["Price", "Mileage", "Year"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Standardize text data
df["Brand"] = df["Brand"].str.title().str.strip()
df["Model"] = df["Model"].str.title().str.strip()
df["Condition"] = df["Condition"].str.lower().replace("like new", "excellent")
df["Fuel Type"] = df["Fuel Type"].str.lower()
df["Transmission"] = df["Transmission"].str.lower()
df["Body Type"] = df["Body Type"].str.lower()
df["Title Status"] = df["Title Status"].str.lower()

# 🚀 Convert 'Cylinders' to numeric if possible
df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce").astype("Int64")

# 🚀 Remove rows where 'Cylinders' is missing (NaN)
df = df.dropna(subset=["Cylinders"])

# 🚀 Remove rows with unknown transmission types
df = df[df["Transmission"].isin(["manual", "automatic"])]

# 🚀 Remove rows where ANY column contains "unknown"
df = df[~df.apply(lambda row: row.astype(str).str.contains("unknown", case=False, na=False).any(), axis=1)]

# 🚀 Remove outliers using the interquartile range (IQR)
def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, "Price")
df = remove_outliers(df, "Mileage")

# 🚀 Encode Brand & Model based on average price
brand_avg_price = df.groupby("Brand")["Price"].transform("mean")
model_avg_price = df.groupby("Model")["Price"].transform("mean")

df["Brand_Encoded"] = brand_avg_price
df["Model_Encoded"] = model_avg_price

# 🚀 One-hot encode categorical features
categorical_cols = ["Fuel Type", "Transmission", "Body Type", "Condition", "Title Status"]
df = pd.get_dummies(df, columns=categorical_cols)

# 🚀 Drop irrelevant columns
df.drop(columns=["VIN", "Link", "Brand", "Model"], errors="ignore", inplace=True)

# 🚀 Feature Scaling (Normalize large differences)
scaler = StandardScaler()
scaled_cols = ["Price", "Year", "Mileage", "Cylinders", "Brand_Encoded", "Model_Encoded"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Save cleaned data
df.to_csv("cleaned_craigslist_cars.csv", index=False)

print("✅ Data cleaning complete! Saved as cleaned_craigslist_cars.csv.")
