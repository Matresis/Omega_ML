import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
file_path = os.path.join(os.path.dirname(__file__), "craigslist_cars_to_clean.csv")
df = pd.read_csv(file_path)


# Drop unnecessary columns
df.drop(columns=["VIN"], errors="ignore", inplace=True)

# Convert 'Price' and 'Mileage' to numeric
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

# Handle missing values
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

# Convert 'Cylinders' to numeric
df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce").astype("Int64")

# Remove missing values in key columns
df.dropna(subset=["Cylinders"], inplace=True)
df = df[df["Transmission"].isin(["manual", "automatic"])]

# Remove rows where ANY column contains "unknown"
df = df[~df.apply(lambda row: row.astype(str).str.contains("unknown", case=False, na=False).any(), axis=1)]

# Remove outliers using IQR
def remove_outliers(df, column):
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, "Price")
df = remove_outliers(df, "Mileage")

# Encode Brand & Model based on average price
df["Brand_Encoded"] = df.groupby("Brand")["Price"].transform("mean")
df["Model_Encoded"] = df.groupby("Model")["Price"].transform("mean")

# One-hot encode categorical features (excluding the target)
categorical_features = ["Title Status"]
df = pd.get_dummies(df, columns=categorical_features)

# Label encode categorical targets
label_encoders = {}
target_columns = ["Fuel Type", "Transmission", "Body Type", "Condition"]
for col in target_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop irrelevant columns
df.drop(columns=["VIN", "Link", "Brand", "Model"], errors="ignore", inplace=True)

# Feature Scaling
scaler = StandardScaler()
scaled_cols = ["Price", "Year", "Mileage", "Cylinders", "Brand_Encoded", "Model_Encoded"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Save cleaned data
df.to_csv("cleaned_craigslist_cars_classification.csv", index=False)

print("✅ Data cleaning complete! Saved as cleaned_craigslist_cars_classification.csv.")
