import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("craigslist_cars.csv")

# Fill missing values
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce").fillna(df["Mileage"].median())
df["Condition"] = df["Condition"].fillna("unknown")
df["Title Status"] = df["Title Status"].fillna("clean")

# Map conditions to risk levels
condition_map = {
    "new": 0, "like new": 1, "excellent": 2,
    "good": 3, "fair": 4, "salvage": 5, "unknown": 3
}
df["Condition_Risk"] = df["Condition"].map(condition_map)

# Assign risk level based on title status
title_risk_map = {"clean": 0, "rebuilt": 2, "salvage": 3}
df["Title_Risk"] = df["Title Status"].map(title_risk_map).fillna(1)

# Define risk categories
df["Risk_Level"] = df["Condition_Risk"] + df["Title_Risk"]
df["Risk_Category"] = pd.cut(df["Risk_Level"], bins=[0, 2, 4, 6], labels=["Low", "Medium", "High"])

# Save cleaned data
df.to_csv("cleaned_risk_data.csv", index=False)
print("✅ Risk data cleaned and saved.")
