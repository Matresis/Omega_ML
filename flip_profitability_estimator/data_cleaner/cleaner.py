import pandas as pd

# Load dataset
df = pd.read_csv("craigslist_cars.csv")

# Ensure numerical values are correct
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df = df[(df["Price"] > 500) & (df["Price"] < 100000)]  # Remove extreme outliers

# Estimate repair costs based on condition
repair_cost_map = {"new": 0, "like new": 500, "excellent": 1000, "good": 2000, "fair": 3000, "salvage": 5000}
df["Estimated_Repairs"] = df["Condition"].map(repair_cost_map).fillna(2500)

# Calculate estimated resale price
df["Estimated_Resale_Price"] = df["Price"] + (df["Estimated_Repairs"] * 1.5)

# Save cleaned data
df.to_csv("cleaned_flip_data.csv", index=False)
print("✅ Flip data cleaned and saved.")
