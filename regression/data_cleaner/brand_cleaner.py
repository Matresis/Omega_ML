import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/craigslist_cars_to_clean.csv")

# Merge similar brands
brand_mapping = {"Chevy": "Chevrolet", "Mercedes": "Mercedes-Benz"}
df["Brand"] = df["Brand"].replace(brand_mapping)

# Keep only the most common brands
top_brands = [
    "Ford", "Toyota", "Chevrolet", "Honda", "Nissan", "BMW", "Dodge", "Jeep",
    "GMC", "Subaru", "Lexus", "Mercedes-Benz", "Hyundai", "Volkswagen",
    "Kia", "Cadillac", "Audi", "Chrysler"
]
df = df[df["Brand"].isin(top_brands)]


# Filter dataset to only include desirable brands
df_filtered = df[df["Brand"].isin(top_brands)]

# Save the extracted dataset
df_filtered.to_csv("data/filtered_cars_by_brand.csv", index=False)

print(f"✅ Extracted {len(df_filtered)} cars from desirable brands and saved to 'filtered_cars_by_brand.csv'.")
