import pandas as pd
import requests

API_KEY = "S711EBOUek2pf145pTwPug==MbebzFBDWwPqNkZK"
API_URL = "https://api.api-ninjas.com/v1/cars"

def fetch_car_details(brand, model):
    """Fetch additional car details from the API."""
    try:
        response = requests.get(
            API_URL,
            headers={"X-Api-Key": API_KEY},
            params={"make": brand, "model": model}
        )
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]  # Return the first matching car
    except Exception as e:
        print(f"⚠️ API request failed: {e}")
    return {}

def clean_data(file_path="data/raw_craigslist_cars.csv", save_path="data/cleaned_craigslist_cars.csv"):
    df = pd.read_csv(file_path)

    print("🛠️ Cleaning and enriching car data...")

    for index, row in df.iterrows():
        brand, model = row["Brand"], row["Model"]
        if brand == "Unknown" or model == "Unknown":
            continue

        print(f"🔍 Fetching details for: {brand} {model}...")
        car_api_data = fetch_car_details(brand, model)

        df.at[index, "Engine Size"] = car_api_data.get("engine", "Unknown")
        df.at[index, "Drivetrain"] = car_api_data.get("drivetrain", "Unknown")
        df.at[index, "Horsepower"] = car_api_data.get("horsepower", "Unknown")
        df.at[index, "Torque"] = car_api_data.get("torque", "Unknown")
        df.at[index, "Fuel Efficiency"] = car_api_data.get("combined_mpg", "Unknown")

    df.to_csv(save_path, index=False)
    print(f"✅ Data cleaning complete! Saved as {save_path}")

# Run cleaner
clean_data()
