import pickle
import pandas as pd

# Load the trained model
with open("models/flip_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input for a new car
new_car = pd.DataFrame([{
    "Price": 12000,
    "Estimated_Repairs": 2000,
    "Mileage": 75000,
    "Car_Age": 5,  # Already precomputed
    "Cylinders": 6,
    "Brand_Encoded": 30000  # Example encoded brand price
}])

# Predict resale price
resale_price = model.predict(new_car)[0]
print(f"💰 Estimated Resale Price: ${resale_price:,.2f}")
print(f"🔄 Potential Profit: ${resale_price - new_car['Price'][0]:,.2f}")
