import pickle
import pandas as pd

# Load trained model
with open("flip_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input
new_car = pd.DataFrame([{
    "Price": 12000,
    "Estimated_Repairs": 2000
}])

# Predict resale price
resale_price = model.predict(new_car)[0]
print(f"💰 Estimated Resale Price: ${resale_price:,.2f}")
print(f"🔄 Potential Profit: ${resale_price - new_car['Price'][0]:,.2f}")