import pickle
import pandas as pd

# Load trained model
with open("risk_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input
new_car = pd.DataFrame([{
    "Mileage": 85000,
    "Condition_Risk": 3,
    "Title_Risk": 2
}])

# Predict risk category
risk_pred = model.predict(new_car)[0]
risk_labels = {0: "Low", 1: "Medium", 2: "High"}

print(f"🚗 Predicted Risk: {risk_labels[risk_pred]}")
