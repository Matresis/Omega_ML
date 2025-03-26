import pickle
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# 🚀 Load the trained model
model_path = "models/xgboost_model.pkl"  # Ensure correct path
with open(model_path, "rb") as file:
    model = pickle.load(file)

# 🚀 Define the car details for prediction
data = {
    "Brand": ["Ford"],
    "Model": ["F-150"],
    "Year": [2018],
    "Mileage": [67772],
    "Transmission": ["automatic"],
    "Body Type": ["pickup"],
    "Condition": ["like new"],
    "Cylinders": [6],
    "Fuel Type": ["gas"]
}

df = pd.DataFrame(data)

# 🚀 Feature Engineering (Match Training Pipeline)
year = datetime.now().year
df["Car_Age"] = year - df["Year"]
df["Mileage_per_Year"] = df["Mileage"] / (df["Car_Age"] + 1)  # Avoid division by zero

# 🚀 Drop Unnecessary Columns
df.drop(columns=["Year", "Brand", "Model", "Transmission", "Body Type", "Condition", "Fuel Type"], inplace=True)

# 🚀 Load the StandardScaler (Ensure same scaling as in training)
scaler_path = "models/scaler.pkl"  # Ensure the scaler was saved during training
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

df_scaled = scaler.transform(df)

# 🚀 Make Prediction
prediction = model.predict(df_scaled)
print(f"🔮 Predicted Price: ${prediction[0]:,.2f}")
