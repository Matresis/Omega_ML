import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Load Model and Preprocessing Objects
with open("models/gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/brand_encoding.pkl", "rb") as f:
    brand_encoding = pickle.load(f)

with open("models/feature_order.pkl", "rb") as f:
    expected_columns = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route("/predict-price", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert input to DataFrame
        df_input = pd.DataFrame([data])

        # Feature Engineering
        df_input["Car_Age"] = datetime.now().year - df_input["Year"]
        df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

        # Encode brand
        df_input["Brand_Encoded"] = df_input["Brand"].map(brand_encoding).fillna(0)

        # Drop unused columns
        df_input.drop(columns=["Year", "Brand"], inplace=True)

        # One-Hot Encoding
        categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
        df_input = pd.get_dummies(df_input, columns=categorical_columns)

        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        # Reorder features
        df_input = df_input[expected_columns]

        # Standardize numeric features
        numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
        df_input[numeric_features] = scaler.transform(df_input[numeric_features])

        # Predict
        predicted_price = model.predict(df_input.values)[0]

        return jsonify({"prediction": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
