import pickle as pc
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Load Model and Preprocessing Objects - Price Prediction Model
with open("models/price/price_model.pkl", "rb") as f:
    price_model = pc.load(f)

with open("models/price/brand_encoding.pkl", "rb") as f:
    price_brand_encoding = pc.load(f)

with open("models/price/feature_order.pkl", "rb") as f:
    price_expected_columns = pc.load(f)

with open("models/price/scaler.pkl", "rb") as f:
    price_scaler = pc.load(f)

# Load Model and Preprocessing Objects - Risk Prediction Model
with open("models/risk/risk_model.pkl", "rb") as f:
    risk_model = pc.load(f)

with open("models/risk/brand_encoding.pkl", "rb") as f:
    risk_brand_encoding = pc.load(f)

with open("models/risk/feature_order.pkl", "rb") as f:
    risk_expected_columns = pc.load(f)

with open("models/risk/scaler.pkl", "rb") as f:
    risk_scaler = pc.load(f)

with open("models/risk/risk_label_map.pkl", "rb") as f:
    risk_labels = pc.load(f)

app = Flask(__name__)


@app.route("/predict-price", methods=["POST"])
def predict_price():
    try:
        data = request.get_json()
        print("Received JSON data:", data)

        df_input = pd.DataFrame([data])

        df_input["Car_Age"] = datetime.now().year - df_input["Year"]
        df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

        df_input["Brand_Encoded"] = df_input["Brand"].map(price_brand_encoding).fillna(0)

        df_input.drop(columns=["Year", "Brand"], inplace=True)
        df_input = pd.get_dummies(df_input,
                                  columns=["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"])

        for col in price_expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[price_expected_columns]

        numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
        df_input[numeric_features] = price_scaler.transform(df_input[numeric_features])

        predicted_price = price_model.predict(df_input.values)

        return jsonify({"prediction": round(float(predicted_price[0]), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-risk", methods=["POST"])
def predict_risk():
    try:
        data = request.get_json(force=True)
        print(f"[DEBUG] Received JSON: {data} (type={type(data)})")

        if not isinstance(data, dict):
            return jsonify({"error": f"Expected JSON object, got {type(data).__name__}"}), 400

        # Convert input to DataFrame
        df_input = pd.DataFrame([data])

        # Normalize text data
        df_input["Brand"] = df_input["Brand"].str.title().str.strip()
        df_input["Condition"] = df_input["Condition"].str.lower().replace("like new", "excellent")
        df_input["Fuel Type"] = df_input["Fuel Type"].str.lower().str.strip()
        df_input["Transmission"] = df_input["Transmission"].str.lower().str.strip()
        df_input["Body Type"] = df_input["Body Type"].str.lower().str.strip()
        df_input["Title Status"] = df_input["Title Status"].str.lower().str.strip()

        # Convert numerical values
        df_input["Price"] = pd.to_numeric(df_input["Price"], errors="coerce").fillna(0)
        df_input["Mileage"] = pd.to_numeric(df_input["Mileage"], errors="coerce").fillna(df_input["Mileage"].median())
        df_input["Cylinders"] = pd.to_numeric(df_input["Cylinders"], errors="coerce").fillna(
            df_input["Cylinders"].median())

        # Handle missing values in categorical columns
        default_mappings = {
            "Transmission": "automatic",
            "Body Type": "sedan",
            "Condition": "good",
            "Fuel Type": "gas",
            "Title Status": "clean"
        }
        for col, default_value in default_mappings.items():
            df_input[col] = df_input[col].fillna(default_value)

        # Feature Engineering
        CURRENT_YEAR = 2025
        df_input["Car_Age"] = CURRENT_YEAR - df_input["Year"]

        # Encode Brand using precomputed mapping
        df_input["Brand_Encoded"] = df_input["Brand"].map(risk_brand_encoding).fillna(
            df_input["Price"].mean())  # Avg price used for unknown brands

        # Drop redundant columns
        df_input.drop(columns=["Year", "Brand"], inplace=True, errors="ignore")

        # One-Hot Encoding for categorical variables
        categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
        df_input = pd.get_dummies(df_input, columns=categorical_columns)

        # Ensure all expected columns exist
        for col in risk_expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0  # Add missing columns with 0

        # Standardize numeric features
        numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
        df_input[numeric_features] = risk_scaler.transform(df_input[numeric_features])

        # Ensure column order matches training
        df_input = df_input[risk_expected_columns]

        # Convert to NumPy array
        df_input_np = df_input.values  # Store separately to avoid overwriting DataFrame

        # Make the prediction
        predicted_risk = risk_model.predict(df_input_np)

        # Reverse the dictionary to use numbers as keys
        reversed_risk_labels = {v: k for k, v in risk_labels.items()}

        # Convert numeric risk to descriptive label
        predicted_risk_label = reversed_risk_labels.get(int(predicted_risk[0]), "Unknown Risk Level")

        return jsonify({"prediction": predicted_risk_label})

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)