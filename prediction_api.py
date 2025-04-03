import pickle as pc
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime

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
        data = request.get_json()

        # Convert input to DataFrame
        df_input = pd.DataFrame([data])

        # Normalize text data
        df_input["Brand"] = df_input["Brand"].str.title().str.strip()
        df_input["Condition"] = df_input["Condition"].str.lower().replace("like new", "excellent")
        df_input["Fuel Type"] = df_input["Fuel Type"].str.lower().str.strip()
        df_input["Transmission"] = df_input["Transmission"].str.lower().str.strip()
        df_input["Body Type"] = df_input["Body Type"].str.lower().str.strip()
        df_input["Title Status"] = df_input["Title Status"].str.lower().str.strip()

        if "Price" not in data:
            return jsonify({"error": "Missing required field: 'Price'"}), 400

        # Convert numerical values
        df_input["Price"] = pd.to_numeric(df_input["Price"], errors="coerce").fillna(0)
        df_input["Year"] = pd.to_numeric(df_input["Year"], errors="coerce").fillna(2020)
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
        current_year = datetime.now().year
        df_input["Car_Age"] = current_year - df_input["Year"]
        df_input["Mileage_per_Year"] = df_input["Mileage"] / (df_input["Car_Age"] + 1)

        # Encode Brand using precomputed mapping
        df_input["Brand_Encoded"] = df_input["Brand"].map(risk_brand_encoding).fillna(0)

        # Drop redundant columns
        df_input.drop(columns=["Year", "Brand"], inplace=True)

        # Risk Mapping with Weights
        condition_map = {"new": 0, "excellent": 1, "good": 2, "fair": 3, "salvage": 4}
        title_risk_map = {"clean": 0, "rebuilt": 2, "salvage": 3}
        body_risk_map = {"sedan": 0, "suv": 1, "coupe": 1, "hatchback": 1, "van": 2, "pickup": 2, "truck": 3}
        fuel_risk_map = {"gas": 1, "diesel": 2, "hybrid": 1, "electric": 0}
        transmission_risk_map = {"automatic": 1, "manual": 2}

        df_input["Condition_Risk"] = df_input["Condition"].map(condition_map).fillna(2)
        df_input["Title_Risk"] = df_input["Title Status"].map(title_risk_map).fillna(1)
        df_input["Body_Risk"] = df_input["Body Type"].map(body_risk_map).fillna(1)
        df_input["Fuel_Risk"] = df_input["Fuel Type"].map(fuel_risk_map).fillna(1)
        df_input["Transmission_Risk"] = df_input["Transmission"].map(transmission_risk_map).fillna(1)

        # One-Hot Encoding for categorical variables
        categorical_columns = ["Transmission", "Body Type", "Condition", "Fuel Type", "Title Status"]
        df_input = pd.get_dummies(df_input, columns=categorical_columns)

        # Ensure 'Total_Risk' is included in expected_columns if it's not already
        if 'Total_Risk' not in risk_expected_columns:
            risk_expected_columns.append('Total_Risk')

        # Ensure all expected columns exist
        for col in risk_expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0  # Add missing columns with 0

        # Custom Risk Formula with Weights
        df_input["Total_Risk"] = (
                df_input["Condition_Risk"] * 1.5 +
                df_input["Title_Risk"] * 1.5 +
                df_input["Body_Risk"] * 1.0 +
                df_input["Fuel_Risk"] * 0.6 +
                df_input["Transmission_Risk"] * 0.4 +
                (df_input["Car_Age"] / 5) * 1.2 +
                (df_input["Mileage"] / 50000) * 1.0 +
                (1 - df_input["Price"] / 50000) * 1.2
        )

        # Ensure numeric risk values remain float
        df_input["Total_Risk"] = df_input["Total_Risk"].astype(float)

        # Ensure column order matches training
        df_input = df_input[risk_expected_columns]

        # Standardize numeric features
        numeric_features = ["Car_Age", "Mileage", "Cylinders", "Brand_Encoded"]
        df_input[numeric_features] = risk_scaler.transform(df_input[numeric_features])

        # Convert to NumPy array
        df_input_np = df_input.values  # Store separately to avoid overwriting DataFrame

        # Make the prediction
        predicted_risk = risk_model.predict(df_input_np)

        # Custom rule: Override model if risk is very high
        total_risk_value = df_input.get("Total_Risk")
        if total_risk_value is not None and total_risk_value.values[0] > 20:
            predicted_risk_label = "Very High"
        elif total_risk_value is not None and total_risk_value.values[0] > 15:
            predicted_risk_label = "High"
        else:
            predicted_risk_label = risk_labels[int(predicted_risk[0])]  # Use model's prediction

        return jsonify({"prediction": predicted_risk_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
