import pandas as pd
import numpy as np
import pickle as pc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv("data/cleaned_risk_data_with_total_risk.csv")

# Ensure 'Total_Risk' exists
if "Total_Risk" not in df.columns:
    raise ValueError("❌ Error: 'Total_Risk' column is missing from the dataset.")

# Improve Risk Categorization
risk_labels = ["Very Low", "Low", "Medium", "High", "Very High", "Extreme"]
df["Risk_Category"] = pd.cut(
    df["Total_Risk"],
    bins=[0, 4, 8, 12, 16, 20, 25],
    labels=risk_labels
)

# Map risk labels to numerical values for training
risk_label_map = {label: i for i, label in enumerate(risk_labels)}
df["Risk_Category"] = df["Risk_Category"].map(risk_label_map)

# Handle missing/invalid data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Feature engineering for risk based on Price, Mileage, and Car Age
df["Price_Risk"] = pd.qcut(df["Price"], q=5, labels=[4, 3, 2, 1, 0]).astype(int)
df["Mileage_Risk"] = pd.qcut(df["Mileage"], q=5, labels=[4, 3, 2, 1, 0]).astype(int)
df["Age_Risk"] = pd.qcut(df["Car_Age"], q=5, labels=[4, 3, 2, 1, 0]).astype(int)

# Feature engineering and including Total_Risk
df["Total_Risk"] = (
    df["Condition_Risk"] + df["Title_Risk"] +
    df["Body_Risk"] + df["Fuel_Risk"] +
    df["Transmission_Risk"] + df["Price_Risk"].astype(int) +
    df["Mileage_Risk"].astype(int) + df["Age_Risk"].astype(int)
)

# Define features and target (including Total_Risk)
X = df.drop(columns=["Risk_Category"])  # Total_Risk is part of features but not the target
y = df["Risk_Category"]

# Handle missing values in features
X.fillna(X.median(), inplace=True)
X.replace([np.inf, -np.inf], 1e10, inplace=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save feature order (including Total_Risk)
feature_order = X.columns.tolist()
with open("models/feature_order.pkl", "wb") as f:
    pc.dump(feature_order, f)

print("✅ Feature order saved.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
y_pred_labels = [risk_labels[int(pred)] for pred in y_pred]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and mappings
with open("models/risk_model.pkl", "wb") as f:
    pc.dump(model, f)

with open("models/risk_label_map.pkl", "wb") as f:
    pc.dump(risk_labels, f)

print("✅ Risk prediction model saved.")
