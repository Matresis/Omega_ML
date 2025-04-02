import pandas as pd
import numpy as np
import pickle as pc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

# Load cleaned data
df = pd.read_csv("data/cleaned_risk_data.csv")

# Ensure 'Total_Risk' exists
if "Total_Risk" not in df.columns:
    raise ValueError("❌ Error: 'Total_Risk' column is missing from the dataset.")

# Recreate Risk Categories using the Total_Risk score
df["Risk_Category"] = pd.cut(
    df["Total_Risk"],
    bins=[0, 5, 10, 15, 20],
    labels=["Low", "Medium", "High", "Very High"]
)

# Convert categorical risk labels to numerical values
df["Risk_Category"] = df["Risk_Category"].map({"Low": 0, "Medium": 1, "High": 2, "Very High": 3})

# Replace inf values with NaN and drop rows with NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Define features (X) and target (y)
# We drop 'Risk_Category' and 'Total_Risk' from features.
X = df.drop(columns=["Risk_Category", "Total_Risk"])
y = df["Risk_Category"]

# Replace NaN values with median
X = X.fillna(X.median())

# Replace infinite values with a large number
X.replace([np.inf, -np.inf], 1e10, inplace=True)

# --- Feature Scaling ---
# Even though tree models don't require scaling, we'll scale the data to keep things consistent with the testing script.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use during prediction
os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pc.dump(scaler, f)

# Save the feature order (list of column names) to use during testing
feature_order = X.columns.tolist()
with open("models/feature_order.pkl", "wb") as f:
    pc.dump(feature_order, f)

print("✅ Feature order saved:", feature_order)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train a RandomForestClassifier with basic hyperparameters (adjust as needed)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model
with open("models/risk_model.pkl", "wb") as f:
    pc.dump(model, f)

print("✅ Risk prediction model saved as models/risk_model.pkl")
