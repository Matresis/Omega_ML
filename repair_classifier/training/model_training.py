import pandas as pd
import pickle as pc
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/cleaned_craigslist_cars_repair.csv")

# Define features (X) and target variable (y)
X = df.drop(columns=["Repair Needed"])  # Features
y = df["Repair Needed"]  # Target variable

# Replace NaN values with median
X = X.fillna(X.median())

# Replace infinite values with a large number
X.replace([np.inf, -np.inf], 1e10, inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Save the feature order after preprocessing
with open("models/feature_order.pkl", "wb") as f:
    pc.dump(X.columns.tolist(), f)

# 🎯 Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# 🎯 Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# 🎯 Gradient Boosting Classifier
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)

# 🎯 XGBoost Classifier
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)

# 🚀 Evaluate Models
models = {
    "RandomForest": rf,
    "LogisticRegression": lr,
    "GradientBoosting": gb,
    "XGBoost": xgb
}

print("\n📊 Model Performance:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🔹 {name}:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"   Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

# 🚀 Save Models
for name, model in models.items():
    filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
    pc.dump(model, open(filename, 'wb'))
    print(f"✅ Model saved: {filename}")

print("\n🎯 All models trained and exported successfully!")
