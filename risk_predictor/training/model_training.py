import pandas as pd
import numpy as np
import pickle as pc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/cleaned_risk_data.csv")

# Map risk labels to numbers
risk_label_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
df["Risk_Category"] = df["Risk_Category"].map(risk_label_map)

# Define features and target
X = df.drop(columns=["Risk_Category"])  # No Total_Risk in features!
y = df["Risk_Category"]

# Handle missing values in features
X.fillna(X.median(), inplace=True)
X.replace([np.inf, -np.inf], 1e10, inplace=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save the feature order after preprocessing
with open("models/feature_order.pkl", "wb") as f:
    pc.dump(X.columns.tolist(), f)

# Train the model (use a more powerful model if needed)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight='balanced',  # Add class weighting to handle imbalanced classes
    random_state=42
)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}\n")

# Detailed debugging for classification report and confusion matrix
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Additional debugging: Check predictions and actual values
print("Predictions vs Actual Values:")
print(pd.DataFrame({"Predicted": y_pred, "Actual": y_test}).head())

# Save the model
with open("models/risk_model.pkl", "wb") as f:
    pc.dump(model, f)

with open("models/risk_label_map.pkl", "wb") as f:
    pc.dump(risk_label_map, f)

print("✅ Risk prediction model saved.")

importances = model.feature_importances_
feature_names = X.columns

# Visualize top features
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Top 10 Important Features:\n", feat_imp.head(10))

feat_imp.head(15).plot(kind="barh", title="Top 15 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
