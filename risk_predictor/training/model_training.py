import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load cleaned data
df = pd.read_csv("cleaned_risk_data.csv")

# Features & labels
X = df[["Mileage", "Condition_Risk", "Title_Risk"]]
y = df["Risk_Category"].astype("category").cat.codes  # Encode labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
with open("risk_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("📦 Risk model saved.")
