import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("cleaned_flip_data.csv")

# Features & labels
X = df[["Price", "Estimated_Repairs"]]
y = df["Estimated_Resale_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"✅ Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):,.2f}")

# Save model
with open("flip_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("📦 Flip model saved.")
