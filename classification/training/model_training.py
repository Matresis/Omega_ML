from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from classification.data_cleaner.cleaner import label_encoders

# 🚀 Load cleaned dataset
df = pd.read_csv("cleaned_craigslist_cars_classification.csv")

# Select the target variable for classification (Fuel Type, Transmission, Body Type, Condition)
target_column = "Fuel Type"  # Change this to classify a different variable

# 🚀 Feature Engineering
df["Car_Age"] = 2025 - df["Year"]
df["Mileage_per_Year"] = df["Mileage"] / (df["Car_Age"] + 1)  # Avoid division by zero

# 🚀 Drop unnecessary columns
df.drop(columns=["Year"], inplace=True)

# 🚀 Define features (X) and target variable (y)
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target variable

# 🚀 Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 🚀 Feature Scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🚀 Train RandomForest with Hyperparameter Tuning
rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# 🚀 Train XGBoost
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

# 🚀 Train Gradient Boosting
gradient = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gradient.fit(X_train, y_train)

# 🚀 Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 🚀 Evaluate Models
models = {
    "RandomForest": best_rf,
    "XGBoost": xgb,
    "GradientBooster": gradient,
    "LogisticRegression": log_reg
}

# Retrieve original class labels from the LabelEncoder
target_names = label_encoders[target_column].classes_

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 {name} Performance:")
    print(f"   Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=target_names))
