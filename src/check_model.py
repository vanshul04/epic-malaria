# src/check_model.py
import joblib, pandas as pd
m = joblib.load("../models/disease_rf_model.joblib")   # adjust name if different
model = m["model"]
le = m["label_encoder"]
features = m["feature_columns"]

print("Classes:", list(le.classes_))
print("Number of features expected by model:", len(features))

# Top 20 feature importances
import pandas as pd
feat = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop 20 features:\n", feat.head(20).to_string())

# Quick test: model predict on first 3 rows of your CSV after preprocessing
df = pd.read_csv("../data/disease_prediction_dataset.csv")
print("\nHead of dataset:\n", df.head(2).to_string())
