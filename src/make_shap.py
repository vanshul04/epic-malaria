import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "disease_rf_model.joblib")
DATA_PATH = os.path.join(ROOT, "data", "disease_prediction_dataset.csv")

# Load model
art = joblib.load(MODEL_PATH)
model = art["model"]
features = art["feature_columns"]

# Load dataset
df = pd.read_csv(DATA_PATH)

# Use only numeric columns that exist in model features
X = df[features].select_dtypes(include=['number']).fillna(0)

# Take small sample for SHAP
X_sample = X.head(200)

#
