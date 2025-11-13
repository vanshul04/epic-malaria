# src/models/calibrate.py
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, log_loss
from pathlib import Path

# Paths
ROOT = Path.cwd()
MODEL_PATH = ROOT / "models" / "xgb_or_rf_baseline.pkl"
OUT_PATH = ROOT / "models" / "calibrated_model.pkl"

# Load data and model
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")["y"]
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")["y"]

print("✅ Loaded training and test data.")
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

base_model = joblib.load(MODEL_PATH)
print("✅ Loaded base model from:", MODEL_PATH.name)

# Apply calibration
print("\nCalibrating model (isotonic method)... this may take a few minutes.")
cal_model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
cal_model.fit(X_train, y_train)

# Evaluate calibration quality
probs = cal_model.predict_proba(X_test)
preds = cal_model.predict(X_test)
loss = log_loss(y_test, probs)
print("\n=== Log Loss (lower is better):", round(loss, 4))
print("\n=== Classification Report ===")
print(classification_report(y_test, preds))

# Save the calibrated model
joblib.dump(cal_model, OUT_PATH)
print("\n✅ Calibrated model saved to:", OUT_PATH)
