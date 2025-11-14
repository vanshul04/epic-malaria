# src/models/train_models.py
import pandas as pd
import joblib
import traceback
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

OUT_DIR = Path.cwd() / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")["y"]
y_test = pd.read_csv("data/y_test.csv")["y"]

print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)

# Try XGBoost first, otherwise fallback to RandomForest
model = None
try:
    from xgboost import XGBClassifier
    print("Using XGBoost classifier")
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
except Exception as e:
    print("XGBoost import failed — falling back to RandomForest.")
    print("Import error:", e)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

# Fit
print("Fitting model — this may take a moment...")
model.fit(X_train, y_train)

# Save raw model
model_path = OUT_DIR / ("xgb_or_rf_baseline.pkl")
joblib.dump(model, model_path)
print("Saved model to", model_path)

# Evaluate
y_pred = model.predict(X_test)
print("\n=== Classification report (test) ===")
print(classification_report(y_test, y_pred))

print("=== Confusion matrix ===")
print(confusion_matrix(y_test, y_pred))

# Cross-val (quick)
try:
    print("\n=== 3-fold CV (accuracy) on training data ===")
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
    print("CV accuracy scores:", scores)
    print("CV accuracy mean:", scores.mean())
except Exception:
    print("Cross-val failed (possibly heavy).")
    traceback.print_exc()
