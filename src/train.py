# src/train.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------- CONFIG -------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "disease_prediction_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "disease_rf_model.joblib")
EVAL_PATH = os.path.join(PROJECT_ROOT, "models", "model_eval_summary.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.20
# ----------------------

print("Project root:", PROJECT_ROOT)
print("Loading CSV from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)

# -------------------------
# 1) Detect disease indicator columns (one-hot style)
# -------------------------
known = ["typhoid","cholera","hepatitis","leptospirosis","amoebiasis","shigellosis","malaria","dengue"]
disease_cols = [c for c in df.columns if any(k in c.lower() for k in known)]
print("Detected disease columns:", disease_cols)

if not disease_cols and "disease" in df.columns:
    # single label exists
    label_col = "disease"
    df["label"] = df[label_col].astype(str)
else:
    # create single-label by picking first positive indicator (if any)
    def pick_label(row):
        positives = [c for c in disease_cols if int(row.get(c, 0)) == 1]
        if len(positives) == 1:
            return positives[0]
        elif len(positives) > 1:
            return positives[0]  # simple rule: pick first if multiple
        else:
            return "NoDisease"
    df["label"] = df.apply(pick_label, axis=1)

print("Label distribution (top 10):")
print(df["label"].value_counts().head(10))

# -------------------------
# 2) Prepare features X and label y
# -------------------------
# Keep original CSV unchanged; all processing in memory.
# Drop disease indicator columns (to avoid label leakage)
X = df.drop(columns=(disease_cols + ["label"]), errors='ignore').copy()
y = df["label"].astype(str).copy()

# 2a) Drop id-like columns (PatientID etc.)
id_cols = [c for c in X.columns if 'id' in c.lower() or c.lower().startswith('patient') or c.lower().endswith('id')]
print("ID-like columns detected (will be dropped):", id_cols)
X.drop(columns=id_cols, inplace=True, errors='ignore')

# 2b) Separate numeric & categorical
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
print("Numeric cols count:", len(num_cols))
print("Categorical cols detected:", cat_cols)

# 2c) For categorical columns: keep only low-cardinality ones, drop high-cardinality text
LOW_CARD_THRESHOLD = 30
low_card_cats = [c for c in cat_cols if X[c].nunique() <= LOW_CARD_THRESHOLD]
high_card_cats = [c for c in cat_cols if c not in low_card_cats]
print("Low-cardinality categorical columns (OHE will be applied):", low_card_cats)
print("High-cardinality categorical columns (will be dropped):", high_card_cats)

# Drop high-cardinality categorical columns (likely names/IDs/long text)
X.drop(columns=high_card_cats, inplace=True, errors='ignore')

# 2d) One-hot encode low-cardinality categorical columns
if low_card_cats:
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe_arr = ohe.fit_transform(X[low_card_cats])
    ohe_cols = ohe.get_feature_names_out(low_card_cats)
    X_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=X.index)
    X = pd.concat([X[num_cols], X_ohe], axis=1)
else:
    X = X[num_cols]

# 2e) Final cleanup: fill NA
X.fillna(0, inplace=True)
print("Final feature matrix shape:", X.shape)

# -------------------------
# 3) Encode labels
# -------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", le.classes_)

# If only one class present, abort
if len(le.classes_) < 2:
    raise SystemExit("Only one target class detected. Cannot train classifier.")

# -------------------------
# 4) Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)
print("Train/test sizes:", X_train.shape, X_test.shape)

# -------------------------
# 5) Train RandomForest
# -------------------------
model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
print("Training RandomForest...")
model.fit(X_train, y_train)
print("Training complete.")

# -------------------------
# 6) Evaluation
# -------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy: {:.4f}".format(acc))
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importances (top 15)
feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)
print("\nTop 10 features:")
print(feat_imp.head(10).to_string(index=False))

# -------------------------
# 7) Save model and eval summary (does NOT modify dataset)
# -------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump({
    "model": model,
    "label_encoder": le,
    "feature_columns": X.columns.tolist()
}, MODEL_PATH)
print("Model saved to:", MODEL_PATH)

pd.DataFrame([{
    "accuracy": float(acc),
    "n_classes": len(le.classes_),
    "n_features": X.shape[1],
    "n_samples": len(df)
}]).to_csv(EVAL_PATH, index=False)
print("Eval summary saved to:", EVAL_PATH)

print("Done. Model training finished without altering your original CSV.")
