# src/save_evaluation.py
import joblib, pandas as pd, numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "disease_rf_model.joblib")
DATA_PATH = os.path.join(ROOT, "data", "disease_prediction_dataset.csv")
OUT_DIR = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# load model
art = joblib.load(MODEL_PATH)
model = art["model"]
le = art["label_encoder"]
features = art["feature_columns"]

# Load dataset and recreate X,y the same way train.py did
df = pd.read_csv(DATA_PATH)
known = ["typhoid","cholera","hepatitis","leptospirosis","amoebiasis","shigellosis","malaria","dengue"]
disease_cols = [c for c in df.columns if any(k in c.lower() for k in known)]

# generate single label if needed
if "label" not in df.columns:
    def pick_label(r): 
        pos = [c for c in disease_cols if int(r.get(c,0))==1]
        return pos[0] if pos else "NoDisease"
    df["label"] = df.apply(pick_label, axis=1)

# Preprocess like train.py: drop ids and disease cols, drop high-cardinality cats
X = df.drop(columns=(disease_cols + ["label"]), errors='ignore').copy()
id_cols = [c for c in X.columns if 'id' in c.lower() or c.lower().startswith('patient') or c.lower().endswith('id')]
X.drop(columns=id_cols, inplace=True, errors='ignore')
# numeric + ohe low-card cats
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
low_card_cats = [c for c in cat_cols if X[c].nunique() <= 30]
# simple transform: numeric + one-hot low card cats (same logic used when saving preprocessor)
if low_card_cats:
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe_arr = ohe.fit_transform(X[low_card_cats])
    import pandas as pd
    ohe_cols = ohe.get_feature_names_out(low_card_cats)
    X_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=X.index)
    X_proc = pd.concat([X[num_cols].reset_index(drop=True), X_ohe.reset_index(drop=True)], axis=1)
else:
    X_proc = X[num_cols].copy()
X_proc.fillna(0, inplace=True)

# Align columns to model features
for f in features:
    if f not in X_proc.columns:
        X_proc[f] = 0
X_proc = X_proc[features]

# stratified split (same random_state)
y = df["label"].astype(str)
from sklearn.preprocessing import LabelEncoder
le_check = LabelEncoder().fit(y)
y_enc = le_check.transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_proc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Save
pd.DataFrame(report).transpose().to_csv(os.path.join(OUT_DIR, "classification_report_by_class.csv"))
pd.DataFrame(cm).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"), index=False, header=False)
pd.DataFrame([{"accuracy": acc}]).to_csv(os.path.join(OUT_DIR, "overall_accuracy.csv"), index=False)

print("Saved report & confusion matrix to", OUT_DIR)
print("Accuracy:", acc)
