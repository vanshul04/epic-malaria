# src/save_preprocessor.py
import os, pandas as pd, joblib
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "disease_prediction_dataset.csv")
OUT_PATH = os.path.join(ROOT, "models", "preprocessor.joblib")

df = pd.read_csv(DATA_PATH)
known = ["typhoid","cholera","hepatitis","leptospirosis","amoebiasis","shigellosis","malaria","dengue"]
disease_cols = [c for c in df.columns if any(k in c.lower() for k in known)]

# create label column if needed
if "label" not in df.columns:
    def pick_label(r): 
        pos = [c for c in disease_cols if int(r.get(c,0))==1]
        return pos[0] if pos else "NoDisease"
    df["label"] = df.apply(pick_label, axis=1)

X = df.drop(columns=(disease_cols + ["label"]), errors='ignore').copy()

# Drop ID-like columns
id_cols = [c for c in X.columns if 'id' in c.lower() or c.lower().startswith('patient') or c.lower().endswith('id')]
X.drop(columns=id_cols, inplace=True, errors='ignore')

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
low_card_cats = [c for c in cat_cols if X[c].nunique() <= 30]

# We'll save the list of numeric and categorical features and an OHE fitted on low-card cats
preprocessor = {
    "num_cols": num_cols,
    "cat_cols": low_card_cats,
    "ohe": None
}
if low_card_cats:
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe.fit(X[low_card_cats])
    preprocessor["ohe"] = ohe

joblib.dump(preprocessor, OUT_PATH)
print("Saved preprocessor to:", OUT_PATH)
print("Numeric cols:", len(num_cols), "OHE cols:", low_card_cats)
