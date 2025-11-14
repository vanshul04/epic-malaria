# src/data_processing/prepare_features.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# Paths
ROOT = Path.cwd()
IN = ROOT / "data" / "cleaned.csv"
OUT_X = ROOT / "data" / "X.csv"
OUT_Y = ROOT / "data" / "y.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned data
df = pd.read_csv(IN)
print("Loaded cleaned data:", df.shape)

# Identify feature columns (all symptom columns)
# We assume 'location' and 'disease' and optional lat/lon are not features
exclude = {"location", "disease", "lat", "lon"}
features = [c for c in df.columns if c not in exclude]
print("Detected features:", features)

# Build X and y
X = df[features].copy()
le = LabelEncoder()
y = le.fit_transform(df["disease"].astype(str))

# Save X, y and artifacts
X.to_csv(OUT_X, index=False)
pd.DataFrame({"y": list(y)}).to_csv(OUT_Y, index=False)
joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
joblib.dump(features, MODEL_DIR / "feature_list.pkl")

print("Saved:")
print(" -", OUT_X)
print(" -", OUT_Y)
print(" -", MODEL_DIR / "label_encoder.pkl")
print(" -", MODEL_DIR / "feature_list.pkl")
