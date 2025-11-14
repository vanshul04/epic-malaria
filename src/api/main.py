# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
from pathlib import Path
import logging
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path.cwd()
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("epic-malaria-api")

# Load artifacts
CAL_MODEL_PATH = MODELS_DIR / "calibrated_model.pkl"
LE_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURES_PATH = MODELS_DIR / "feature_list.pkl"

if not CAL_MODEL_PATH.exists() or not LE_PATH.exists() or not FEATURES_PATH.exists():
    raise SystemExit(f"Required model artifacts missing. Expected: {CAL_MODEL_PATH}, {LE_PATH}, {FEATURES_PATH}")

model = joblib.load(CAL_MODEL_PATH)
label_encoder = joblib.load(LE_PATH)
features: List[str] = joblib.load(FEATURES_PATH)

# For safety: ensure features is list of strings
features = [str(f) for f in features]

app = FastAPI(title="Epic Malaria - Symptom Classifier API", version="0.1")

# Allow CORS for local dev (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming symptom payload
class SymptomPayload(BaseModel):
    fever: conint(ge=0, le=1)
    headache: conint(ge=0, le=1)
    body_pain: conint(ge=0, le=1)
    nausea: conint(ge=0, le=1)
    vomiting: conint(ge=0, le=1)
    diarrhea: conint(ge=0, le=1)
    abdominal_pain: conint(ge=0, le=1)
    jaundice: conint(ge=0, le=1)
    chills: conint(ge=0, le=1)
    rash: conint(ge=0, le=1)
    joint_pain: conint(ge=0, le=1)
    eye_redness: conint(ge=0, le=1)
    location: Optional[str] = None

# Simple health check
@app.get("/health")
def health():
    return {"status":"ok"}

# Predict endpoint
@app.post("/predict")
def predict(payload: SymptomPayload):
    try:
        # Convert payload to DataFrame with proper feature ordering
        payload_dict = payload.dict()
        loc = payload_dict.pop("location", None)
        # Ensure all expected features are present
        row = {f: int(payload_dict.get(f, 0)) for f in features}
        df = pd.DataFrame([row], columns=features)

        probs = model.predict_proba(df)[0]
        # classes in same order as label_encoder.classes_
        classes = list(label_encoder.inverse_transform(range(len(probs))))
        result = {c: float(p) for c, p in zip(classes, probs)}
        top_class = max(result, key=result.get)

        # Log prediction (rotate to file in production)
        logger.info(f"Prediction: loc={loc} top={top_class} probs={result}")

        return {
            "top": top_class,
            "probabilities": result,
            "location_received": loc
        }
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

# Map-data endpoint: aggregated counts per location for a disease
@app.get("/map-data")
def map_data(disease: Optional[str] = "Malaria"):
    try:
        df = pd.read_csv(DATA_DIR / "cleaned_geocoded.csv")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Geocoded data not found. Run geocode_locations and ensure data/cleaned_geocoded.csv exists.")

    if disease:
        df = df[df["disease"].str.lower() == disease.lower()]

    # Group by location/lat/lon and return counts
    grouped = (
        df.groupby(["location", "lat", "lon"])
        .size()
        .reset_index(name="count")
        .dropna(subset=["lat", "lon"])
    )

    # Convert to list of dicts
    records = grouped.to_dict(orient="records")
    return {"disease": disease, "locations": records}

# Simple endpoint to return model class list and features
@app.get("/metadata")
def metadata():
    return {"classes": list(label_encoder.classes_), "features": features}
