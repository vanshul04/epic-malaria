# src/api/main.py
import os
import time
import json
import logging
import warnings
from typing import Dict, Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------
# Config / Paths
# -----------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

# Replace this with the actual filename if different
PRIMARY_MODEL_FILENAME = "disease_rf_model.joblib"
FALLBACK_MODEL_FILENAME = "disease_rf_model.joblib"

PRIMARY_MODEL_PATH = os.path.join(MODELS_DIR, PRIMARY_MODEL_FILENAME)
FALLBACK_MODEL_PATH = os.path.join(MODELS_DIR, FALLBACK_MODEL_FILENAME)
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
PREDICTION_LOG = os.path.join(MODELS_DIR, "prediction_log.jsonl")

# -----------------------
# Logging config
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("epic-malaria-api")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Disease Prediction API", version="1.0")

# -----------------------
# Safe model loader
# -----------------------
def safe_load_model(primary_path: str, fallback_path: Optional[str] = None):
    """Attempt to load primary model; fall back to fallback_path if provided."""
    if os.path.exists(primary_path):
        try:
            logger.info(f"Loading primary model from: {primary_path}")
            return joblib.load(primary_path)
        except Exception as e:
            logger.warning(f"Failed to load primary model '{primary_path}': {e}")
    if fallback_path and os.path.exists(fallback_path):
        try:
            logger.info(f"Attempting fallback model from: {fallback_path}")
            return joblib.load(fallback_path)
        except Exception as e:
            logger.warning(f"Failed to load fallback model '{fallback_path}': {e}")
    raise RuntimeError("No model could be loaded from provided paths.")

# -----------------------
# Load model & preprocessor
# -----------------------
try:
    model_artifact = safe_load_model(PRIMARY_MODEL_PATH, FALLBACK_MODEL_PATH)
except Exception as exc:
    # The server can still start but prediction will fail; health endpoint will reflect this.
    logger.exception("Model load failed at startup.")
    model_artifact = None

# Normalize model_artifact to a dict-like structure with keys:
# - "model": estimator object (or artifact may be the estimator itself)
# - "label_encoder": optional LabelEncoder
# - "feature_columns": list of feature names
if model_artifact is None:
    model = None
    label_encoder = None
    feature_columns = None
else:
    if isinstance(model_artifact, dict):
        model = model_artifact.get("model", None)
        label_encoder = model_artifact.get("label_encoder", None)
        feature_columns = model_artifact.get("feature_columns", None)
    else:
        # joblib might have returned the raw estimator
        model = model_artifact
        label_encoder = None
        feature_columns = None

# Try load preprocessor (we saved a small dict with num_cols, cat_cols, ohe)
preprocessor = None
num_cols = []
cat_cols = []
ohe = None
if os.path.exists(PREPROCESSOR_PATH):
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        num_cols = preprocessor.get("num_cols", []) or []
        cat_cols = preprocessor.get("cat_cols", []) or []
        ohe = preprocessor.get("ohe", None)
        logger.info(f"Loaded preprocessor. num_cols={len(num_cols)}, cat_cols={len(cat_cols)}")
    except Exception as e:
        logger.warning(f"Failed to load preprocessor: {e}")
        preprocessor = None

# -----------------------
# Request / response models
# -----------------------
class PredictPayload(BaseModel):
    data: Dict[str, Any]

class PredictResponse(BaseModel):
    prediction: Any
    probabilities: Optional[Dict[str, float]] = None

# -----------------------
# Helper: preprocessing
# -----------------------
def preprocess_input(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Build DataFrame with model's expected features:
    - if a preprocessor exists, use its num_cols and cat_cols & OHE
    - else, attempt to create features from payload keys and align with feature_columns
    """
    # safe copy
    inp = dict(payload)

    # If we have preprocessor info, construct using num_cols & cat_cols
    if preprocessor:
        # numeric features: default 0
        row_num = {c: float(inp.get(c, 0)) if inp.get(c, None) is not None else 0.0 for c in num_cols}
        # categorical features: keep original strings or defaults
        row_cat = {c: inp.get(c, "") for c in cat_cols}
        # make DataFrame
        df_num = pd.DataFrame([row_num])
        if cat_cols:
            df_cat = pd.DataFrame([row_cat])
            if ohe is not None:
                # transform categorical via fitted OHE
                try:
                    ohe_arr = ohe.transform(df_cat)
                    ohe_cols = list(ohe.get_feature_names_out(cat_cols))
                    df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols)
                    df = pd.concat([df_num.reset_index(drop=True), df_ohe.reset_index(drop=True)], axis=1)
                except Exception as e:
                    logger.warning(f"OHE transform failed: {e}; falling back to numeric-only features.")
                    df = df_num
            else:
                # if no OHE, try to include raw cat columns (but many models expect numeric only)
                df = pd.concat([df_num.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
        else:
            df = df_num
    else:
        # No preprocessor saved. We will attempt a best-effort:
        # - Use numeric-looking keys as numbers; other keys as categorical (turn into dummies).
        # - Then align to feature_columns if available.
        numeric_keys = {}
        other_keys = {}
        for k, v in inp.items():
            if isinstance(v, (int, float)):
                numeric_keys[k] = float(v)
            else:
                # try convert numeric-looking strings
                try:
                    numeric_keys[k] = float(v)
                except Exception:
                    other_keys[k] = str(v)
        df_num = pd.DataFrame([numeric_keys])
        if other_keys:
            df_other = pd.DataFrame([other_keys])
            df_other_dummies = pd.get_dummies(df_other, dummy_na=False)
            df = pd.concat([df_num.reset_index(drop=True), df_other_dummies.reset_index(drop=True)], axis=1)
        else:
            df = df_num

    # Ensure all model feature columns exist (if we know them), filling missing with 0
    if feature_columns:
        for f in feature_columns:
            if f not in df.columns:
                df[f] = 0
        # ensure column order matches exactly
        df = df[feature_columns].astype(float).fillna(0)
    else:
        # No feature_columns known: return df as-is after converting types
        df = df.fillna(0)
        # convert all to numeric where possible
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                # leave as-is (model may fail if non-numeric)
                pass

    return df

# -----------------------
# Helper: write prediction log
# -----------------------
def append_prediction_log(entry: Dict[str, Any]):
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(PREDICTION_LOG, "a", encoding="utf-8") as wf:
            wf.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write prediction log: {e}")

# -----------------------
# Health endpoint
# -----------------------
@app.get("/health")
def health():
    ok = model is not None
    return {
        "status": "ok",
        "model_loaded": ok,
        "primary_model": PRIMARY_MODEL_PATH,
        "fallback_model": FALLBACK_MODEL_PATH,
        "preprocessor_loaded": preprocessor is not None
    }

# -----------------------
# Predict endpoint
# -----------------------
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on the server.")

    try:
        X = preprocess_input(payload.data)
    except Exception as e:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    # Prediction
    try:
        pred_idx = model.predict(X)[0]
    except Exception as e:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Resolve predicted label (if we have a label encoder)
    try:
        if label_encoder is not None:
            prediction_label = label_encoder.inverse_transform([pred_idx])[0]
        else:
            # If the model was saved as a dict with label mapping inside, try to fetch it
            # As a fallback, return the raw index/class value
            prediction_label = str(pred_idx)
    except Exception:
        prediction_label = str(pred_idx)

    # Get probabilities if available (some estimators support predict_proba)
    prob_dict = None
    try:
        proba = model.predict_proba(X)
        # For multiclass, map classes -> probabilities
        if label_encoder is not None:
            classes = list(label_encoder.classes_)
        else:
            # try to obtain class labels from model.classes_ if present
            classes = getattr(model, "classes_", None)
            if classes is None:
                classes = [str(c) for c in range(proba.shape[1])]
        prob_vals = proba[0].tolist()
        prob_dict = {str(c): float(p) for c, p in zip(classes, prob_vals)}
    except Exception:
        # model doesn't support predict_proba or failed — ignore probabilities
        prob_dict = None

    # Log the prediction
    log_entry = {
        "timestamp": time.time(),
        "input": payload.data,
        "prediction": prediction_label,
        "probabilities": prob_dict
    }
    append_prediction_log(log_entry)

    return {"prediction": prediction_label, "probabilities": prob_dict}
