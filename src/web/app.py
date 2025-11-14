# src/api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os, pandas as pd
from logger_config import logger
# then in predict() before return:
logger.info({"input": inp, "prediction": pred_label})


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "models", "disease_rf_model.joblib")
PREP_PATH = os.path.join(ROOT, "models", "preprocessor.joblib")

art = joblib.load(MODEL_PATH)
model = art["model"]
le = art["label_encoder"]
features = art["feature_columns"]

prep = joblib.load(PREP_PATH)
num_cols = prep["num_cols"]
cat_cols = prep["cat_cols"]
ohe = prep["ohe"]

app = FastAPI(title="Disease Predictor API")

class Payload(BaseModel):
    data: dict

def preprocess_input(inp: dict):
    # build DataFrame with expected numeric + ohe columns
    base = {}
    for c in num_cols:
        base[c] = float(inp.get(c, 0))
    # categorical
    for c in cat_cols:
        base[c] = inp.get(c, "")
    df = pd.DataFrame([base])
    if ohe is not None and len(cat_cols)>0:
        ohe_arr = ohe.transform(df[cat_cols])
        ohe_cols = ohe.get_feature_names_out(cat_cols)
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols)
        df = pd.concat([df[num_cols].reset_index(drop=True), df_ohe.reset_index(drop=True)], axis=1)
    else:
        df = df[num_cols]
    # ensure columns order matches model's feature columns
    for f in features:
        if f not in df.columns:
            df[f] = 0
    df = df[features]
    return df

@app.post("/predict")
def predict(payload: Payload):
    inp = payload.data
    X = preprocess_input(inp)
    pred_idx = model.predict(X)[0]
    probs = {}
    try:
        prob_arr = model.predict_proba(X)
        # for RF single estimator, prob_arr is shape (n_samples, n_classes)
        for cls, p in zip(le.classes_, prob_arr[0]):
            probs[cls] = float(p)
    except Exception:
        pass
    pred_label = le.inverse_transform([pred_idx])[0]
    return {"prediction": pred_label, "probabilities": probs}
    