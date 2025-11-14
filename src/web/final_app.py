import streamlit as st
import joblib
import pandas as pd
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "models", "disease_rf_model.joblib")
PREP_PATH = os.path.join(ROOT, "models", "preprocessor.joblib")

# Load model + preprocessor
art = joblib.load(MODEL_PATH)
model = art["model"]
le = art["label_encoder"]
features = art["feature_columns"]

prep = joblib.load(PREP_PATH)
num_cols = prep["num_cols"]
cat_cols = prep["cat_cols"]
ohe = prep["ohe"]

st.set_page_config(page_title="Disease Prediction", layout="wide")
st.title("🩺 Disease Prediction System Based on Symptoms")

st.write("""
This tool predicts the *most likely disease* from your symptoms using a trained ML model.  
**Disclaimer:** This tool is for demo purposes. Always consult a doctor.
""")

st.sidebar.header("Enter Symptoms / Lab Values")

# Inputs based on model features
input_values = {}
for f in num_cols:
    input_values[f] = st.sidebar.number_input(f, value=0, format="%f")

for c in cat_cols:
    input_values[c] = st.sidebar.text_input(c, value="")

if st.sidebar.button("Predict"):
    row = {}

    # numeric values
    for f in num_cols:
        row[f] = input_values[f]

    # categorical
    for c in cat_cols:
        row[c] = input_values[c]

    row_df = pd.DataFrame([row])

    # OHE
    if ohe and cat_cols:
        ohe_arr = ohe.transform(row_df[cat_cols])
        ohe_cols = ohe.get_feature_names_out(cat_cols)
        df_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols)
        final_df = pd.concat([row_df[num_cols].reset_index(drop=True), df_ohe], axis=1)
    else:
        final_df = row_df[num_cols]

    # Ensure features match training
    for f in features:
        if f not in final_df.columns:
            final_df[f] = 0

    final_df = final_df[features]

    pred_idx = model.predict(final_df)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    st.success(f"### 🧬 Predicted Disease: **{pred_label}**")

    try:
        probs = model.predict_proba(final_df)[0]
        prob_chart = pd.DataFrame({
            "Disease": le.classes_,
            "Probability": probs
        })
        st.subheader("📊 Disease Probabilities")
        st.bar_chart(prob_chart.set_index("Disease"))
    except:
        pass

