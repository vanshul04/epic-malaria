# src/models/explain.py
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

OUT = Path.cwd() / "experiments"
OUT.mkdir(exist_ok=True, parents=True)

print("Loading artifacts...")
cal_model = joblib.load("models/calibrated_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
features = joblib.load("models/feature_list.pkl")  # list of feature names

# If calibrated wrapper exists, get underlying estimator for TreeExplainer where needed
# SHAP TreeExplainer works with tree models (XGBoost, LightGBM, RandomForest). For other models use KernelExplainer (slower).
estimator = getattr(cal_model, "base_estimator", cal_model)

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")["y"]

print("X_test shape:", X_test.shape)

try:
    import shap
    import matplotlib.pyplot as plt
    print("Using SHAP", shap.__version__)
except Exception as e:
    raise SystemExit("Install shap first (pip install shap). Error: " + str(e))

# Choose appropriate explainer
is_tree = False
try:
    # Try to detect tree-like model
    from xgboost import XGBClassifier  # noqa
    is_tree = isinstance(estimator, (XGBClassifier,)) or hasattr(estimator, "get_booster")
except Exception:
    pass

from sklearn.ensemble import RandomForestClassifier
if isinstance(estimator, RandomForestClassifier):
    is_tree = True

if is_tree:
    explainer = shap.TreeExplainer(estimator)
else:
    # KernelExplainer is slower and needs a background dataset. Use a small background sample.
    background = X_test.sample(min(100, len(X_test)), random_state=42)
    explainer = shap.KernelExplainer(estimator.predict_proba, background)

# Compute SHAP values for a sample or full test set (careful with KernelExplainer: expensive)
sample_for_shap = X_test.sample(min(500, len(X_test)), random_state=42)  # limit to 500 rows for speed
print("Computing SHAP values for sample of size:", sample_for_shap.shape)
shap_values = explainer.shap_values(sample_for_shap)  # list (n_classes) of arrays or single array for binary

# Save global summary plot per-class if multiclass
class_names = list(label_encoder.inverse_transform(range(len(shap_values)))) if isinstance(shap_values, list) else [label_encoder.inverse_transform([0])[0]]
print("Class names detected:", class_names)

# Create overall summary plot (for multiclass, shap_values is list)
plt_fname = OUT / "shap_summary.png"
try:
    if isinstance(shap_values, list):
        # For multiclass: combine absolute mean across classes
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        order = np.argsort(mean_abs)[::-1]
        top_feats = [sample_for_shap.columns[i] for i in order[:20]]
        # Use summary_plot for each class too
        for i, cls in enumerate(class_names):
            shap.summary_plot(shap_values[i], sample_for_shap, show=False)
            plt.title(f"SHAP summary - class: {cls}")
            plt.savefig(OUT / f"shap_summary_class_{cls}.png", bbox_inches="tight")
            plt.clf()
        # Also save a combined bar plot (mean abs)
        import matplotlib.pyplot as plt2
        plt2.figure(figsize=(8,6))
        feat_means = pd.Series(mean_abs, index=sample_for_shap.columns).sort_values(ascending=False)[:20]
        feat_means.plot.barh()
        plt2.gca().invert_yaxis()
        plt2.title("Mean |SHAP| (combined classes)")
        plt2.savefig(plt_fname, bbox_inches="tight")
        plt2.close()
    else:
        shap.summary_plot(shap_values, sample_for_shap, show=False)
        plt.savefig(plt_fname, bbox_inches="tight")
        plt.clf()
    print("Saved global SHAP plots to:", OUT)
except Exception as e:
    print("Warning: failed to save some SHAP plots:", e)

# Save per-sample SHAP values for first 10 samples to JSON for UI use
try:
    import json
    sample_idx = sample_for_shap.index[:10]
    sample_df = sample_for_shap.loc[sample_idx]
    if isinstance(shap_values, list):
        # shap_values[class][n_samples, n_features]
        per_sample = []
        for j, idx in enumerate(sample_idx):
            sample_vals = {}
            for i, cls in enumerate(class_names):
                sv = shap_values[i][j].tolist()
                sample_vals[cls] = dict(zip(sample_df.columns.tolist(), sv))
            per_sample.append({"index": int(idx), "features": sample_df.loc[idx].to_dict(), "shap": sample_vals})
    else:
        per_sample = []
        for j, idx in enumerate(sample_idx):
            sv = shap_values[j].tolist() if len(shap_values.shape) == 3 else shap_values[j].tolist()
            per_sample.append({"index": int(idx), "features": sample_df.loc[idx].to_dict(), "shap": dict(zip(sample_df.columns.tolist(), sv))})
    with open(OUT / "shap_sample.json", "w", encoding="utf8") as f:
        json.dump(per_sample, f, ensure_ascii=False, indent=2)
    print("Saved sample SHAP JSON to:", OUT / "shap_sample.json")
except Exception as e:
    print("Warning: could not save sample SHAP JSON:", e)

print("Done.")
