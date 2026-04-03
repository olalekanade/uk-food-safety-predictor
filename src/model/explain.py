"""
SHAP explainability for the food safety LightGBM model.

Outputs:
  outputs/shap_global.png
  outputs/shap_beeswarm.png
  data/processed/shap_values.parquet
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEAT_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
MODEL_PATH = BASE_DIR / "models" / "lgbm_best.pkl"
OUT_DIR = BASE_DIR / "outputs"
SHAP_PATH = BASE_DIR / "data" / "processed" / "shap_values.parquet"

OUT_DIR.mkdir(exist_ok=True)

# ── Same feature definitions as train.py ──────────────────────────────────────
NUMERIC_FEATS = [
    "days_since_inspection",
    "imd_decile",
    "imd_rank",
    "imd_income_score",
    "imd_employment_score",
    "business_type_encoded",
    "scores_Hygiene",
    "scores_ConfidenceInManagement",
    "Latitude",
    "Longitude",
]
CATEGORICAL_FEATS = ["rating_trajectory", "rural_urban_flag"]
ALL_FEATS = NUMERIC_FEATS + CATEGORICAL_FEATS

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ── Load features ─────────────────────────────────────────────────────────────
print("Loading features...")
df = pd.read_parquet(FEAT_PATH)
for col in CATEGORICAL_FEATS:
    df[col] = df[col].fillna("unknown").astype("category")

# Use a sample for SHAP (full dataset takes long; 20k is sufficient for global)
sample_size = min(20_000, len(df))
df_sample = df[ALL_FEATS].sample(n=sample_size, random_state=42)
print(f"  Computing SHAP on {sample_size:,} samples...")

# ── SHAP TreeExplainer ────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_sample)

# LightGBM binary returns list [neg_class, pos_class] or single array
if isinstance(shap_values, list):
    sv = shap_values[1]   # positive class (fail)
else:
    sv = shap_values

# ── Global feature importance bar plot ───────────────────────────────────────
print("Saving SHAP global importance plot...")
mean_abs_shap = np.abs(sv).mean(axis=0)
feat_importance = pd.Series(mean_abs_shap, index=ALL_FEATS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
feat_importance.plot.barh(ax=ax, color="steelblue")
ax.set_title("SHAP Global Feature Importance", fontweight="bold")
ax.set_xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_global.png", dpi=120)
plt.close()
print(f"  Saved -> {OUT_DIR / 'shap_global.png'}")

# ── Beeswarm plot ─────────────────────────────────────────────────────────────
print("Saving SHAP beeswarm plot...")
shap_expl = shap.Explanation(
    values=sv,
    base_values=np.full(len(sv), explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1]),
    data=df_sample.values,
    feature_names=ALL_FEATS,
)
fig, ax = plt.subplots(figsize=(10, 7))
shap.plots.beeswarm(shap_expl, max_display=12, show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved -> {OUT_DIR / 'shap_beeswarm.png'}")

# ── Save per-business SHAP values ────────────────────────────────────────────
print("Saving SHAP values parquet...")
df_shap = pd.DataFrame(sv, columns=[f"shap_{c}" for c in ALL_FEATS])
df_shap["FHRSID"] = df.loc[df_sample.index, "FHRSID"].values
df_shap["BusinessName"] = df.loc[df_sample.index, "BusinessName"].values
df_shap["PostCode"] = df.loc[df_sample.index, "PostCode"].values
df_shap["fail_prob"] = model.predict_proba(df_sample)[:, 1]

SHAP_PATH.parent.mkdir(exist_ok=True)
df_shap.to_parquet(SHAP_PATH, engine="pyarrow", index=False)
print(f"  Saved -> {SHAP_PATH}")
print(f"\nDone. SHAP values shape: {df_shap.shape}")
