"""
LightGBM training with Optuna hyperparameter search.

Outputs:
  models/lgbm_best.pkl
  data/processed/predictions.parquet
  outputs/calibration_curve.png
"""

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEAT_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
PRED_PATH = BASE_DIR / "data" / "processed" / "predictions.parquet"

MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── Feature columns used in the model ────────────────────────────────────────
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

CATEGORICAL_FEATS = [
    "rating_trajectory",
    "rural_urban_flag",
]

ALL_FEATS = NUMERIC_FEATS + CATEGORICAL_FEATS
TARGET = "fail"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading features...")
df = pd.read_parquet(FEAT_PATH)
print(f"  Shape: {df.shape}")

# Encode categoricals
for col in CATEGORICAL_FEATS:
    df[col] = df[col].fillna("unknown").astype("category")

# Drop rows with no usable features
df_model = df[ALL_FEATS + [TARGET, "FHRSID", "BusinessName", "PostCode", "RatingDate"]].copy()
n_before = len(df_model)
df_model = df_model.dropna(subset=[TARGET])
print(f"  Dropped {n_before - len(df_model):,} rows with null target -> {len(df_model):,}")

X = df_model[ALL_FEATS]
y = df_model[TARGET].astype(int)

# ── Stratified split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df_model.index, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Train failure rate: {y_train.mean()*100:.2f}%")
print(f"  Test  failure rate: {y_test.mean()*100:.2f}%")

# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(y_train.iloc[val_idx], preds))
    return float(np.mean(scores))


print("\nRunning Optuna (50 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)
print(f"  Best PR-AUC: {study.best_value:.4f}")
print(f"  Best params: {study.best_params}")

# ── Train final model with best params ───────────────────────────────────────
print("\nTraining final model...")
best_params = {
    "objective": "binary",
    "metric": "average_precision",
    "verbosity": -1,
    "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
    **study.best_params,
}
model = lgb.LGBMClassifier(**best_params)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["pass", "fail"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
pr_auc = average_precision_score(y_test, y_prob)
print(f"PR-AUC on test set: {pr_auc:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
model_path = MODEL_DIR / "lgbm_best.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved -> {model_path}")

# ── Save predictions ──────────────────────────────────────────────────────────
df_preds = df_model.loc[idx_test, ["FHRSID", "BusinessName", "PostCode", "RatingDate"]].copy()
df_preds["y_true"] = y_test.values
df_preds["y_pred"] = y_pred
df_preds["fail_prob"] = y_prob
PRED_PATH.parent.mkdir(exist_ok=True)
df_preds.to_parquet(PRED_PATH, engine="pyarrow", index=False)
print(f"Predictions saved -> {PRED_PATH}")

# ── Calibration curve ────────────────────────────────────────────────────────
print("\nPlotting calibration curve...")
frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(mean_pred, frac_pos, "s-", label="LightGBM", color="steelblue")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve — Food Safety Failure Model")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "calibration_curve.png", dpi=120)
print(f"Calibration curve saved -> {OUT_DIR / 'calibration_curve.png'}")
