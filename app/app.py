"""
UK Food Safety Risk Predictor — Streamlit app.

On first run (Streamlit Cloud) the processed artefacts won't exist.
The app detects this and runs the full pipeline before loading.
"""

import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pgeocode
import shap
import streamlit as st
from streamlit_folium import st_folium

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "lgbm_best.pkl"
FEAT_PATH = BASE / "data" / "processed" / "features.parquet"
SHAP_PATH = BASE / "data" / "processed" / "shap_values.parquet"

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

FEATURE_LABELS = {
    "days_since_inspection": "time since last inspection",
    "imd_decile": "area deprivation level",
    "imd_rank": "area deprivation rank",
    "scores_Hygiene": "hygiene score",
    "scores_ConfidenceInManagement": "management confidence score",
    "business_type_encoded": "business type risk profile",
    "imd_income_score": "area income deprivation",
    "imd_employment_score": "area employment deprivation",
    "rating_trajectory": "rating trend over time",
    "rural_urban_flag": "urban/rural location",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Food Safety Risk Predictor",
    page_icon="🍽️",
    layout="wide",
)

# ── First-run pipeline ────────────────────────────────────────────────────────
def _run_step(label: str, cmd: list[str]) -> bool:
    """Run a pipeline step; return True on success, False on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(
            f"**Pipeline step failed: {label}**\n\n"
            f"```\n{result.stderr[-3000:]}\n```"
        )
        return False
    return True


def build_pipeline_if_needed() -> bool:
    """
    Check whether artefacts exist. If not, run the full pipeline.
    Returns True when artefacts are ready, False if a step failed.
    """
    if FEAT_PATH.exists() and MODEL_PATH.exists():
        return True

    py = sys.executable
    steps = [
        ("FSA data ingest",        [py, "-m", "src.ingest.fsa_ingest"]),
        ("Feature engineering",    [py, "-m", "src.features.build_features"]),
        ("Model training",         [py, "-m", "src.model.train"]),
        ("SHAP explainability",    [py, "-m", "src.model.explain"]),
    ]

    with st.spinner("First run: downloading FSA data and building model — takes around 10 minutes..."):
        for label, cmd in steps:
            st.info(f"Running: {label}...")
            if not _run_step(label, cmd):
                return False

    st.success("Pipeline complete. Loading app...")
    return True


if not build_pipeline_if_needed():
    st.stop()

# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df_feat = pd.read_parquet(FEAT_PATH)
    for col in CATEGORICAL_FEATS:
        df_feat[col] = df_feat[col].fillna("unknown").astype("category")

    X = df_feat[ALL_FEATS]
    df_feat["fail_prob"] = model.predict_proba(X)[:, 1]

    df_shap = pd.read_parquet(SHAP_PATH)
    explainer = shap.TreeExplainer(model)

    return model, df_feat, df_shap, explainer


model, df_feat, df_shap, explainer = load_resources()
nomi = pgeocode.Nominatim("GB")

# ── Session state initialisation ──────────────────────────────────────────────
if "last_postcode" not in st.session_state:
    st.session_state["last_postcode"] = None
if "nearby_businesses" not in st.session_state:
    st.session_state["nearby_businesses"] = None
if "search_lat" not in st.session_state:
    st.session_state["search_lat"] = None
if "search_lon" not in st.session_state:
    st.session_state["search_lon"] = None
if "selected_business" not in st.session_state:
    st.session_state["selected_business"] = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_postcode(postcode: str):
    result = nomi.query_postal_code(postcode.strip().upper())
    if pd.isna(result.latitude) or pd.isna(result.longitude):
        return None, None
    return float(result.latitude), float(result.longitude)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def risk_color(prob: float) -> str:
    if prob > 0.6:
        return "red"
    if prob >= 0.3:
        return "orange"
    return "green"


def feature_label(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat.replace("_", " "))


# ── Header ────────────────────────────────────────────────────────────────────
st.title("UK Food Safety Risk Predictor")
st.warning(
    "**Research tool only — not for official inspection use.** "
    "Predictions are derived from a statistical model and may be inaccurate."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Search")
postcode_input = st.sidebar.text_input("Enter postcode", placeholder="e.g. SW1A 1AA")
search_btn = st.sidebar.button("Search")

# ── On Search: compute results and store in session_state ─────────────────────
if search_btn and postcode_input:
    lat, lon = resolve_postcode(postcode_input)

    if lat is None:
        st.sidebar.error(f"Could not resolve postcode: {postcode_input}")
    else:
        df_local = df_feat.dropna(subset=["Latitude", "Longitude", "fail_prob"]).copy()
        df_local = df_local[
            df_local["Latitude"].between(49, 61) & df_local["Longitude"].between(-8, 2)
        ]
        df_local["dist_km"] = haversine_km(
            lat, lon, df_local["Latitude"].values, df_local["Longitude"].values
        )
        df_nearby = df_local[df_local["dist_km"] <= 3.0].copy()

        st.session_state["last_postcode"] = postcode_input.strip().upper()
        st.session_state["search_lat"] = lat
        st.session_state["search_lon"] = lon
        st.session_state["nearby_businesses"] = df_nearby
        st.session_state["selected_business"] = None   # reset selection on new search

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Risk Map", "Top At-Risk", "Why Flagged?"])

df_nearby = st.session_state["nearby_businesses"]
last_postcode = st.session_state["last_postcode"]
lat = st.session_state["search_lat"]
lon = st.session_state["search_lon"]

# ── Tab 1: Risk Map ───────────────────────────────────────────────────────────
with tab1:
    if df_nearby is None:
        st.info("Enter a UK postcode in the sidebar and click Search.")
    elif df_nearby.empty:
        st.warning(f"No businesses found within 3km of {last_postcode}.")
    else:
        st.subheader(f"Businesses within 3km of {last_postcode}")
        st.caption(f"{len(df_nearby):,} businesses found")

        m = folium.Map(location=[lat, lon], zoom_start=14, tiles="CartoDB positron")
        folium.Marker(
            [lat, lon],
            popup="Your postcode",
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(m)

        for _, row in df_nearby.iterrows():
            color = risk_color(row["fail_prob"])
            date_str = str(row["RatingDate"])[:10] if pd.notna(row.get("RatingDate")) else "Unknown"
            popup_html = (
                f"<b>{row.get('BusinessName', 'Unknown')}</b><br>"
                f"Type: {row.get('BusinessType', 'Unknown')}<br>"
                f"Risk score: {row['fail_prob']*100:.1f}%<br>"
                f"Last inspected: {date_str}"
            )
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)

        st_folium(m, width=900, height=550)

# ── Tab 2: Top At-Risk ────────────────────────────────────────────────────────
with tab2:
    if df_nearby is None:
        st.info("Enter a UK postcode in the sidebar and click Search.")
    elif df_nearby.empty:
        st.warning(f"No businesses found within 3km of {last_postcode}.")
    else:
        st.subheader("Top 10 Highest-Risk Businesses Nearby")

        top10_df = df_nearby.nlargest(10, "fail_prob").reset_index()  # preserves original index
        top10_display = (
            top10_df[["BusinessName", "BusinessType", "RatingDate", "days_since_inspection", "fail_prob"]]
            .rename(columns={
                "BusinessName": "Name",
                "BusinessType": "Type",
                "RatingDate": "Last Inspected",
                "days_since_inspection": "Days Since",
                "fail_prob": "Risk Score",
            })
            .copy()
        )
        top10_display["Risk Score"] = (top10_display["Risk Score"] * 100).round(1).astype(str) + "%"
        top10_display["Last Inspected"] = top10_display["Last Inspected"].astype(str).str[:10]

        selected = st.dataframe(
            top10_display,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        if selected and selected.selection.rows:
            row_pos = selected.selection.rows[0]
            orig_idx = top10_df.loc[row_pos, "index"]
            st.session_state["selected_business"] = df_feat.loc[orig_idx]
            st.success(
                f"Selected: {st.session_state['selected_business'].get('BusinessName', 'Unknown')} "
                f"— switch to **Why Flagged?** tab to see the explanation."
            )

# ── Tab 3: Why Flagged? ───────────────────────────────────────────────────────
with tab3:
    selected_row = st.session_state["selected_business"]

    if selected_row is None and df_nearby is None:
        st.info("Enter a UK postcode in the sidebar and click Search.")
    elif selected_row is None:
        st.info("Select a business from the **Top At-Risk** tab to see its explanation.")
    else:
        biz_name = selected_row.get("BusinessName", "Selected business")
        st.subheader(f"SHAP explanation for: {biz_name}")

        X_row = pd.DataFrame([selected_row[ALL_FEATS]])
        for col in CATEGORICAL_FEATS:
            X_row[col] = X_row[col].astype("category")

        sv = explainer.shap_values(X_row)
        sv_row = sv[1][0] if isinstance(sv, list) else sv[0]

        shap_series = pd.Series(sv_row, index=ALL_FEATS).sort_values(key=abs, ascending=False)
        top3 = shap_series.head(3)

        reasons = []
        for feat, sv_val in top3.items():
            direction = "increases" if sv_val > 0 else "reduces"
            reasons.append(f"**{feature_label(feat)}** {direction} risk")
        st.info("This business is flagged mainly because: " + ", ".join(reasons) + ".")

        top_n = shap_series.head(10)
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_n.values]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            [f.replace("_", " ") for f in top_n.index[::-1]],
            top_n.values[::-1],
            color=colors[::-1],
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"SHAP Waterfall — {biz_name}", fontweight="bold")
        ax.set_xlabel("SHAP value (impact on fail probability)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Global Feature Importance")
        st.image(str(BASE / "outputs" / "shap_global.png"))
