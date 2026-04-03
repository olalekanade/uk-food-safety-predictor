"""
Feature engineering pipeline.

Input:  data/raw/fsa_full.parquet + three raw reference files
Output: data/processed/features.parquet
"""

from datetime import date
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW = BASE_DIR / "data" / "raw"
OUT = BASE_DIR / "data" / "processed" / "features.parquet"
TODAY = date.today()


def log_drop(label: str, before: int, after: int) -> None:
    dropped = before - after
    print(f"  [{label}] dropped {dropped:,} rows ({dropped/before*100:.1f}%) -> {after:,} remaining")


# ── 1. Load FSA base ─────────────────────────────────────────────────────────
print("Loading FSA data...")
df = pd.read_parquet(RAW / "fsa_full.parquet")
print(f"  Base shape: {df.shape}")
n0 = len(df)

# ── 2. Drop Exempt / AwaitingInspection before feature engineering ────────────
print("\nDropping non-scoreable RatingValues...")
exclude = {"Exempt", "AwaitingInspection", "AwaitingPublication", "", "None"}
df = df[~df["RatingValue"].isin(exclude) & df["RatingValue"].notna()]
log_drop("non-numeric RatingValue", n0, len(df))

# Coerce RatingValue to int (0-5)
df["RatingNum"] = pd.to_numeric(df["RatingValue"], errors="coerce")
n1 = len(df)
df = df[df["RatingNum"].notna()]
log_drop("unparseable RatingValue", n1, len(df))

# ── 3. Binary target ──────────────────────────────────────────────────────────
df["fail"] = (df["RatingNum"] <= 2).astype(int)
print(f"\nTarget: {df['fail'].sum():,} failures ({df['fail'].mean()*100:.2f}%)")

# ── 4. days_since_inspection ─────────────────────────────────────────────────
print("\nEngineering days_since_inspection...")
df["RatingDate"] = pd.to_datetime(df["RatingDate"], errors="coerce")
df["days_since_inspection"] = (pd.Timestamp(TODAY) - df["RatingDate"]).dt.days
n2 = len(df)
df = df[df["days_since_inspection"].notna() & (df["days_since_inspection"] >= 0)]
log_drop("missing/future RatingDate", n2, len(df))
df["days_since_inspection"] = df["days_since_inspection"].astype(int)

# ── 5. rating_trajectory ─────────────────────────────────────────────────────
print("\nEngineering rating_trajectory...")
# Sort by business then date to detect multi-inspection businesses
df_sorted = df.sort_values(["BusinessName", "PostCode", "RatingDate"])
grp = df_sorted.groupby(["BusinessName", "PostCode"])["RatingNum"]

first_rating = grp.transform("first")
last_rating = grp.transform("last")
count_inspections = grp.transform("count")

conditions = [
    count_inspections <= 1,
    last_rating > first_rating,
    last_rating < first_rating,
]
choices = ["unknown", "improved", "worsened"]
df["rating_trajectory"] = np.select(conditions, choices, default="stable")
print(f"  trajectory counts:\n{df['rating_trajectory'].value_counts().to_string()}")

# ── 6. IMD join ───────────────────────────────────────────────────────────────
print("\nJoining IMD deprivation data...")
df_imd = pd.read_excel(RAW / "imd_2019_lsoa.xlsx", sheet_name="IMD2019", header=0)
df_imd = df_imd.rename(columns={
    "LSOA code (2011)": "lsoa11cd",
    "Index of Multiple Deprivation (IMD) Rank": "imd_rank",
    "Index of Multiple Deprivation (IMD) Decile": "imd_decile",
})[["lsoa11cd", "imd_rank", "imd_decile"]]

df_pc = pd.read_csv(
    RAW / "postcode_lsoa_lookup.csv",
    usecols=["pcds", "lsoa11cd"],
    dtype=str,
    encoding="latin-1",
)
df_pc["pcds_clean"] = df_pc["pcds"].str.replace(" ", "", regex=False).str.upper()
df_pc = df_pc.drop_duplicates("pcds_clean").set_index("pcds_clean")[["lsoa11cd"]]

df["postcode_clean"] = df["PostCode"].fillna("").str.replace(" ", "", regex=False).str.upper()
df = df.join(df_pc, on="postcode_clean")
df = df.merge(df_imd, on="lsoa11cd", how="left")

imd_matched = df["imd_decile"].notna().sum()
print(f"  IMD matched: {imd_matched:,} / {len(df):,} ({imd_matched/len(df)*100:.1f}%)")

# ── 7. Income & Employment scores (from IoD domain file) ─────────────────────
print("\nJoining income & employment domain scores...")
shutil.copy(RAW / "rural_urban_lsoa.csv", RAW / "rural_urban_lsoa_temp.xlsx")

df_income = pd.read_excel(RAW / "rural_urban_lsoa_temp.xlsx", sheet_name="IoD2019 Income Domain")
df_income = df_income.rename(columns={
    "LSOA code (2011)": "lsoa11cd",
    "Income Domain numerator": "imd_income_score",
})[["lsoa11cd", "imd_income_score"]]

df_employ = pd.read_excel(RAW / "rural_urban_lsoa_temp.xlsx", sheet_name="IoD2019 Employment Domain")
df_employ = df_employ.rename(columns={
    "LSOA code (2011)": "lsoa11cd",
    "Employment Domain numerator": "imd_employment_score",
})[["lsoa11cd", "imd_employment_score"]]

df = df.merge(df_income, on="lsoa11cd", how="left")
df = df.merge(df_employ, on="lsoa11cd", how="left")
print(f"  Income score matched: {df['imd_income_score'].notna().sum():,}")
print(f"  Employment score matched: {df['imd_employment_score'].notna().sum():,}")

# ── 8. rural_urban_flag from Barriers domain ──────────────────────────────────
print("\nDeriving rural_urban_flag from Barriers domain...")
df_barriers = pd.read_excel(RAW / "rural_urban_lsoa_temp.xlsx", sheet_name="IoD2019 Barriers Domain")
df_barriers = df_barriers.rename(columns={
    "LSOA code (2011)": "lsoa11cd",
    "Road distance to a GP surgery indicator (km)": "dist_gp_km",
})[["lsoa11cd", "dist_gp_km"]].dropna()

gp_median = df_barriers["dist_gp_km"].median()
df_barriers["rural_urban_flag"] = np.where(df_barriers["dist_gp_km"] > gp_median, "Rural", "Urban")
df_barriers = df_barriers[["lsoa11cd", "rural_urban_flag"]]

df = df.merge(df_barriers, on="lsoa11cd", how="left")
print(f"  rural_urban_flag matched: {df['rural_urban_flag'].notna().sum():,}")

# ── 9. business_type_encoded (target encoding) ────────────────────────────────
print("\nTarget-encoding BusinessTypeID...")
bt_mean = df.groupby("BusinessTypeID")["fail"].mean()
df["business_type_encoded"] = df["BusinessTypeID"].map(bt_mean)

# ── 10. Select & clean final feature set ─────────────────────────────────────
FEATURE_COLS = [
    "FHRSID",
    "BusinessName",
    "PostCode",
    "BusinessType",
    "BusinessTypeID",
    "Latitude",
    "Longitude",
    "RatingValue",
    "RatingNum",
    "RatingDate",
    "days_since_inspection",
    "rating_trajectory",
    "imd_decile",
    "imd_rank",
    "imd_income_score",
    "imd_employment_score",
    "business_type_encoded",
    "rural_urban_flag",
    "LocalAuthorityName",
    "lsoa11cd",
    "scores_Hygiene",
    "scores_Structure",
    "scores_ConfidenceInManagement",
    "fail",
]
# Keep only columns that exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
df_out = df[FEATURE_COLS].reset_index(drop=True)

print(f"\nFinal feature shape: {df_out.shape}")
print(df_out.dtypes)

OUT.parent.mkdir(parents=True, exist_ok=True)
df_out.to_parquet(OUT, engine="pyarrow", index=False)
print(f"\nSaved to {OUT}")
