# UK Food Safety Risk Predictor

An open-source machine-learning pipeline that predicts the probability of a UK food business receiving a low hygiene rating (0–2 on the Food Standards Agency scale). The FSA uses proprietary risk-scoring internally to prioritise inspection resources; this project replicates and extends that approach using publicly available data, applying LightGBM with SHAP explainability over the full ~600k establishment dataset.

---

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| FSA Food Hygiene Ratings | [ratings.food.gov.uk API](https://api.ratings.food.gov.uk) | ~600k UK food businesses with ratings, scores, dates |
| English IMD 2019 | [MHCLG / ONS](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) | LSOA-level deprivation deciles and ranks |
| Postcode–LSOA Lookup | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk) | Maps UK postcodes to LSOA codes |
| IoD 2019 Domain Scores | [MHCLG](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) | Income, employment, barriers domain numerators by LSOA |

---

## How to Run Locally

```bash
# 1. Clone and set up
git clone <repo-url>
cd uk-food-safety-predictor
python -m venv venv
source venv/Scripts/activate        # Windows
pip install -r requirements.txt

# 2. Download data (place in data/raw/)
#    - FSA data is fetched automatically:
python -m src.ingest.fsa_ingest

# 3. Build features
python -m src.features.build_features

# 4. Train model (Optuna, 50 trials)
python -m src.model.train

# 5. SHAP explainability
python -m src.model.explain

# 6. Launch app
streamlit run app/app.py
```

---

## Key Findings

**481,636 establishments analysed** after excluding businesses marked 
Exempt or Awaiting Inspection, which have no numeric rating to predict.

**Overall failure rate: 2.99% (14,392 businesses).** While this sounds 
low, at national scale it represents tens of thousands of businesses 
serving food to the public below acceptable hygiene standards at any 
given time.

**Deprivation is the strongest predictor.** Failure rate in the most 
deprived areas (IMD decile 1) is 5.8% — nearly 3× the rate in the 
least deprived areas (decile 10, 1.9%). This held consistently across 
all deciles, not just at the extremes.

**Pearson r = −0.957 (p < 0.0001)** between IMD decile and failure 
rate. This near-perfect negative correlation means that knowing only 
the postcode of a new business gives substantial predictive power 
before any inspection has taken place — suggesting the FSA could use 
deprivation-adjusted scores to prioritise first inspections of newly 
registered premises.

**Model PR-AUC: 0.842 with 89% recall on failures.** PR-AUC is used 
instead of accuracy because only ~3% of businesses fail — a model that 
predicts "pass" for everyone would achieve 97% accuracy while being 
completely useless. PR-AUC measures how well the model ranks actual 
failures above passes, which is what matters for inspection triage.

**76.5% of FSA records matched to IMD data** via postcode lookup. The 
remaining 23.5% were mostly Scottish businesses (which use a different 
inspection scheme) and a small number of malformed or missing postcodes.

---

## Architecture

```
uk-food-safety-predictor/
│
├── data/
│   ├── raw/                   # Source data (not committed)
│   └── processed/             # features.parquet, predictions.parquet, shap_values.parquet
│
├── src/
│   ├── ingest/
│   │   └── fsa_ingest.py      # Fetches all 363 local authorities from FSA API
│   ├── features/
│   │   └── build_features.py  # Joins FSA + IMD + IoD; engineers 24 features
│   └── model/
│       ├── train.py           # LightGBM + Optuna 50-trial HPO -> PR-AUC 0.84
│       └── explain.py         # SHAP TreeExplainer -> global + beeswarm plots
│
├── notebooks/
│   └── 01_eda.ipynb           # EDA: rating distribution, IMD join, geospatial map
│
├── app/
│   └── app.py                 # Streamlit: map, top-10 table, SHAP waterfall
│
├── outputs/                   # Plots and HTML map (committed)
└── models/                    # lgbm_best.pkl (not committed)
```

---

## Limitations & Future Work

- **England-only IMD**: Scottish, Welsh and Northern Irish LSOA codes do not appear in the English IMD file, leaving ~24% of records without deprivation features.
- **Single inspection snapshot**: the FSA API returns only the most recent rating per establishment; temporal modelling of re-inspection dynamics is not possible with this data alone.
- **No operator-level features**: ownership chains, franchise membership, or prior enforcement actions are not available in the open dataset.
- **Calibration**: while PR-AUC is strong, the model is not well-calibrated at very low probabilities — a Platt scaling step would improve score interpretability.
- **Future work**: integrate Companies House data for operator history; add seasonal re-inspection patterns; retrain monthly using the `--daily` flag of `fsa_ingest.py`.

---