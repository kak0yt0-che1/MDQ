# Hidden-Entrepreneur Detection Pipeline — Full Review Report

> **Project:** MDQ — Identify consumer cardholders who transact like businesses  
> **Date:** May 26, 2026  
> **Data:** Synthetic card transactions (consumer + business), 6-month window  

---

## 1. Data Overview

| Dataset | File | Description |
|---------|------|-------------|
| Consumer transactions | `consumer_cards_MDQ.parquet` (~154 MB) | Individual consumer card transactions |
| Business transactions | `business_cards_MDQ.parquet` (~53 MB) | Known business card transactions |
| Merchant reference | `merchants_reference.parquet` (~45 KB) | Merchant metadata (recurring_capable, etc.) |
| Feature matrix | `features_card_level.parquet` (~17 MB) | Aggregated card-level features (one row per card) |

- **Unit of analysis:** card (`card_number`)
- **Label:** `label = 1` → business card, `label = 0` → consumer card
- **Population:** ~80,000 consumer cards + ~20,000 business cards (estimated from positive rate ≈ 0.20)

---

## 2. Feature Engineering — 35+ Card-Level Features

All transactions are aggregated to **one row per card**. The following feature groups were created:

### 2.1 Spending Volume & Distribution
| Feature | Description |
|---------|-------------|
| `tx_count` | Total number of transactions |
| `amt_sum` | Total spend (KZT) |
| `amt_mean` | Mean transaction amount |
| `amt_median` | Median transaction amount |
| `amt_std` | Standard deviation of amounts |
| `amt_max` / `amt_min` | Max / min single transaction |
| `amt_cv` | Coefficient of variation (std / mean) |

### 2.2 Channel & Timing
| Feature | Description |
|---------|-------------|
| `online_share` | Share of online transactions |
| `weekend_share` | Share of weekend transactions |
| `evening_share` | Share of evening (18–23h) transactions |
| `bizhours_share` | Share of business-hours (M–F, 9–17) transactions |
| `foreign_share` | Share of foreign-country transactions |

### 2.3 B2B / Wholesale Exposure
| Feature | Description |
|---------|-------------|
| `b2b_mcc_share` | Share of transactions at B2B MCCs (domain-curated list) |
| `b2b_amt_share` | Share of **spend** (not count) at B2B MCCs |
| `b2b_unique_merchants` | Count of distinct B2B merchants used |

The B2B MCC basket is defined in `config.py` — 40 codes across wholesale/distribution, professional services, and logistics. MCC `5122` (pharmaceutical wholesale) was **excluded** because it is business-exclusive in this sample (would leak the label).

### 2.4 Merchant Concentration
| Feature | Description |
|---------|-------------|
| `n_unique_merchants` | Distinct merchants |
| `n_unique_mcc` | Distinct MCC codes |
| `merchant_hhi` | Herfindahl–Hirschman Index of merchant concentration |
| `merchant_top_ratio` | Share of transactions at the top-1 merchant |
| `merchant_entropy` | Shannon entropy of merchant distribution |
| `merchants_per_tx` | Merchants / total transactions |

### 2.5 Recurrence & Behavior Patterns
| Feature | Description |
|---------|-------------|
| `recurring_share` | Share of transactions flagged `is_recurring` |
| `recurring_capable_share` | Share at merchants with `recurring_capable = True` |
| `tokenized_share` | Share of tokenized transactions |

### 2.6 Activity Cadence & Burstiness
| Feature | Description |
|---------|-------------|
| `active_days` / `active_weeks` | Number of distinct active days / weeks |
| `span_days` | Calendar span from first to last transaction + 1 |
| `tx_per_active_day` | Transactions per active day |
| `recency_days` | Days since last transaction to end of observation window |
| `gap_mean` / `gap_std` | Mean / std of inter-arrival time (hours) |
| `burstiness` | Goh–Barabási burstiness index: (σ − μ) / (σ + μ), range [−1, 1] |
| `monthly_spend_cv` | CV of monthly spend totals |

---

## 3. Leakage Detection & Exclusion

| Source | Action | Reason |
|--------|--------|--------|
| `card_tier` | **Dropped entirely** | 100% `"Business"` for business cards, 0% for consumer — perfect leak |
| `bank_name` | **Dropped** | No predictive signal |
| MCC `5122` | **Removed from B2B basket** | Business-exclusive in this sample → would leak into `b2b_*_share` |
| 32 business-only MCCs | Not one-hot encoded | Coverage artifacts of synthetic data, not true signal |
| 428 consumer-only MCCs | Not one-hot encoded | Same artifact treatment |
| 105 business-only merchant IDs | Not one-hot encoded | Coverage artifact |
| 1,684 consumer-only merchant IDs | Not one-hot encoded | Coverage artifact |

**Verdict:** No residual leakage remains — univariate AUC confirms no single feature achieves AUC = 1.0.

---

## 4. Modeling

### 4.1 Split Strategy

- **Stratified 80/20 random split** on cards (no card in both train and validation)
- A chronological split is **not needed**: the label is an intrinsic card type observed over a fixed window, not a future event
- **Seed:** `SEED = 42` (deterministic across all components)

### 4.2 Models Trained

| Model | Configuration |
|-------|--------------|
| **Logistic Regression** | `C=1.0`, `class_weight="balanced"`, log1p + StandardScaler on skewed features |
| **LightGBM** (main) | `n_estimators=400`, `lr=0.05`, `num_leaves=31`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` |
| **Random Forest** | `n_estimators=300`, `min_samples_leaf=20`, `class_weight="balanced"` |
| **CatBoost** (archive) | Results exist in `archive/model_comparison.csv` but not in the main pipeline |

### 4.3 Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|-----|---------|--------|
| Logistic Regression | 0.9998 | 0.9994 | 0.9998 | 0.9996 | 1.0000 | 1.0000 |
| CatBoost | 0.9999 | 0.9996 | 0.9998 | 0.9997 | 1.0000 | 1.0000 |
| **LightGBM** | **0.9998** | **0.9996** | **0.9996** | **0.9996** | **1.0000** | **1.0000** |
| Random Forest | 0.9998 | 0.9994 | 0.9996 | 0.9995 | 1.0000 | 1.0000 |

> **Note:** All models achieve near-perfect separation (ROC-AUC ≈ 1.0). This is a property of the **synthetic data**, not expected real-world performance.

### 4.4 Threshold Tuning

Thresholds tuned on **train out-of-fold predictions** (never on the validation set):

| Operating Point | Purpose | How it works |
|----------------|---------|--------------|
| **F1-max** | Automated tariff migration | Threshold maximizing F1 score |
| **Recall ≥ 0.95** | Outreach / review lists | Highest-precision threshold maintaining ≥95% recall (a missed SME costs more than a cheap call) |

---

## 5. Top Feature Drivers (SHAP & Gain Importance)

### 5.1 SHAP Global Mean |Impact| (Top Features)

| Feature | Lift over Consumer Baseline | Interpretation |
|---------|---------------------------|----------------|
| `b2b_mcc_share` | 8.37× | Strongest single signal: businesses spend heavily at B2B/wholesale MCCs |
| `b2b_amt_share` | 7.73× | Volume-weighted B2B concentration reinforces the count-based signal |
| `b2b_unique_merchants` | 6.70× | Breadth of B2B supplier base — consumers rarely touch B2B merchants |
| `recurring_share` | 4.92× | Businesses run payroll, subscriptions, lease payments; consumers almost never do |
| `recurring_capable_share` | 4.54× | Reflects merchant type quality, not just behavior |
| `amt_median` | 3.99× | Ticket size is the simplest and most durable signal |

### 5.2 Key Patterns

- **`merchant_hhi`** and **`merchant_top_ratio`** also strong — businesses spend with fewer, larger suppliers
- All three B2B features cluster together, measuring the same underlying phenomenon from different angles
- Timing features (`bizhours_share`, `evening_share`, `weekend_share`) provide incremental but weaker signal

---

## 6. Consumer Scoring & Hidden Entrepreneurs

### 6.1 Method

**Out-of-fold (OOF) scoring:** every consumer card is scored by a LightGBM model that did **not** see it during training → honest P(business).

### 6.2 Results

- **11 consumer cards** scored above the `P(business) ≥ 0.30` threshold
- These are the actionable **hidden-entrepreneur leads**
- On real data, the same pipeline would surface a far larger population

### 6.3 Lead Profiles

The 11 leads are exported to `hidden_entrepreneurs.csv` with the following key columns:

| Column | Description |
|--------|-------------|
| `card_number` | Card identifier |
| `oof_p` | Out-of-fold probability of being a business |
| `amt_sum` | Total spend |
| `amt_median` | Median transaction size |
| `online_share` | Online transaction share |
| `recurring_share` | Recurring transaction share |
| `b2b_mcc_share` | B2B MCC exposure |
| `b2b_amt_share` | B2B spend concentration |
| `b2b_unique_merchants` | Number of B2B merchants used |
| `recurring_capable_share` | Recurring-capable merchant share |
| `merchant_top_ratio` | Top-1 merchant concentration |

### 6.4 Top Leads (by P(business))

| # | Card Number | P(business) | Total Spend (KZT) | Median Tx | B2B Share |
|---|-------------|------------|-------------------|-----------|-----------|
| 1 | 5201491354169846 | **0.992** | 13,635,876 | 102,470 | 26.9% |
| 2 | 5531513098848970 | **0.968** | 21,549,560 | 170,060 | 74.3% |
| 3 | 5228591076618522 | **0.859** | 28,168,049 | 211,448 | 89.9% |
| 4 | 5228597629027905 | **0.803** | 16,073,145 | 250,188 | 68.3% |
| 5 | 5211556274611016 | **0.621** | 44,568,391 | 222,381 | 70.7% |

### 6.5 Business Product Recommendations

For identified hidden entrepreneurs, the following products are mapped to their behavioral profile:

- **POS acquiring** — high B2B and merchant volume
- **Working-capital loans** — large spend footprint
- **Payroll services** — high recurring share
- **Cash management** — frequent, high-value transactions
- **Bookkeeping / accounting tools** — diverse supplier base

---

## 7. Export & Data Access

### 7.1 Generated Artifacts

| File | Format | Content |
|------|--------|---------|
| `features_card_level.parquet` | Parquet | Full feature matrix (one row per card) |
| `hidden_entrepreneurs.csv` | CSV | 11 hidden-entrepreneur leads with profiles |
| `model_lgbm.joblib` | Joblib | Serialized LightGBM model + thresholds + feature list |
| `merchants_reference.parquet` | Parquet | Merchant metadata reference |
| `docs/mathematical_justification.md` | Markdown | Mathematical proof pack for feature logic, metrics, thresholds, and errors |
| `plots/` | PNG/CSV | Statistical-test outputs generated by `math_justification.py` |

### 7.2 Export Commands

```python
import pandas as pd

# Feature matrix to CSV / Excel
feat = pd.read_parquet("features_card_level.parquet")
feat.to_csv("features_card_level.csv", index=False)
feat.to_excel("features_card_level.xlsx", index=False)

# Raw consumer transactions
pd.read_parquet("consumer_cards_MDQ.parquet").to_csv("consumer_cards.csv", index=False)

# Merchant reference
pd.read_parquet("merchants_reference.parquet").to_csv("merchants_reference.csv", index=False)
```

**Multi-sheet Excel export:**

```python
with pd.ExcelWriter("mdq_review.xlsx", engine="openpyxl") as w:
    pd.read_parquet("features_card_level.parquet").to_excel(w, sheet_name="features", index=False)
    pd.read_csv("hidden_entrepreneurs.csv").to_excel(w, sheet_name="leads", index=False)
    pd.read_parquet("merchants_reference.parquet").to_excel(w, sheet_name="merchants", index=False)
```

---

## 8. Pipeline File Map

| File | Purpose |
|------|---------|
| `config.py` | Single source of truth: seed, file paths, B2B MCC list, thresholds |
| `features.py` | Aggregates transactions → one row per card (35+ features) |
| `mdq_utils.py` | Shared helpers: metrics, threshold tuning, OOF scoring, SHAP, plots |
| `train_eval.py` | Train + evaluate three models; persist the main model |
| `score_consumers.py` | Out-of-fold consumer scoring + hidden-entrepreneur leads + SHAP |
| `math_justification.py` | Defense support: statistical tests, metric curves, threshold/cost analysis, calibration |
| `docs/mathematical_justification.md` | Mathematical justification prepared for Miras's part of the defense |
| `build_notebook.py` | Regenerates `notebook.ipynb` from the modules |
| `notebook.ipynb` | Presentation deliverable (EDA → leakage → models → SHAP → leads) |

### How to Run

```bash
# 1. Create / activate environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run pipeline end-to-end
python features.py         # raw tx -> features_card_level.parquet  (~1 min)
python train_eval.py       # LogReg / LightGBM / RF -> model_lgbm.joblib
python score_consumers.py  # OOF scoring -> hidden_entrepreneurs.csv + SHAP

# 3. OR run the notebook
python -m nbconvert --to notebook --execute --inplace notebook.ipynb
```

---

## 9. Completion Status

### ✅ Completed

- [x] Data loading, schema audit, EDA
- [x] Leakage detection and exclusion (`card_tier`, `bank_name`, MCC `5122`, no one-hot of raw IDs)
- [x] 35 card-level features engineered and cached to Parquet
- [x] Three models trained + evaluated (LogReg, LightGBM, RandomForest)
- [x] Threshold tuning (F1-max + recall-leaning, on out-of-fold data)
- [x] SHAP global importance chart
- [x] SHAP per-card reason codes for all exported leads
- [x] Out-of-fold consumer scoring
- [x] 11 hidden-entrepreneur leads identified and exported to CSV
- [x] Business product recommendations mapped to lead profiles
- [x] Mathematical justification document for features, metrics, thresholds, and error analysis
- [x] Statistical support script with tests, plots, calibration, and threshold analysis
- [x] README with run instructions
- [x] Central config with B2B MCC list, paths, seed
- [x] `model_lgbm.joblib` saved
- [x] `score_consumers.py` exists as standalone file

### ⚠️ Partially Completed

- [ ] SHAP beeswarm plot — code is written but commented out (one line to uncomment)
- [ ] CatBoost — results exist in `archive/model_comparison.csv` but not integrated into main pipeline

### ❌ Not Yet Done

- [ ] Calibration integrated into the production scoring artifact (analysis exists in `math_justification.py`)
- [ ] Feature importance comparison across all three models
- [ ] Segment-level analysis (by bank, by MCC cluster)
- [ ] Time-based validation or cohort check

---

## 10. Recommended Next Steps

| Priority | Action | Why |
|----------|--------|-----|
| 🔴 **1** | Integrate calibrated probabilities into the saved scoring artifact | So the 0.30 threshold is interpretable as an actual probability in production |
| 🟡 **2** | Uncomment SHAP beeswarm plot in notebook cell 09 | More readable than bar chart for presentations |
| 🟡 **3** | Add segment-level validation by bank / MCC cluster | Shows model stability across subpopulations |
| 🟢 **4** | Export feature matrix to Excel for manual review of borderline cards (P = 0.30–0.50) | Business review of edge cases |
| 🟢 **5** | Integrate CatBoost into main pipeline | Results show it matches/beats LightGBM slightly |

---

## Key Caveat

> All models separate the two synthetic populations almost perfectly (ROC-AUC ≈ 1.0). This is a property of the **synthetic data**, not real-world performance. **Re-baseline on real labeled cards before trusting any threshold.** The consumer pool contains essentially **no embedded hidden entrepreneurs** (~13 of 80,000 score business-like); on real data the same pipeline would surface a far larger population.
