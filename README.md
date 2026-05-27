# Hidden-Entrepreneur Detection from Card Transactions

Classify each card as **consumer (0)** vs **business / hidden entrepreneur (1)** from
transaction behavior, then score the consumer pool to surface clients who behave like
businesses and should move to business products (POS acquiring, working-capital loans,
payroll, cash management, bookkeeping).

## How to run

```bash
# 1. create / activate the environment (Windows PowerShell shown)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. run the pipeline end-to-end (scripts)
python features.py         # raw tx -> features_card_level.parquet  (~1 min)
python train_eval.py       # LogReg / LightGBM / RF -> model_lgbm.joblib
python score_consumers.py  # OOF scoring -> hidden_entrepreneurs.csv + SHAP

# 3. OR run the notebook from scratch, top to bottom
python -m nbconvert --to notebook --execute --inplace notebook.ipynb
# (in an IDE, select the "Python (MDQ .venv)" kernel and Run All)
```

Everything is deterministic: a single `SEED = 42` in `config.py` drives the split,
cross-validation, and models.

## File map

| File | Purpose |
|------|---------|
| `config.py` | Single source of truth: seed, file paths, B2B MCC list, thresholds |
| `features.py` | Aggregates transactions -> one row per card (39 features) |
| `mdq_utils.py` | Shared helpers: metrics, threshold tuning, OOF scoring, SHAP, plots |
| `train_eval.py` | Train + evaluate the three models; persist the main model |
| `score_consumers.py` | Out-of-fold consumer scoring + hidden-entrepreneur leads + SHAP |
| `math_justification.py` | Statistical tests, metric/threshold analysis, calibration and plots for defense |
| `docs/mathematical_justification.md` | Mathematical justification: why features work, how metrics/thresholds/errors are computed |
| `build_notebook.py` | Regenerates `notebook.ipynb` from the modules |
| `notebook.ipynb` | Presentation deliverable (EDA -> leakage -> models -> SHAP -> leads) |
| `*.parquet`, `*.joblib`, `*.csv` | Generated artifacts |

## Modeling choices (short)

- **Unit = card.** Transactions are aggregated to one row per `card_number`; the split is
  on cards, so no card appears in both train and validation.
- **Split.** Stratified 80/20 random split. A *chronological* split is **not** needed here:
  the label is an intrinsic card type observed over a fixed 6-month window, not a future
  event. (It *would* be needed to predict "becomes a business next quarter.")
- **Leakage handling.** `card_tier` (100% `"Business"` for businesses) and `bank_name`
  (no signal) are dropped. Raw `mcc`/`merchant_id` are **not** one-hot encoded — their
  class-exclusive values are coverage artifacts. The B2B basket in `config.py` is
  domain-curated; `5122` was removed because it is business-exclusive in this sample.
- **Models.** Logistic Regression (baseline) ≈ LightGBM (main) ≈ Random Forest, all
  ROC-AUC ≈ 1.0. LightGBM is kept for SHAP per-customer reason codes.
- **Thresholds.** Tuned on **train out-of-fold** scores (never the validation set).
  Recall-leaning point for outreach lists (a missed SME costs more than a cheap call);
  F1-max for automated tariff migration.
- **Mathematical defense.** `docs/mathematical_justification.md` explains the formulas
  behind feature groups, ROC-AUC / PR-AUC / F1, Bayes threshold choice, expected loss,
  calibration, and error analysis. `math_justification.py` generates supporting tests
  and plots under `plots/`.

## Key result & caveat

All models separate the two synthetic populations almost perfectly (ROC-AUC ≈ 1.0). This is
a property of the **synthetic data**, not real-world performance — re-baseline on real
labeled cards before trusting any threshold. The consumer pool contains essentially **no
embedded hidden entrepreneurs** (~13 of 80,000 score business-like); on real data the same
pipeline would surface a far larger population.
