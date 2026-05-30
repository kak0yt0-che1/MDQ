# Jury Q&A — Hidden-Entrepreneur Detection

Quick, defensible answers to the 15 questions a Mastercard Data Quest jury is most likely to ask. Optimised for ML quality, leakage, metrics, business value, deployment.

---

### 1. The model hits ROC-AUC = 1.0. Is it overfit?

No. Three pieces of evidence on top of the headline number:
- **Honest split.** Stratified 80/20 on cards; threshold tuned on train **out-of-fold**, validation set never touched.
- **Bootstrap CI.** 1000 resamples of validation → ROC-AUC 95 % CI [1.0000, 1.0000]. Not a lucky split.
- **Baseline match.** Logistic Regression hits the same score as LightGBM — if it were tree overfit, the linear baseline would lag.

The 1.0 reflects clean synthetic class separation. On real data we expect 0.85–0.95.

---

### 2. How do you know there is no label leakage?

Active defensive engineering, verified twice:
- Dropped `card_tier` (100 % "Business" for businesses — perfect leak) and `bank_name` (no signal).
- Raw `mcc` and `merchant_id` are **never** one-hot encoded — class-exclusive values are a coverage artifact, not behavior.
- B2B MCC basket is curated from MCC **semantics**, not from data. MCC 5122 was explicitly excluded because it is business-exclusive in this sample.
- **Sanity re-check:** max univariate AUC across the final 35 features = 0.9948 < 1.0. No single feature is a perfect separator.

---

### 3. Why predict at the card level and not at the transaction level?

Because the bank's decision unit is the card. A transaction-level model would force you to aggregate predictions back to the card anyway, with no signal gain and a leakage risk (the same card straddling train and test). We aggregate first to one row per `card_number` (105 000 rows total), then the random split is automatically a card-level split — no card appears in both train and validation.

---

### 4. Why a random split instead of chronological?

The target is an **intrinsic** card type observed over a fixed 6-month window, not a future event. Every card spans the same window; `recency_days` is measured from one global cutoff. A chronological split would only be needed if the question were "will this card become a business next quarter".

---

### 5. Why LightGBM as the main model?

It ties LogReg and RandomForest on the raw metrics on this data, but it gives:
- **SHAP TreeExplainer** for clean per-card reason codes (drop-in for an RM call script).
- Native handling of unscaled numeric features.
- Strong out-of-box performance on tabular data (Grinsztajn et al. 2022, Shwartz-Ziv & Armon 2022).

LogReg is kept as the baseline: its perfect score confirms the signal is genuine and near-linear, not tree overfitting.

---

### 6. Why are the SHAP top drivers different from the "B2B 8x" lift numbers in the deck?

Two different metrics, both reported:
- **Class-mean ratio** (business mean / consumer mean) — the raw separability of one feature, ignoring the model. Top: `b2b_mcc_share` = 9.15×, `b2b_amt_share` = 8.06×.
- **SHAP global mean |impact|** — how much the feature actually moves the LightGBM output, given access to all the others. Top: `evening_share` = 2.66, `tokenized_share` = 1.77, `online_share` = 1.37, `b2b_unique_merchants` = 1.37.

They disagree on purpose. Once the model knows `b2b_unique_merchants` plus temporal pattern plus tokenization, the per-transaction B2B share becomes redundant. That's healthy — it shows the model uses multiple independent behavioral pillars.

---

### 7. Why two thresholds (F1-max and Outreach) instead of just 0.5?

The default 0.5 is arbitrary in cost-asymmetric problems. We expose two operating points tied to two business workflows:
- **F1-max** (τ ≈ 0.32): strict, for automated tariff migration where a false move is awkward.
- **Outreach** (τ ≈ 2 × 10⁻⁶, smallest threshold with precision ≥ 0.5): widest viable RM call list. A missed SME costs the bank multi-year recurring revenue; one cheap call costs nothing.

Both are tuned on **train OOF**, not on validation.

---

### 8. The dataset is moderately imbalanced (24 % positive). Why not just report accuracy?

Accuracy is misleading under imbalance: a "predict everyone consumer" classifier scores 76 % accuracy and catches zero entrepreneurs. We report:
- **ROC-AUC** (competition primary) — threshold-independent ranking quality.
- **PR-AUC** — the sharper view on the positive class. Baseline = positive prevalence = 0.20.
- **Confusion matrix + precision / recall / F1** at both operating points.

---

### 9. What does the confusion matrix actually mean in business terms?

At F1-max on validation: TN = 16 000, FP = 0, FN = 1, TP = 4 999.
- **FP** = consumer falsely flagged as business → one cheap manual review, low cost.
- **FN** = business that goes unflagged → wrong tariff + missed acquiring + no SME credit = multi-year recurring revenue lost + compliance exposure.
- The cost asymmetry `C_FN ≫ C_FP` is exactly why the operational threshold leans toward recall, not toward 0.5.

---

### 10. Are the model probabilities calibrated?

Yes — ECE = 0.0001 on validation (Appendix A3 of the notebook). Platt overlay is fit on train OOF and shown as a deployable scaffold for when real data needs it. On the current synthetic data the overlay does not improve calibration because raw scores already concentrate near 0 and 1.

---

### 11. How were the 11 hidden entrepreneurs validated?

They are consumers (`label == 0`) with out-of-fold `P(business) ≥ 0.30`. "Out-of-fold" means each one was scored by a LightGBM that did not train on it, so the score is honest. Their median profile (170 k KZT median ticket, 89 % B2B amount share, 85 % online share) sits right next to the business population's median — they behave like SMEs in every feature family that the model uses. Each lead in `hidden_entrepreneurs.csv` carries a `reason_codes` column with its top SHAP drivers (sign and magnitude), so an RM can call with specifics.

---

### 12. How would the bank actually use this in production?

Monthly batch:
1. Pull the last 6 months of transactions per active card.
2. Run `build_card_features` → 35-feature card table.
3. Score with the full-data-refit LightGBM.
4. Route:
   - `score ≥ τ_F1-max` → automated tariff migration trigger.
   - `0.30 ≤ score < τ_F1-max` → RM call list with SHAP reason codes attached.
   - `score < 0.30` → no action this cycle.

Pilot in Q1 (500 leads, measure conversion to each SME product), recalibrate τ on real cost data in Q2, scale to automated migration in Q3 once real-data calibration holds.

---

### 13. What if the data drifts in production?

Three monitors and one trigger:
- **Feature drift:** per-feature PSI against the training distribution, monthly.
- **Score drift:** distribution of `P(business)` against the training OOF distribution.
- **Performance drift:** AUC on a fresh labeled sample (from compliance reviews) at least quarterly.

Trigger retraining (rolling 6-month window) when PSI > 0.2 on a top-5 SHAP feature or when fresh-sample AUC drops more than 0.05 from the rolling average.

A champion / challenger setup (LightGBM champion, RandomForest challenger) catches model-specific overfit to synthetic shape early on real data.

---

### 14. How reproducible is the solution?

End-to-end deterministic from one seeded notebook:
- Single `SEED = 42` drives the train/val split, 5-fold CV, every model, the SHAP sampling.
- One self-contained notebook (no module imports) regenerates every artifact: `features_card_level.parquet`, `submission.csv`, `hidden_entrepreneurs.csv`, all SHAP and metric plots.
- Run from a clean environment: `pip install -r requirements.txt && jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`.

---

### 15. What are the most honest limitations?

- **AUC = 1.0 is a synthetic artifact.** Real-data AUC realistically lands in 0.85–0.95. The headline number does not transfer; the *framework* (features + OOF scoring + reason codes) does.
- **No real edge cases.** Freelancers, mixed-use cards, seasonal traders are absent from this generator — the real population will be messier and the threshold will need recalibration on the first month of labeled real data.
- **Small lead count (11).** Direct consequence of the synthetic generator barely overlapping. On real data we expect orders of magnitude more.
- **No causal claim.** The model identifies *behavioral* business activity. Whether each surfaced card is legally a business is a compliance question for the RM, not for the model.
