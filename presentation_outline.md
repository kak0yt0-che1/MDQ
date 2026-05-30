# Presentation Outline — Hidden-Entrepreneur Detection

Mastercard Data Quest · slide-ready, business-first.

---

## 1. Executive Summary

**Problem.** Some bank clients run a small business but pay through a personal card. The bank misses revenue (wrong tariff, no acquiring, no SME credit) and carries regulatory risk.

**Solution.** A per-card classifier built only from transactional behavior. For every `card_number` it outputs `P(business) ∈ [0, 1]`. Cards labeled *consumer* with a high score are **hidden entrepreneurs** — the actionable target list.

**Why it matters.**
- New revenue: tariff migration, POS acquiring, working-capital loans, payroll projects, FX, bookkeeping.
- Lower compliance risk: surface business activity sitting on consumer infrastructure.
- Reusable framework: feature engineering + OOF scoring + per-card explanations transfer to real data.

**Headline numbers (synthetic data, 105 000 cards, 23.8 % positive).**
- ROC-AUC = 1.0000 (bootstrap 95 % CI [1.0000, 1.0000])
- PR-AUC = 1.0000 (baseline 0.20)
- 11 actionable hidden-entrepreneur leads found in the consumer pool (top score = 0.992)

---

## 2. Problem Framing

**Hidden entrepreneur** = a `card_number` that the bank holds as *consumer* (label = 0) but whose transactional behavior matches the *business* population (label = 1).

**Why personal cards used for business matter to the bank:**
- Foregone fee income (consumer tariff vs SME tariff).
- Missed cross-sell window for POS acquiring, payroll, working capital, cash management.
- Reduced operational visibility into actual SME activity in the portfolio.
- Compliance risk: undisclosed business activity on consumer products.

**Monetization vectors per surfaced lead:**
- Tariff migration to an SME plan.
- POS acquiring sale.
- Working-capital line.
- Payroll / salary-project setup.
- FX margin on cross-border activity.
- Cash management + bookkeeping cross-sell.

---

## 3. Data and Approach

**Inputs.**
- `consumer_cards_MDQ.parquet` — 9.8 M transactions, 80 000 cards.
- `business_cards_MDQ.parquet` — 3.0 M transactions, 25 000 cards.
- `merchants_reference.parquet` — 2 165 merchants with `recurring_capable` flag.
- 6-month window, Kazakhstan local time.

**Approach.**
- **Unit of analysis = card.** Everything is aggregated per `card_number`. No transaction-level prediction.
- 35 behavioural features in 5 families (volume, diversity, B2B exposure, temporal pattern, burstiness).
- Stratified 80/20 split on cards.
- LightGBM as the main model, LogReg and RandomForest as comparison.
- Honest out-of-fold scoring of every card (5-fold) → the score that goes into the submission.
- SHAP for global drivers and a one-line reason code per surfaced lead.

**Most informative signals (SHAP top 4):** `evening_share`, `tokenized_share`, `online_share`, `b2b_unique_merchants`. The model uses several independent behavioural pillars, not a single shortcut.

---

## 4. Feature Engineering Rationale

35 card-level features. Five groups, each tied to one piece of business intuition.

### 4.1 Volume and dispersion
`tx_count`, `amt_sum`, `amt_mean`, `amt_median`, `amt_std`, `amt_min`, `amt_max`, `amt_cv`

- **Intuition.** A business card processes larger and more variable tickets than a household card (procurement, payroll, wholesale orders).
- **Business meaning.** Higher `amt_median` and `amt_sum` are an immediate proxy for SME activity.
- **Signal strength.** `amt_median` business/consumer mean ratio ≈ 3.99×.

### 4.2 Merchant diversity and concentration
`n_unique_merchants`, `n_unique_mcc`, `n_unique_countries`, `merchant_hhi`, `merchant_top_ratio`, `merchant_entropy`, `merchants_per_tx`

- **Intuition.** Consumers hit a small set of repeat shops; businesses route procurement through many specialised suppliers.
- **Business meaning.** Low HHI / high entropy = diversified supply base = procurement behavior.
- **Math.** Dual measures of the same shape — HHI = Σpₘ², Shannon entropy = −Σpₘ ln pₘ.

### 4.3 B2B exposure (the headline pillar)
`b2b_mcc_share`, `b2b_amt_share`, `b2b_unique_merchants`

- **Intuition.** Direct dealings with wholesale / professional services / logistics MCCs are a near-tautological SME footprint.
- **Business meaning.** The strongest raw class separator in the entire feature table.
- **How we built it.** A 43-MCC basket curated from MCC semantics (wholesale 5044-5199, professional services 7311-8931, freight 4214-4816). MCC 5122 was deliberately excluded — it appears only in the business pool and would leak the label.
- **Signal strength.** `b2b_mcc_share` b/c ratio = 9.15×; `b2b_unique_merchants` SHAP impact = 1.37 (top-4 in the model).

### 4.4 Temporal and recurring patterns
`online_share`, `weekend_share`, `evening_share`, `bizhours_share`, `recurring_share`, `recurring_capable_share`, `tokenized_share`, `foreign_share`, `active_days`, `active_weeks`, `span_days`, `tx_per_active_day`, `recency_days`, `monthly_spend_cv`

- **Intuition.** Businesses spend in business hours and run automated recurring payments (rent, SaaS, payroll). Consumers spend evenings and weekends and rarely have a recurring B2B cadence.
- **Business meaning.** Schedule pattern + automation cadence is the most universal SME tell.
- **Signal strength.** `evening_share` SHAP impact = 2.66 (top-1), `tokenized_share` = 1.77 (top-2), `online_share` = 1.37.

### 4.5 Burstiness
`gap_mean`, `gap_std`, `burstiness`

- **Intuition.** Goh-Barabási burstiness `B = (σ − μ) / (σ + μ)` ∈ [-1, 1]: B > 0 = bursty consumer, B < 0 = scheduled business.
- **Business meaning.** Operational consistency captured in one number.

### What we deliberately did NOT use
- `card_tier` — perfect leak (100 % "Business" for businesses).
- `bank_name` — no signal.
- Raw `mcc` / `merchant_id` one-hot — their class-exclusive values are coverage artifacts of the synthetic generator, not behavioral signal.

---

## 5. Modeling and Validation

**Baseline.** Logistic Regression with log1p + standard scaling inside a `Pipeline` (so the transformer fits on train only).

**Final model.** LightGBM, 400 trees, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, gain importance. Kept because it ties LogReg on raw metrics but gives clean SHAP reason codes per card — essential for RM outreach.

**Comparison.** RandomForest with similar regularization.

**Validation.**
- Stratified 80/20 random split on cards (one row per card_number, so a row-level split is automatically card-level).
- A chronological split is *not* needed — the label is an intrinsic card type observed over a fixed 6-month window, not a future event.
- 5-fold stratified out-of-fold scoring on the full labeled set for the leads file and the submission CSV.

**Threshold tuning.** Always on train OOF, never on the validation set. Two operating points:

| Point | Definition | Business use |
|-------|------------|---------------|
| F1-max | maximizes F1 on train OOF | automated tariff migration (strict) |
| Outreach | smallest threshold whose precision ≥ 0.50 on train OOF | RM call list (largest viable lead pool — missed SME costs more than one cheap call) |

**Confusion matrix on validation (F1-max, τ ≈ 0.32):**

|  | predicted consumer | predicted business |
|---|---|---|
| actual consumer | 16 000 | 0 |
| actual business | 1 | 4 999 |

Reads as: zero false alarms, one missed SME out of 5 000.

---

## 6. Results

| Model | ROC-AUC | PR-AUC | F1 @ F1-max |
|---|---:|---:|---:|
| Logistic Regression (baseline) | 1.0000 | 1.0000 | 0.9996 |
| **LightGBM (main)** | **1.0000** | **1.0000** | **0.9999** |
| Random Forest | 1.0000 | 1.0000 | 0.9995 |

**Bootstrap 95 % CI** (B = 1000 resamples of validation): ROC-AUC [1.0000, 1.0000], PR-AUC [1.0000, 1.0000]. Not a lucky split.

**Why this is evidence the model works, not a leak.**
- No single feature reaches AUC = 1.0 alone (max univariate = 0.9948).
- The linear baseline matches the boosted model — signal is genuine and near-linear, not tree overfitting.
- Class-exclusive identifiers (`card_tier`, `bank_name`, business-only MCCs) are explicitly excluded; B2B basket re-verified to contain zero business-exclusive codes.

**FP / FN tradeoff.** Both errors are tiny on this synthetic data. On real bank data the design choice is asymmetric: a missed SME = ongoing revenue loss + compliance exposure; a false positive = one cheap manual review. That is why the operational threshold leans toward recall.

---

## 7. Explainability and Trust

**Global drivers (SHAP mean |impact|, top 4 of 15 shown in the deck).**
1. `evening_share` — 2.66
2. `tokenized_share` — 1.77
3. `online_share` — 1.37
4. `b2b_unique_merchants` — 1.37

**Per-card reason codes.** Every entry in `hidden_entrepreneurs.csv` has a `reason_codes` column listing the top SHAP drivers with sign and magnitude — drop-in copy for an RM call script.

**Leakage controls (defensive engineering).**
- `card_tier` dropped (perfect separator).
- `bank_name` dropped (no signal).
- Raw `mcc` and `merchant_id` never one-hot encoded.
- B2B MCC basket curated from semantics; MCC 5122 excluded because it is business-exclusive in this sample.
- Univariate AUC re-check confirms no single feature reaches 1.0 after exclusions.

**Why a bank can trust the model.**
- Card-level prediction unit matches the business decision unit.
- Reproducible end-to-end from one seeded notebook (`SEED = 42`).
- Threshold tuned on train OOF; validation set never touched.
- Probability calibration verified: ECE = 0.0001 on validation, no overlay needed.
- Bayesian expected-loss curve confirms the recall-leaning operating point under a 1:10 FP:FN cost ratio (Appendix A4).

**Limitations specific to synthetic data.**
- ROC-AUC = 1.0 reflects clean generator separation. Real-world AUC realistically lands in 0.85-0.95.
- The synthetic consumer pool contains essentially zero embedded entrepreneurs (only 11 cross the 0.30 cutoff). Real data would surface a much larger population.

---

## 8. Deployment Recommendations

**Operating model — monthly batch scoring.**

1. Pull the last 6 months of transactions for every active card.
2. Run `build_card_features` to produce the 35-feature card table.
3. Score with the refit LightGBM (full-data refit, not the train-only fit).
4. Route by threshold:
   - `score ≥ τ_F1-max` → automated tariff migration trigger (low-friction, strict).
   - `0.30 ≤ score < τ_F1-max` → RM call list with per-card SHAP reason codes attached.
   - `score < 0.30` → no action this cycle.

**Product playbook per surfaced lead.**

| Signal in the lead | Bank action | Revenue lever |
|--------------------|-------------|----------------|
| High online share + merchant concentration | POS acquiring offer | acquiring fees, terminal lease |
| Large recurring B2B outflows | Working-capital loan | interest, origination fee |
| High recurring share to many counterparties | Payroll product, salary project | account fees, cross-sell |
| Median ticket > 100 k KZT, high `amt_sum` | Tariff migration to SME plan | account-fee uplift |
| Broad B2B supplier base | Cash management + bookkeeping | per-transaction + SaaS fees |
| `foreign_share` > 0.20 | FX / cross-border product | FX margin |

**Pilot design.**
- Quarter 1: score the full portfolio, sample 500 leads above the outreach threshold, hand to RMs with reason codes, track conversion to each SME product.
- Quarter 2: relabel the audited cards, retrain, recalibrate threshold against observed FP/FN cost on real data.
- Quarter 3: scale to full automated migration on the F1-max threshold once real-data calibration holds.

---

## 9. Limitations and Next Steps

**Synthetic-data caveats.**
- Per-class generators barely overlap, so the headline 1.0 AUC will compress on real data.
- No real edge cases in the data: freelancers, mixed-use cards, seasonal traders. Real population will be messier.

**Operational risks on the way to production.**
- Dataset shift in MCC mix and merchant population — needs monitoring on feature distributions, not just on AUC.
- Threshold recalibration after the first labeled month: recompute `τ_F1-max` and `τ_outreach` on real OOF scores.
- Drift in the B2B basket — review annually with payments operations.

**Next steps.**
1. Champion / challenger: keep the LightGBM as champion, run RandomForest as challenger to detect overfit to the synthetic shape once on real data.
2. Probability calibration on real data (Platt or isotonic). Already scaffolded in Appendix A3.
3. Active monitoring: per-feature PSI vs training distribution, monthly AUC on a fresh labeled sample.
4. Quarterly retraining with the latest 6-month window.
5. (Extra-mile bonus) thin internal web service: card_number in, score + reason codes out, for RM tooling.

---

## 10. Slide-by-Slide Outline (11 slides)

### Slide 1 · Title
- "Hidden Entrepreneurs in the Consumer Card Portfolio"
- Team, date, Mastercard Data Quest
- **Visual:** one-line tagline + bank-revenue ladder graphic.
- **Speaker note:** open with the asymmetric cost — one missed SME is multi-year recurring revenue lost.

### Slide 2 · The Problem
- Consumer pool hides active small businesses → revenue leakage + compliance risk.
- Definition of "hidden entrepreneur" (card filed as consumer, behaves as business).
- Concrete examples: freelance specialist, side-business owner, mixed-use card.
- **Visual:** Venn — declared business / declared consumer / behavioral business.
- **Speaker note:** anchor in money — a single migrated SME = recurring acquiring + tariff uplift.

### Slide 3 · Approach in One Picture
- Unit = card. 35 behavioral features. LightGBM main, LogReg / RF as comparison.
- 5-fold OOF scoring for every card → honest score per `card_number`.
- **Visual:** pipeline diagram (transactions → card features → model → score → action).
- **Speaker note:** stress the one-row-per-card unit — judges check this explicitly.

### Slide 4 · Feature Engineering
- Five families: volume · diversity · B2B exposure · temporal/recurring · burstiness.
- Top business signals: B2B supplier breadth, evening vs business-hours mix, online + tokenized cadence, recurring share, merchant concentration.
- **Visual:** the 1.8.a / 1.8.b twin tables from the math justification (b/c ratio vs SHAP impact).
- **Speaker note:** explicitly point out that the two rankings disagree on purpose — model uses several independent pillars.

### Slide 5 · Leakage Discipline
- Dropped `card_tier` (100 % "Business" for businesses), `bank_name` (no signal).
- Raw `mcc` / `merchant_id` never one-hot; B2B basket curated from semantics; MCC 5122 excluded as business-exclusive.
- Univariate AUC sanity: max single-feature AUC = 0.9948 < 1.0.
- **Visual:** the univariate-AUC bar chart from §6 of the notebook.
- **Speaker note:** this is the first defensive question judges will ask — answer it before they do.

### Slide 6 · Validation Setup
- Stratified 80/20 split on cards (no leakage since one row per card).
- Why random, not chronological — label is intrinsic, not a future event.
- Threshold tuning on train OOF only; validation set never touched.
- **Visual:** simple train/val split diagram with K-fold inside train.
- **Speaker note:** explain F1-max vs Outreach as automated-vs-call-list, not as two arbitrary thresholds.

### Slide 7 · Results & Confusion Matrix
- ROC-AUC = 1.0000 (bootstrap 95 % CI [1.0000, 1.0000]), PR-AUC = 1.0000.
- Confusion matrix at F1-max: 16 000 / 0 / 1 / 4 999.
- All three models agree → near-linear, genuine signal.
- **Visual:** confusion matrix heatmap + ROC + PR curves side by side.
- **Speaker note:** acknowledge the 1.0 number, immediately frame it as synthetic-data property, pivot to the framework.

### Slide 8 · Explainability
- SHAP global drivers (top 8).
- One per-card reason-code example from `hidden_entrepreneurs.csv`.
- ECE = 0.0001 → probabilities are already calibrated; Platt overlay ready for real data.
- **Visual:** SHAP beeswarm + a single reason-code row from the leads file.
- **Speaker note:** stress that every lead gets a "why" line ready for an RM to use on a call.

### Slide 9 · Hidden Entrepreneurs Found
- 11 cards in the consumer pool with `P(business) ≥ 0.30`, top score 0.992.
- Profile vs typical consumer: median ticket ~170 k KZT, ~89 % B2B amount share, 85 % online share.
- Output file: `hidden_entrepreneurs.csv` (one card per row + reason codes).
- **Visual:** the leads-profile table (consumer median vs lead median vs business median).
- **Speaker note:** the small count is a synthetic artifact; on real data the same pipeline surfaces thousands.

### Slide 10 · Business Actions and Revenue
- Product map (POS acquiring · working-capital loan · payroll · cash management · FX · bookkeeping).
- Operational split: automated migration above F1-max, RM call list above Outreach.
- Quarterly pilot: 500 leads → measure conversion to each product → refit threshold on real cost data.
- **Visual:** the product-mapping table from §8 (Signal → Action → Revenue lever).
- **Speaker note:** translate one detected entrepreneur into recurring annual revenue (tariff + acquiring + lending).

### Slide 11 · Limitations and Next Steps
- Synthetic AUC = 1.0 → expect 0.85-0.95 on real data; recalibrate on first labeled month.
- Monitor feature drift (PSI) and threshold stability; retrain quarterly.
- Champion / challenger setup; optional internal web service for RM tooling.
- **Visual:** roadmap timeline (pilot → calibrate → automate → monitor).
- **Speaker note:** end on the framework, not the headline number — the framework is what transfers.

---

## Appendix · Speaker cues to keep in mind

- Always say **card-level**, never **transaction-level**.
- Always say **out-of-fold** when you mention the score, never **predicted on train**.
- If asked about AUC = 1.0, **lead with the bootstrap CI and the linear-vs-boosted match**, then mention synthetic separation.
- If asked about "why these MCCs", say **curated from MCC semantics, not from the data — 5122 was explicitly excluded to prove the point**.
- If asked about deployment, anchor on the **monthly batch + RM tooling + pilot loop**.
