"""Generate notebook.ipynb (the presentation deliverable) via nbformat.
Cells reuse the importable modules: config.py, features.py, train_eval.py, mdq_utils.py.
Run:    .venv/Scripts/python.exe build_notebook.py
Execute: .venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace notebook.ipynb
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(s): cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

md(r"""
# Hidden-Entrepreneur Detection from Consumer Card Transactions

**Goal.** Classify each card as *consumer* (0) vs *business / hidden entrepreneur* (1) from
transaction behavior, then score the consumer pool to surface clients who behave like
businesses and should move to business products (POS acquiring, working-capital loans,
payroll, cash management, bookkeeping).

**Framing.** Train on the two labeled pools (`business=1`, `consumer=0`) aggregated to
**one row per card**, then score consumers **out-of-fold** to find hidden entrepreneurs.
The consumer pool is a *contaminated* negative class (it may contain the very positives we
hunt), so ranking & precision@top-K matter more than headline accuracy.

### Headline findings (read first)
1. **No single-feature label leak** after dropping `card_tier` (business is 100% `"Business"`).
   Raw `mcc`/`merchant_id` are **not** one-hot encoded (class-exclusive values are coverage
   artifacts); the B2B basket is domain-curated and `5122` was removed (business-exclusive here).
2. **All models reach ROC-AUC ≈ 1.00** (even Logistic Regression). This is a property of the
   **synthetic data** — the per-class behavioral generators barely overlap. *Real-world AUC
   would be lower.* The durable value is the feature interpretation + the ranking method.
3. **The synthetic consumer pool contains essentially no embedded hidden entrepreneurs:**
   out-of-fold `P(business)` is ~0 for 99.9% of consumers; ~11 score business-like. Those few
   form the actionable list. On *real* bank data this same pipeline would surface far more.

*Reproducibility:* a single `SEED` in `config.py` drives split / CV / models. Heavy logic lives
in `features.py`, `train_eval.py`, `mdq_utils.py` so cells stay short. See `README.md` to run.
""")

# ---------------------------------------------------------------- 00
md("## 00 · Setup & reproducibility")
code(r"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, gc
import shap
from IPython.display import display
from sklearn.model_selection import train_test_split

from config import (SEED, LEAD_THR, TEST_SIZE, ID_COL, LABEL_COL, B2B_MCC,
                    CONSUMER_FILE, BUSINESS_FILE, MERCHANT_FILE, FEAT_FILE)
from features import load_labeled, build_card_features, LEAKY_OR_EXCLUDED
from train_eval import make_logreg, make_lgbm, make_rf
from mdq_utils import (ranking_metrics, metrics_at_threshold, tune_thresholds,
                       oof_proba, shap_positive_values, plot_confusion)

np.random.seed(SEED)
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
plt.rcParams["figure.figsize"] = (8, 4); plt.rcParams["figure.dpi"] = 110
print("setup ok | SEED =", SEED)
""")

# ---------------------------------------------------------------- 01
md("## 01 · Load & schema audit\nConfirm the real column names/dtypes (snake_case) before any feature code.")
code(r"""
consumer  = pd.read_parquet(CONSUMER_FILE)
business  = pd.read_parquet(BUSINESS_FILE)
merchants = pd.read_parquet(MERCHANT_FILE)
print("consumer", consumer.shape, "| business", business.shape, "| merchants", merchants.shape)
display(consumer.dtypes.to_frame("dtype"))
display(consumer.head(3))
""")

# ---------------------------------------------------------------- 02
md("## 02 · EDA\nQuality, scale, and the behavioral contrast between the two classes.")
code(r"""
df = load_labeled()   # consumer+business with label; card_number is globally unique
print("rows", f"{len(df):,}", "| cards", f"{df[ID_COL].nunique():,}")
print("missing values:", int(df.isna().sum().sum()), "| duplicate rows:", int(df.duplicated().sum()))
print("date range:", df['transaction_date'].min(), "->", df['transaction_date'].max())
print("unique cards  consumer:", f"{df.loc[df.label==0, ID_COL].nunique():,}",
      " business:", f"{df.loc[df.label==1, ID_COL].nunique():,}")
print("\namount (KZT) by class:")
display(df.groupby(LABEL_COL)["transaction_amount_kzt"].describe(percentiles=[.5,.9,.99]).round(0))
""")
code(r"""
# behavioral share contrast (tx-level)
ts = pd.to_datetime(df["transaction_timestamp"])
contrast_src = df.assign(
    online=(df.channel=="online"), wknd=(ts.dt.dayofweek>=5), eve=ts.dt.hour.between(18,23))
rows = []
for col in ["online", "is_recurring", "tokenized", "wknd", "eve"]:
    s = contrast_src.groupby(LABEL_COL)[col].mean()
    rows.append((col, s[0], s[1]))
contrast = pd.DataFrame(rows, columns=["signal","consumer","business"]).set_index("signal")
display(contrast.round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for lab, name, c in [(0,"consumer","#4C72B0"), (1,"business","#DD8452")]:
    ax[0].hist(np.log10(df.loc[df.label==lab,"transaction_amount_kzt"].clip(lower=1)),
               bins=60, alpha=.55, label=name, color=c, density=True)
ax[0].set_title("log10 transaction amount (KZT)"); ax[0].set_xlabel("log10 KZT"); ax[0].legend()
contrast.plot.bar(ax=ax[1], color=["#4C72B0","#DD8452"]); ax[1].set_title("behavioral shares by class")
ax[1].set_ylabel("share"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 03
md("## 03 · Leakage screen\nSynthetic data often hides perfect separators. Policy: **flag and exclude**.")
code(r"""
print("card_tier by class (perfect giveaway -> dropped):")
display(pd.crosstab(df.label, df.card_tier))

con_m, biz_m = df[df.label==0], df[df.label==1]
for col in ["mcc","merchant_id"]:
    cset, bset = set(con_m[col].unique()), set(biz_m[col].unique())
    only_b = bset - cset
    vol = biz_m[col].isin(only_b).mean()
    print(f"{col}: biz-only={len(only_b)}  con-only={len(cset-bset)}  shared={len(cset & bset)}  "
          f"| biz volume in biz-only {col}: {vol:.3%}  (artifact, so {col} is NOT one-hot encoded)")
print("\ndropped entirely:", LEAKY_OR_EXCLUDED, "| B2B basket size:", len(B2B_MCC), "(5122 removed: biz-exclusive)")
""")

# ---------------------------------------------------------------- 04
md(r"""
## 04 · Feature engineering (card level)
`build_card_features` aggregates transactions → one row per card: amount stats, counts,
diversity/entropy, merchant concentration (HHI / top-ratio), recurring/tokenized/channel/
temporal shares, curated B2B exposure, B2B supplier breadth, monthly stability, burstiness,
recency. Redundant duplicates (mcc-level concentration, `night_share`, `gap_cv`, …) were
pruned in the audit. See `features.py`.
""")
code(r"""
import os
feat = pd.read_parquet(FEAT_FILE) if os.path.exists(FEAT_FILE) else build_card_features(df)
FEATURES = [c for c in feat.columns if c not in (ID_COL, LABEL_COL)]
print("feature matrix:", feat.shape, "| n_features:", len(FEATURES),
      "| positive_rate:", round(feat[LABEL_COL].mean(), 4))
del df; gc.collect()                 # free the 12.8M-row frame; not needed past here
display(feat[FEATURES].head(3))
""")
code(r"""
# class-mean contrast (sanity: do features separate as expected?)
g = feat.groupby(LABEL_COL)[FEATURES].mean().T
g.columns = ["consumer","business"]; g["b/c_ratio"] = (g.business/g.consumer.replace(0,np.nan)).round(2)
display(g.round(3))
""")
md(r"""
### 04b · Univariate leakage re-check
No engineered feature should be a perfect (AUC=1.0) separator on its own — that would signal
residual leakage. The high-but-<1.0 AUCs below are legitimate behavioral signals; their
*cleanliness* is the synthetic-data caveat, not a leak.
""")
code(r"""
from sklearn.metrics import roc_auc_score
y_all = feat[LABEL_COL].values
uni = pd.Series({c: max((a:=roc_auc_score(y_all, feat[c])), 1-a) for c in FEATURES}).sort_values(ascending=False)
print("max univariate AUC:", round(uni.max(), 4), "(1.0 would mean a hard leak)")
uni.head(12).iloc[::-1].plot.barh(color="#55A868"); plt.title("top-12 single-feature AUC")
plt.xlabel("AUC"); plt.xlim(0.5, 1.0); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 05
md("## 05 · Stratified train / validation split\nSplit is on **cards**, so no card appears in both sets.")
code(r"""
X, y = feat[FEATURES], feat[LABEL_COL].values
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
print("train", Xtr.shape, "| val", Xva.shape, "| val positive_rate", round(yva.mean(), 4))
""")

# ---------------------------------------------------------------- 06/07
md(r"""
## 06–07 · Models: LogReg (baseline) · LightGBM (main) · RandomForest (compare)
Preprocessing for LogReg (log1p + scaling) is fit **inside a Pipeline on train only**.
""")
code(r"""
models = {"LogReg": make_logreg(FEATURES), "LightGBM": make_lgbm(), "RandomForest": make_rf()}
fitted, rows = {}, []
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict_proba(Xva)[:,1]; fitted[name] = m
    rows.append({"model": name, **ranking_metrics(yva, p)})
display(pd.DataFrame(rows).set_index("model").round(4))
print("All models saturate (~1.0): consumer vs business is near-linearly separable in this "
      "synthetic data. LogReg's perfect score confirms the signal is clean & near-linear; we "
      "keep LightGBM as the main model for SHAP reason codes (RM outreach).")
""")

# ---------------------------------------------------------------- 08
md(r"""
## 08 · Evaluation & threshold tuning (LightGBM)
Thresholds are tuned on **train out-of-fold** predictions (never the validation set), then
applied to validation. Two operating points: **recall-leaning** for outreach lists (a missed
SME is lost revenue; a false positive is a cheap call) and **F1-max** for auto-decisions.
""")
code(r"""
lgbm = fitted["LightGBM"]; pva = lgbm.predict_proba(Xva)[:,1]
print("VAL ranking:", {k: round(v,4) for k,v in ranking_metrics(yva, pva).items()})

thr = tune_thresholds(ytr, oof_proba(make_lgbm, Xtr, ytr))   # tuned on TRAIN OOF
print(f"train-OOF thresholds -> F1-max={thr['f1']:.3f} | recall>=0.95={thr['recall']:.3f}")

for name, t in [("F1-max", thr['f1']), ("recall>=0.95", thr['recall'])]:
    m = metrics_at_threshold(yva, pva, t)
    print(f"[{name}] thr={t:.3f}  precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}")

cm = metrics_at_threshold(yva, pva, thr['f1'])["confusion"]
fig, ax = plt.subplots(figsize=(4, 3.4)); plot_confusion(cm, ax, title="LightGBM confusion (val, F1-max)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 09
md("## 09 · Explainability (SHAP)\nGlobal drivers + a per-customer reason code (what an RM would see).")
code(r"""
expl = shap.TreeExplainer(lgbm)
samp = X.sample(min(4000, len(X)), random_state=SEED)
glob = pd.Series(np.abs(shap_positive_values(expl, samp)).mean(0),
                 index=FEATURES).sort_values(ascending=False)
glob.head(15).iloc[::-1].plot.barh(color="#C44E52"); plt.title("SHAP global mean|impact| (top 15)")
plt.xlabel("mean |SHAP|"); plt.tight_layout(); plt.show()
# Optional beeswarm: shap.summary_plot(shap_positive_values(expl, samp), samp)
""")

# ---------------------------------------------------------------- 10
md(r"""
## 10 · Score consumers → hidden entrepreneurs
Out-of-fold `P(business)` for every card (each consumer scored by a model that did not train
on it). High-scoring consumers are the leads.
""")
code(r"""
feat["oof_p"] = oof_proba(make_lgbm, X, y)
con = feat[feat.label==0].copy(); biz = feat[feat.label==1].copy()
print("consumer OOF P(business) percentiles:")
print(con.oof_p.describe(percentiles=[.5,.9,.99,.999]).round(4).to_string())
print("\nconsumers above threshold:", {f"P>={t}": int((con.oof_p>=t).sum()) for t in [0.1,0.3,0.5,0.9]})

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(con.oof_p, bins=np.linspace(0,1,51), color="#4C72B0"); ax.set_yscale("log")
ax.set_title("consumer OOF P(business) — almost all ~0"); ax.set_xlabel("P(business)")
ax.set_ylabel("count (log)"); plt.tight_layout(); plt.show()
""")
code(r"""
leads = con[con.oof_p>=LEAD_THR].sort_values("oof_p", ascending=False)
if len(leads) < 10: leads = con.sort_values("oof_p", ascending=False).head(25)
cmp_cols = ["oof_p","amt_sum","amt_median","online_share","recurring_share","b2b_mcc_share",
            "b2b_amt_share","b2b_unique_merchants","recurring_capable_share","merchant_top_ratio"]
profile = pd.DataFrame({"typical_consumer":con[cmp_cols].median(),
                        "hidden_entrepreneur":leads[cmp_cols].median(),
                        "typical_business":biz[cmp_cols].median()})
print(f"{len(leads)} hidden-entrepreneur leads (P>={LEAD_THR}); profile vs baselines:")
display(profile.round(3))
leads[[ID_COL]+cmp_cols].to_csv("hidden_entrepreneurs.csv", index=False)
print("saved hidden_entrepreneurs.csv")

top = leads.iloc[[0]][FEATURES]
rc = pd.Series(shap_positive_values(expl, top)[0], index=FEATURES).sort_values(key=np.abs, ascending=False).head(8)
print(f"\nreason codes - card {leads.iloc[0][ID_COL]} (P={leads.iloc[0].oof_p:.3f}); +ve => business:")
print(rc.round(3).to_string())
""")

# ---------------------------------------------------------------- 11
md(r"""
## 11 · Business recommendations

**Business signature the model learned:** high **online** share, **business-hours** activity
(low weekend/evening), high **recurring** share (payroll/subscriptions), broad **B2B supplier**
base, larger tickets, and spend concentrated on few merchants (high HHI / top-merchant ratio).

**Action — for the surfaced leads (`hidden_entrepreneurs.csv`):**
- **Tariff migration** — spend profile (median ~170K KZT/tx, ~89% B2B amount share) is SME-like.
- **POS acquiring** — high online + merchant concentration → likely selling.
- **Working-capital loan** — recurring outflows to multiple B2B suppliers → financing fit.
- **Payroll & cash management** — recurring-payment cadence suggests staff/supplier payments.
- **Bookkeeping** — diverse B2B spend → value in automated reconciliation.

**Operating point:** recall-leaning threshold for outreach lists (cheap false positives), F1-max
for auto-migration. Hand RMs the SHAP reason codes per lead for specific, credible outreach.
""")

# ---------------------------------------------------------------- 12
md(r"""
## 12 · Methodology summary (notebook-ready)

1. **Unit & label.** One row per `card_number`; `business=1`, `consumer=0`. ~105k cards, 23.8% positive.
2. **Features (35).** Card-level aggregates only — amount shape, counts, merchant diversity/
   concentration (HHI, entropy, top-ratio), recurring/tokenized/channel/temporal shares, curated
   B2B count & amount share, B2B supplier breadth, recurring-capable-merchant share, monthly-spend
   stability, inter-arrival burstiness, recency.
3. **Leakage control.** Dropped `card_tier` (perfect) and `bank_name` (no signal); no raw
   `mcc`/`merchant_id` one-hot (exclusivity is a coverage artifact); removed `5122` from the B2B
   basket; pruned redundant duplicates (|r|>0.97) and near-random features.
4. **Validation.** Stratified 80/20 split on cards; thresholds tuned on train out-of-fold;
   consumers scored out-of-fold so each is judged by a model that never saw it.
5. **Models.** LogReg (baseline) ≈ LightGBM (main) ≈ RF, all ROC-AUC≈1.0; LightGBM kept for SHAP.
6. **Why random (not chronological) split.** The label is an intrinsic card type observed over a
   fixed 6-month window, not a future event; every card spans the same window and `recency_days`
   is measured from one global cutoff — so there is no temporal target to leak. A chronological
   split would be required only to predict "becomes a business next quarter."
""")

# ---------------------------------------------------------------- 13
md(r"""
## 13 · Why this is trustworthy (presentation-ready)

- **Leakage handled, not hidden.** The one perfect giveaway (`card_tier`) and a business-exclusive
  MCC (`5122`) were removed; we verified **no remaining feature separates the classes alone**
  (max univariate AUC < 1.0). Identifiers (`mcc`, `merchant_id`) are never used as raw features.
- **Honest validation.** Split on cards (no overlap), preprocessing fit on train only, thresholds
  tuned on out-of-fold data, consumers scored out-of-fold. So the ~1.0 metric is not from peeking.
- **The model is explainable & robust.** A linear baseline matches the boosted model → the signal
  is genuine and near-linear, **not** tree overfitting. SHAP gives per-customer reason codes.
- **Right metric, right threshold.** We report PR-AUC and confusion matrices, and choose a
  **recall-leaning** threshold for outreach (a missed SME costs far more than a cheap call) and
  **F1-max** for automated migration.
- **Stated limitation.** ROC-AUC≈1.0 reflects clean synthetic separation and will **not** transfer
  to production unchanged — re-baseline and recalibrate on real labeled cards before trusting any
  threshold. The pipeline (features + OOF scoring + reason codes) is what transfers.
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name":"mdq-venv", "display_name":"Python (MDQ .venv)", "language":"python"}
nb.metadata["language_info"] = {"name":"python"}
with open("notebook.ipynb","w",encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote notebook.ipynb with {len(cells)} cells")
