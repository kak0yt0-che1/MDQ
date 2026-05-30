"""Build notebook.ipynb from a single source.

The competition allows one notebook only, so this file inlines every helper from
the dev modules (config / features / mdq_utils / train_eval) into the notebook.
Modules stay in the repo as a more comfortable place to edit logic.

Usage:
    python build_notebook.py
    python -m nbconvert --to notebook --execute --inplace notebook.ipynb
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(s):   cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))


# ----- title -------------------------------------------------------------
md(r"""
# Hidden-Entrepreneur Detection

Train a per-card classifier on the two labeled pools (consumer / business), then
score the consumer pool out-of-fold to find cards that behave like a business but
are filed as a consumer. The bank can move those clients to SME products
(POS acquiring, working-capital loans, payroll, cash management, bookkeeping).

Data: 105k cards (80k consumer + 25k business), 12.8M transactions over 6 months,
Kazakhstan-time timestamps. Synthetic, but the same pipeline transfers to real data.

Output: `submission.csv` with one row per `card_number` and a `score` in [0, 1].

Run top to bottom. Everything is seeded.
""")


# ----- 1. setup ----------------------------------------------------------
md("## 1. Setup")
code(r"""
from __future__ import annotations
import os, gc, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             classification_report, precision_recall_curve, roc_curve)
from lightgbm import LGBMClassifier
from IPython.display import display

warnings.filterwarnings("ignore", category=UserWarning, module="shap")

SEED = 42
CONSUMER_FILE = "consumer_cards_MDQ.parquet"
BUSINESS_FILE = "business_cards_MDQ.parquet"
MERCHANT_FILE = "merchants_reference.parquet"
CACHE_FILE    = "features_card_level.parquet"
SUBMISSION    = "submission.csv"
LEADS_FILE    = "hidden_entrepreneurs.csv"

ID_COL, LABEL_COL = "card_number", "label"
TEST_SIZE, N_SPLITS = 0.20, 5
LEAD_THR = 0.30          # consumer score cutoff for the leads file
OUTREACH_PRECISION = 0.50

# B2B / wholesale / professional-services MCCs, curated from MCC semantics.
# 5122 sits in this category by definition but only business cards use it in this
# sample, so it would leak into b2b_*_share if kept. Hence excluded.
B2B_MCC = {
    "5044","5045","5046","5047","5051","5065","5072","5074","5085",
    "5111","5131","5137","5139","5169","5172","5192","5198","5199",
    "7311","7321","7333","7338","7339","7342","7349","7361","7372",
    "7375","7379","7392","7393","7394","7399","8742","8911","8931",
    "2741","2791","2842",
    "4214","4215","4225","4816",
}
DROP_COLS = ["card_tier", "bank_name"]


def ranking_metrics(y, p):
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p)}


def metrics_at_threshold(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"threshold": float(thr),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall":    recall_score(y, yhat, zero_division=0),
            "f1":        f1_score(y, yhat, zero_division=0),
            "confusion": confusion_matrix(y, yhat)}


def tune_thresholds(y, proba, min_precision=OUTREACH_PRECISION):
    # f1: strict cutoff for automated migration.
    # outreach: smallest cutoff where precision is still acceptable, so the lead
    # pool stays as large as possible (a missed SME costs more than one cheap call).
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    thr_f1 = float(thr[np.nanargmax(f1[:-1])])
    mask = prec[:-1] >= min_precision
    thr_outreach = float(thr[mask].min()) if mask.any() else thr_f1
    return {"f1": thr_f1, "outreach": thr_outreach}


def oof_proba(make_estimator, X, y, seed=SEED, n_splits=N_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(make_estimator(), X, y, cv=skf,
                             method="predict_proba", n_jobs=-1)[:, 1]


def shap_positive(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def plot_confusion(cm, ax, title=""):
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["consumer", "business"])
    ax.set_yticklabels(["consumer", "business"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)


np.random.seed(SEED)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["figure.dpi"] = 110
print("setup ok, seed =", SEED)
""")


# ----- 2. load data ------------------------------------------------------
md("## 2. Load data")
code(r"""
consumer  = pd.read_parquet(CONSUMER_FILE)
business  = pd.read_parquet(BUSINESS_FILE)
merchants = pd.read_parquet(MERCHANT_FILE)
print("consumer", consumer.shape, "| business", business.shape, "| merchants", merchants.shape)
display(consumer.dtypes.to_frame("dtype"))
display(consumer.head(3))
""")


# ----- 3. quick EDA ------------------------------------------------------
md("""
## 3. EDA

A few sanity checks before features: nulls, duplicates, amount range, and the
hour / channel / recurring profile differences that the model should pick up.
""")
code(r"""
def load_labeled():
    c = pd.read_parquet(CONSUMER_FILE); c[LABEL_COL] = 0
    b = pd.read_parquet(BUSINESS_FILE); b[LABEL_COL] = 1
    return pd.concat([c, b], ignore_index=True)


df = load_labeled()
print(f"rows={len(df):,}  cards={df[ID_COL].nunique():,}  "
      f"consumer={df.loc[df.label==0, ID_COL].nunique():,}  "
      f"business={df.loc[df.label==1, ID_COL].nunique():,}")
print(f"nulls={int(df.isna().sum().sum())}  duplicates={int(df.duplicated().sum())}")
print(f"timestamp: {df.transaction_timestamp.min()}  ->  {df.transaction_timestamp.max()}")
print(f"amount KZT: min={df.transaction_amount_kzt.min():,}  "
      f"max={df.transaction_amount_kzt.max():,}  "
      f"non-positive={(df.transaction_amount_kzt <= 0).sum()}")

print("\namount per class:")
display(df.groupby(LABEL_COL).transaction_amount_kzt
          .describe(percentiles=[.5, .9, .99]).round(0))
""")
code(r"""
# Transaction-level share contrast. The class gap on these signals shows up
# again as feature gaps once we aggregate per card.
ts = pd.to_datetime(df["transaction_timestamp"])
sig = df.assign(
    online   = (df.channel == "online"),
    wknd     = (ts.dt.dayofweek >= 5),
    eve      = ts.dt.hour.between(18, 23),
    bizhours = (ts.dt.dayofweek < 5) & ts.dt.hour.between(9, 17),
)
contrast = (sig.groupby(LABEL_COL)[["online","is_recurring","tokenized","wknd","eve","bizhours"]]
              .mean().T)
contrast.columns = ["consumer", "business"]
display(contrast.round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for lab, name, col in [(0, "consumer", "#4C72B0"), (1, "business", "#DD8452")]:
    ax[0].hist(np.log10(df.loc[df.label == lab, "transaction_amount_kzt"].clip(lower=1)),
               bins=60, alpha=.55, label=name, color=col, density=True)
ax[0].set_title("log10 transaction amount (KZT)")
ax[0].set_xlabel("log10 KZT"); ax[0].legend()
contrast.plot.bar(ax=ax[1], color=["#4C72B0", "#DD8452"])
ax[1].set_title("transaction-level shares by class")
ax[1].set_ylabel("share")
plt.tight_layout(); plt.show()
""")


# ----- 4. leakage screen -------------------------------------------------
md("""
## 4. Leakage screen

Two known hazards in synthetic data: a perfect-separator column, and high-cardinality
ID-like columns that look harmless but split the classes by coverage. We flag and
drop them here so the rest of the pipeline is forced to find behavior, not artifacts.
""")
code(r"""
print("card_tier x label (consumer cards never carry a Business-tier):")
display(pd.crosstab(df.label, df.card_tier))

con_m, biz_m = df[df.label == 0], df[df.label == 1]
for col in ["mcc", "merchant_id"]:
    cset, bset = set(con_m[col].unique()), set(biz_m[col].unique())
    only_b = bset - cset
    vol = biz_m[col].isin(only_b).mean()
    print(f"{col:11s}  biz_only={len(only_b):4d}  con_only={len(cset-bset):4d}  "
          f"shared={len(cset & bset):4d}   biz vol in biz-only {col}: {vol:.3%}")

# Sanity check on the curated basket: no business-exclusive code may sneak in.
biz_only_mcc = set(biz_m.mcc.unique()) - set(con_m.mcc.unique())
basket_leak  = sorted(B2B_MCC & biz_only_mcc)
print(f"\nB2B basket size: {len(B2B_MCC)}  "
      f"basket ∩ biz-exclusive: {basket_leak or 'none'}")
print(f"dropped before modeling: {DROP_COLS}")
""")


# ----- 5. features -------------------------------------------------------
md("""
## 5. Card-level features

35 features in five rough groups: spend volume and dispersion, merchant diversity
and concentration, B2B exposure (count, amount share, supplier breadth), temporal
and recurring patterns, and inter-arrival burstiness.

Aggregations are always per `card_number`, so transaction-level leakage cannot
survive the rollup.
""")
code(r"""
def _helpers(df):
    ts = pd.to_datetime(df["transaction_timestamp"])
    df["_hour"]       = ts.dt.hour.astype("int16")
    dow               = ts.dt.dayofweek.astype("int16")
    df["_is_weekend"] = (dow >= 5)
    df["_is_evening"] = df["_hour"].between(18, 23)
    df["_is_bizhours"]= (~df["_is_weekend"]) & df["_hour"].between(9, 17)
    df["_is_online"]  = (df["channel"] == "online")
    df["_is_foreign"] = (df["country"] != "Kazakhstan")
    df["_is_b2b_mcc"] = df["mcc"].isin(B2B_MCC)
    df["_amt"]        = df["transaction_amount_kzt"].astype("float64")
    df["_date"]       = ts.dt.normalize()
    # Period-strings so distinct weeks never collide across year boundaries.
    df["_week"]       = ts.dt.to_period("W").astype("string")
    df["_month"]      = ts.dt.to_period("M").astype("string")
    return df


def _concentration(df, key, prefix):
    g = df.groupby([ID_COL, key], observed=True).size().rename("n")
    p = g / g.groupby(level=0, observed=True).transform("sum")
    return pd.DataFrame({
        f"{prefix}_hhi":       (p * p).groupby(level=0, observed=True).sum(),
        f"{prefix}_top_ratio":  p.groupby(level=0, observed=True).max(),
        f"{prefix}_entropy":   (-(p * np.log(p))).groupby(level=0, observed=True).sum(),
    })


def build_card_features(tx, merchants=None):
    tx = _helpers(tx)
    obs_end = tx["_date"].max()
    if merchants is None:
        merchants = pd.read_parquet(MERCHANT_FILE)
    rec_cap = (merchants.drop_duplicates("merchant_id")
                        .set_index("merchant_id")["recurring_capable"])
    tx["_rec_capable"] = tx["merchant_id"].map(rec_cap).fillna(False)

    g = tx.groupby(ID_COL, observed=True)
    feat = g.agg(
        label=("label", "first"),
        tx_count=("_amt", "size"),
        amt_sum=("_amt", "sum"), amt_mean=("_amt", "mean"), amt_median=("_amt", "median"),
        amt_std=("_amt", "std"), amt_max=("_amt", "max"), amt_min=("_amt", "min"),
        n_unique_merchants=("merchant_id", "nunique"),
        n_unique_mcc=("mcc", "nunique"),
        n_unique_countries=("country", "nunique"),
        recurring_share=("is_recurring", "mean"),
        tokenized_share=("tokenized", "mean"),
        online_share=("_is_online", "mean"),
        weekend_share=("_is_weekend", "mean"),
        evening_share=("_is_evening", "mean"),
        bizhours_share=("_is_bizhours", "mean"),
        foreign_share=("_is_foreign", "mean"),
        b2b_mcc_share=("_is_b2b_mcc", "mean"),
        recurring_capable_share=("_rec_capable", "mean"),
        active_days=("_date", "nunique"),
        active_weeks=("_week", "nunique"),
        first_date=("_date", "min"),
        last_date=("_date", "max"),
    )

    # B2B share of spend (not just count): a higher-fidelity exposure signal.
    feat["b2b_amt_share"] = (
        tx.assign(_b2b_amt=tx["_amt"].where(tx["_is_b2b_mcc"], 0.0))
          .groupby(ID_COL, observed=True)["_b2b_amt"].sum() / feat["amt_sum"]
    )
    feat["amt_cv"]            = feat["amt_std"] / feat["amt_mean"].replace(0, np.nan)
    feat["span_days"]         = (feat["last_date"] - feat["first_date"]).dt.days + 1
    feat["tx_per_active_day"] = feat["tx_count"] / feat["active_days"]
    feat["recency_days"]      = (obs_end - feat["last_date"]).dt.days
    feat["merchants_per_tx"]  = feat["n_unique_merchants"] / feat["tx_count"]
    feat = feat.join(_concentration(tx, "merchant_id", "merchant"))

    b2b_u = tx[tx["_is_b2b_mcc"]].groupby(ID_COL, observed=True)["merchant_id"].nunique()
    feat["b2b_unique_merchants"] = b2b_u.reindex(feat.index).fillna(0).astype("int64")

    monthly = tx.groupby([ID_COL, "_month"], observed=True)["_amt"].sum()
    mm = monthly.groupby(level=0, observed=True).agg(m="mean", s="std")
    feat["monthly_spend_cv"] = mm["s"] / mm["m"].replace(0, np.nan)

    d = tx[[ID_COL, "transaction_timestamp"]].sort_values([ID_COL, "transaction_timestamp"])
    gap_h = (d.groupby(ID_COL, observed=True)["transaction_timestamp"]
               .diff().dt.total_seconds() / 3600.0)
    gstat = (d.assign(gap=gap_h)
              .groupby(ID_COL, observed=True)["gap"]
              .agg(gap_mean="mean", gap_std="std"))
    # Goh-Barabasi burstiness in [-1, 1]: >0 bursty, <0 regular.
    gstat["burstiness"] = ((gstat["gap_std"] - gstat["gap_mean"]) /
                           (gstat["gap_std"] + gstat["gap_mean"]).replace(0, np.nan))
    feat = feat.join(gstat)

    feat = feat.drop(columns=["first_date", "last_date"])
    # Cards with one transaction have no std / no gaps; fill those with 0.
    feat = feat.fillna({"amt_std": 0.0, "amt_cv": 0.0, "gap_mean": 0.0,
                        "gap_std": 0.0, "burstiness": 0.0, "monthly_spend_cv": 0.0})
    return feat.reset_index()


if os.path.exists(CACHE_FILE):
    feat = pd.read_parquet(CACHE_FILE)
    print(f"loaded cached features {feat.shape}")
else:
    feat = build_card_features(df, merchants)
    feat.to_parquet(CACHE_FILE, index=False)
    print(f"built features {feat.shape}")

FEATURES = [c for c in feat.columns if c not in (ID_COL, LABEL_COL)]
print(f"features={len(FEATURES)}  positive_rate={feat[LABEL_COL].mean():.4f}")
del df; gc.collect()
display(feat[FEATURES].head(3))
""")

code(r"""
# Quick sanity: do the features separate the classes the way we expect?
g = feat.groupby(LABEL_COL)[FEATURES].mean().T
g.columns = ["consumer", "business"]
g["b/c"] = (g.business / g.consumer.replace(0, np.nan)).round(2)
display(g.round(3))
""")


# ----- 6. univariate AUC -------------------------------------------------
md("""
## 6. Univariate AUC

If a single feature reaches AUC = 1.0 we still have a leak. The top features here
sit at 0.92-0.99 — strong but not perfect, which is what we want.
""")
code(r"""
y_all = feat[LABEL_COL].values
uni = pd.Series({c: max(roc_auc_score(y_all, feat[c]),
                        1 - roc_auc_score(y_all, feat[c])) for c in FEATURES})
uni = uni.sort_values(ascending=False)
print(f"max univariate AUC: {uni.max():.4f}")
uni.head(12).iloc[::-1].plot.barh(color="#55A868")
plt.title("top 12 single-feature AUC")
plt.xlabel("AUC"); plt.xlim(0.5, 1.0)
plt.tight_layout(); plt.show()
""")


# ----- 7. split ----------------------------------------------------------
md("""
## 7. Stratified split

One row per `card_number`, so a row-level split is automatically a card-level split.
Stratify on the label to keep the 23.8 % positive rate in both sides.

A chronological split would be the right call if the label was "becomes a business
next quarter", but this label is an intrinsic card type observed over a fixed
window, so random is fine here.
""")
code(r"""
X, y = feat[FEATURES], feat[LABEL_COL].values
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE,
                                      stratify=y, random_state=SEED)
print(f"train={Xtr.shape}  val={Xva.shape}  val_pos={yva.mean():.4f}")
""")


# ----- 8. three models ---------------------------------------------------
md("""
## 8. Three models

LogReg is the baseline; LightGBM is the main model; RandomForest is a sanity check.
Logistic regression gets log1p + scaling inside a Pipeline so the transformer is
fit on train only. Tree models take the raw features.
""")
code(r"""
SKEWED = ["tx_count","amt_sum","amt_mean","amt_median","amt_std","amt_max","amt_min",
          "n_unique_merchants","n_unique_mcc","n_unique_countries","active_days",
          "active_weeks","span_days","gap_mean","gap_std","tx_per_active_day",
          "b2b_unique_merchants"]


def make_logreg(features):
    skewed = [c for c in SKEWED if c in features]
    rest   = [c for c in features if c not in skewed]
    log_pipe = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("sc", StandardScaler()),
    ])
    pre = ColumnTransformer([("log", log_pipe, skewed),
                             ("sc", StandardScaler(), rest)])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                   class_weight="balanced", random_state=SEED)),
    ])


def make_lgbm():
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, importance_type="gain",
        random_state=SEED, n_jobs=-1, verbose=-1,
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=20,
        n_jobs=-1, class_weight="balanced", random_state=SEED,
    )


models = {"LogReg": make_logreg(FEATURES), "LightGBM": make_lgbm(), "RandomForest": make_rf()}
fitted, rows = {}, []
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xva)[:, 1]
    fitted[name] = m
    rows.append({"model": name, **ranking_metrics(yva, p)})
display(pd.DataFrame(rows).set_index("model").round(4))
print("All three saturate near 1.0. The fact that the linear baseline matches")
print("the boosted model is a signal that the classes are near-linearly separable")
print("on these features, not that the trees are overfitting.")
""")


# ----- 9. evaluation -----------------------------------------------------
md("""
## 9. ROC + PR curves

The dataset is 23.8 % positive, which is not catastrophically imbalanced, but
accuracy is still a bad headline metric: a "predict consumer" classifier scores
76 % accuracy and zero recall on businesses. ROC-AUC is the competition metric;
PR-AUC gives a sharper view of the positive class.
""")
code(r"""
lgbm = fitted["LightGBM"]
pva  = lgbm.predict_proba(Xva)[:, 1]
print("val ranking:", {k: round(v, 4) for k, v in ranking_metrics(yva, pva).items()})

fpr, tpr, _ = roc_curve(yva, pva)
prec_v, rec_v, _ = precision_recall_curve(yva, pva)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(fpr, tpr, color="#C44E52")
ax[0].plot([0, 1], [0, 1], "--", color="grey")
ax[0].set_title(f"ROC, AUC = {roc_auc_score(yva, pva):.4f}")
ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[1].plot(rec_v, prec_v, color="#4C72B0")
ax[1].axhline(yva.mean(), ls="--", color="grey", label=f"base rate {yva.mean():.3f}")
ax[1].set_title(f"PR, AP = {average_precision_score(yva, pva):.4f}")
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].legend()
plt.tight_layout(); plt.show()
""")


# ----- 10. thresholds ----------------------------------------------------
md("""
## 10. Two operating points

Both thresholds are tuned on the **training** OOF scores (not on validation):

- `f1`: F1-max, for an automated tariff-migration trigger that should rarely
  misfire.
- `outreach`: smallest cutoff whose precision is still >= 0.5, used to build an
  RM call list. A missed SME costs more than a cheap call, so we want the lead
  pool as large as it can be while staying at least half-right.
""")
code(r"""
thr = tune_thresholds(ytr, oof_proba(make_lgbm, Xtr, ytr))
print(f"train-OOF thresholds: f1={thr['f1']:.4f}  "
      f"outreach (prec>={OUTREACH_PRECISION})={thr['outreach']:.4f}")

for name, t in [("f1", thr["f1"]), ("outreach", thr["outreach"])]:
    m = metrics_at_threshold(yva, pva, t)
    print(f"[{name:8s}] thr={t:.4f}  prec={m['precision']:.4f}  "
          f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")

cm_f1 = metrics_at_threshold(yva, pva, thr["f1"])["confusion"]
cm_or = metrics_at_threshold(yva, pva, thr["outreach"])["confusion"]
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
plot_confusion(cm_f1, ax[0], title=f"f1-max ({thr['f1']:.3f})")
plot_confusion(cm_or, ax[1], title=f"outreach ({thr['outreach']:.3f})")
plt.tight_layout(); plt.show()

print("\nclassification report at f1-max:")
print(classification_report(yva, (pva >= thr["f1"]).astype(int),
                            target_names=["consumer", "business"], digits=4))
""")


# ----- 11. shap ----------------------------------------------------------
md("""
## 11. SHAP

Global driver order plus a beeswarm. The same explainer feeds the per-card reason
codes in the leads file below.
""")
code(r"""
expl = shap.TreeExplainer(lgbm)
samp = X.sample(min(4000, len(X)), random_state=SEED)
shap_vals = shap_positive(expl, samp)

glob = pd.Series(np.abs(shap_vals).mean(0), index=FEATURES).sort_values(ascending=False)
glob.head(15).iloc[::-1].plot.barh(color="#C44E52")
plt.title("SHAP global mean |impact| (top 15)")
plt.xlabel("mean |SHAP|")
plt.tight_layout(); plt.show()

shap.summary_plot(shap_vals, samp, max_display=15, show=False)
plt.tight_layout(); plt.show()
""")


# ----- 12. OOF on full data ---------------------------------------------
md("""
## 12. Out-of-fold scoring of every card

5-fold OOF on the whole labeled set, so each card gets a score from a model that
never trained on it. This is the score we'll use both for the submission file
and for finding hidden entrepreneurs in the consumer pool.
""")
code(r"""
feat["oof_p"] = oof_proba(make_lgbm, X, y)
print("OOF ranking:", {k: round(v, 4) for k, v in ranking_metrics(y, feat.oof_p.values).items()})
display(feat.groupby(LABEL_COL).oof_p.describe(percentiles=[.5, .9, .99, .999]).round(4))
""")


# ----- 13. submission ----------------------------------------------------
md("""
## 13. Submission

`submission.csv` with one row per `card_number` and a `score` column. We write
the OOF scores first and overwrite with the full-refit model at the bottom of
the notebook (the actual scoring artifact).
""")
code(r"""
sub = feat[[ID_COL, "oof_p"]].rename(columns={"oof_p": "score"})
sub.to_csv(SUBMISSION, index=False)
print(f"wrote {SUBMISSION}, rows={len(sub):,}")
display(sub.head())
print("\nscore distribution:")
print(sub.score.describe(percentiles=[.5, .9, .99, .999]).round(4).to_string())
""")


# ----- 14. hidden entrepreneurs -----------------------------------------
md("""
## 14. Hidden-entrepreneur leads

Filter consumers (`label == 0`) by OOF score and rank. On this synthetic data
only ~11 consumers cross the 0.30 cutoff; on real bank data the same pipeline
would surface a much larger pool.

Each row of the leads file also gets a short SHAP reason-code string so an RM
can open the file and see the top drivers for that specific card.
""")
code(r"""
con = feat[feat.label == 0].copy()
biz = feat[feat.label == 1].copy()
print(f"consumers={len(con):,}  businesses={len(biz):,}")
print("consumers above each cutoff:")
for t in [0.1, 0.2, 0.3, 0.5, 0.8, 0.9]:
    n = int((con.oof_p >= t).sum())
    print(f"  P>={t:.2f}: {n:,} ({n/len(con)*100:.3f}%)")

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(con.oof_p, bins=np.linspace(0, 1, 51), color="#4C72B0")
ax.set_yscale("log")
ax.set_title("consumer OOF P(business), almost all near 0")
ax.set_xlabel("P(business)"); ax.set_ylabel("count (log)")
plt.tight_layout(); plt.show()
""")
code(r"""
leads = con[con.oof_p >= LEAD_THR].sort_values("oof_p", ascending=False)
if len(leads) < 10:
    # fallback so a sparse synthetic pool still gives the analyst something to look at
    leads = con.sort_values("oof_p", ascending=False).head(25)

profile_cols = ["oof_p","amt_sum","amt_median","online_share","recurring_share",
                "b2b_mcc_share","b2b_amt_share","b2b_unique_merchants",
                "recurring_capable_share","merchant_top_ratio","bizhours_share"]
profile = pd.DataFrame({
    "typical_consumer":    con[profile_cols].median(),
    "hidden_entrepreneur": leads[profile_cols].median(),
    "typical_business":    biz[profile_cols].median(),
})
print(f"{len(leads)} leads (P>={LEAD_THR}). median profile vs baselines:")
display(profile.round(3))


def reason_codes(idx, k=5):
    rc = pd.Series(shap_positive(expl, leads.loc[[idx], FEATURES])[0],
                   index=FEATURES).sort_values(key=np.abs, ascending=False).head(k)
    return ", ".join(f"{f}({'+' if v >= 0 else ''}{v:.2f})" for f, v in rc.items())


leads["reason_codes"] = [reason_codes(i) for i in leads.index]
leads[[ID_COL] + profile_cols + ["reason_codes"]].to_csv(LEADS_FILE, index=False)
print(f"\nsaved {LEADS_FILE} ({len(leads)} rows)")
display(leads[[ID_COL, "oof_p", "amt_sum", "b2b_amt_share", "reason_codes"]].head(10))
""")


# ----- 15. business mapping ---------------------------------------------
md("""
## 15. Business mapping

The model picks up a fairly clean SME signature: high online share, more activity
in business hours than evenings, larger and more concentrated B2B spend, and a
recurring-payment cadence. Each of those maps to a bank product:

- **POS acquiring** for cards with high online share + merchant concentration.
- **Working-capital loans** for cards with large recurring B2B outflows.
- **Payroll / salary projects** for cards with high recurring share to many
  counterparties.
- **Tariff migration** to an SME plan once `amt_sum` and `amt_median` cross the
  consumer band.
- **Cash management and bookkeeping** for cards with a broad B2B supplier base.
- **FX / cross-border** offer for cards with `foreign_share` above ~0.2.

Operationally: outreach-threshold list goes to an RM with the reason codes;
F1-max threshold is the strict gate for automated migration.
""")


# ----- 16. methodology summary ------------------------------------------
md("""
## 16. Methodology notes

1. **Unit & label.** 105 000 cards, 23.8 % business. One row per `card_number`.
2. **Features (35).** Card-level aggregates only: amount shape, merchant
   diversity and concentration, curated B2B exposure, temporal pattern,
   recurring behavior, monthly stability, inter-arrival burstiness, recency.
3. **Leakage handling.** `card_tier` and `bank_name` dropped; raw `mcc` and
   `merchant_id` never one-hot encoded; B2B basket curated from MCC semantics
   and checked to contain zero business-exclusive codes.
4. **Validation.** Stratified 80/20 on cards; preprocessing inside a Pipeline so
   it is fit on train only; thresholds tuned on train OOF; leads scored OOF.
5. **Models.** LogReg, LightGBM, RandomForest all reach ROC-AUC ~1.0 on this
   synthetic data. We keep LightGBM as the main model for SHAP.
6. **Metrics.** ROC-AUC, PR-AUC, plus confusion matrices and the classification
   report at the two tuned thresholds.
7. **Caveat.** The near-perfect score reflects clean synthetic separation; the
   ranking and explanations transfer, the absolute numbers will not.
""")


# ----- 17. inference recipe ---------------------------------------------
md("""
## 17. Scoring a new test set

If a held-out test parquet arrives with the same schema, the recipe is short:

```python
test_tx = pd.read_parquet("test_cards.parquet")
test_tx["label"] = 0
test_feat = build_card_features(test_tx, merchants)
final = make_lgbm().fit(feat[FEATURES], feat[LABEL_COL])
test_feat["score"] = final.predict_proba(test_feat[FEATURES])[:, 1]
test_feat[[ID_COL, "score"]].to_csv("test_submission.csv", index=False)
```

The cell below does the same refit + score on the labeled data and overwrites
`submission.csv` with the full-fit model's predictions, which is what we ship.
""")
code(r"""
final = make_lgbm().fit(feat[FEATURES], feat[LABEL_COL])
feat["score"] = final.predict_proba(feat[FEATURES])[:, 1]
print("full-refit ranking:",
      {k: round(v, 4) for k, v in ranking_metrics(feat[LABEL_COL], feat.score).items()})
feat[[ID_COL, "score"]].to_csv(SUBMISSION, index=False)
print(f"overwrote {SUBMISSION} with full-refit scores")
""")


# ----- write out --------------------------------------------------------
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "name": "mdq-venv", "display_name": "Python (MDQ .venv)", "language": "python",
}
nb.metadata["language_info"] = {"name": "python"}
with open("notebook.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote notebook.ipynb with {len(cells)} cells")
