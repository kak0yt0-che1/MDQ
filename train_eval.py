"""Train + evaluate models for hidden-entrepreneur detection.

Models: Logistic Regression (baseline), LightGBM (main), Random Forest (compare).
Stratified 80/20 split. Reports ROC-AUC, PR-AUC, confusion matrix, P/R/F1.
Threshold tuned on TRAIN out-of-fold predictions (never on the validation set).

Run: .venv/Scripts/python.exe train_eval.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

from config import SEED, FEAT_FILE, MODEL_FILE, ID_COL, LABEL_COL, TEST_SIZE
from mdq_utils import ranking_metrics, metrics_at_threshold, tune_thresholds, oof_proba

# heavy-tailed columns -> log1p before scaling (helps the linear baseline only)
SKEWED = ["tx_count", "amt_sum", "amt_mean", "amt_median", "amt_std", "amt_max",
          "amt_min", "n_unique_merchants", "n_unique_mcc",
          "n_unique_countries", "active_days", "active_weeks", "span_days",
          "gap_mean", "gap_std", "tx_per_active_day", "b2b_unique_merchants"]


def load_xy():
    f = pd.read_parquet(FEAT_FILE)
    feats = [c for c in f.columns if c not in (ID_COL, LABEL_COL)]
    return f, feats


def make_logreg(features):
    skewed = [c for c in SKEWED if c in features]
    rest = [c for c in features if c not in skewed]
    log_pipe = Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                         ("sc", StandardScaler())])
    pre = ColumnTransformer([("log", log_pipe, skewed),
                             ("sc", StandardScaler(), rest)])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                                class_weight="balanced",
                                                random_state=SEED))])


def make_lgbm():
    # importance_type='gain' so feature_importances_ reflects loss reduction,
    # not raw split counts.
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, max_depth=-1,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, importance_type="gain",
        random_state=SEED, n_jobs=-1, verbose=-1)


def make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=20,
        n_jobs=-1, class_weight="balanced", random_state=SEED)


def _print_point(name, m):
    cm = m["confusion"]
    print(f"\n[{name}] threshold={m['threshold']:.3f}  "
          f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}")
    print(f"  confusion [rows=true 0/1, cols=pred 0/1]:\n{cm}")


def main():
    f, feats = load_xy()
    X, y = f[feats], f[LABEL_COL].values
    print(f"samples={len(f):,}  features={len(feats)}  positive_rate={y.mean():.4f}")

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"train={len(Xtr):,}  val={len(Xva):,}")

    models = {"LogReg": make_logreg(feats), "LightGBM": make_lgbm(), "RandomForest": make_rf()}
    fitted, results = {}, []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        results.append({"model": name, **ranking_metrics(yva, p)})
        fitted[name] = m
    print("\n=== VALIDATION RANKING METRICS ===")
    print(pd.DataFrame(results).set_index("model").round(4).to_string())

    # ---- threshold tuning on TRAIN out-of-fold preds (not the val set) ----
    oof = oof_proba(make_lgbm, Xtr, ytr)
    thr = tune_thresholds(ytr, oof)
    print(f"\nTuned on TRAIN OOF -> F1-max thr={thr['f1']:.3f} | outreach (prec>=0.5) thr={thr['outreach']:.3f}")

    # ---- final evaluation of the main model on VAL ----
    lgbm = fitted["LightGBM"]
    pva = lgbm.predict_proba(Xva)[:, 1]
    print("\n=== LightGBM final on VALIDATION ===")
    print("ranking:", {k: round(v, 4) for k, v in ranking_metrics(yva, pva).items()})
    _print_point("default", metrics_at_threshold(yva, pva, 0.50))
    _print_point("F1-max", metrics_at_threshold(yva, pva, thr["f1"]))
    _print_point("outreach (prec>=0.5)", metrics_at_threshold(yva, pva, thr["outreach"]))
    print("\nclassification report @ F1-max:")
    print(classification_report(yva, (pva >= thr["f1"]).astype(int),
                                target_names=["consumer", "business"], digits=4))

    # ---- gain importance ----
    imp = pd.Series(lgbm.feature_importances_, index=feats).sort_values(ascending=False)
    print("=== LightGBM gain importance (top 15) ===")
    print(imp.head(15).round(1).to_string())

    # ---- persist artifacts for SHAP + consumer scoring ----
    joblib.dump({"model": lgbm, "features": feats,
                 "thr_f1": thr["f1"], "thr_outreach": thr["outreach"]}, MODEL_FILE)
    print(f"\nsaved {MODEL_FILE}")


if __name__ == "__main__":
    main()
