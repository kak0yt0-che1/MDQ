"""Find hidden entrepreneurs among consumer cards + explain the model.

Method: out-of-fold (OOF) scoring so every consumer is scored by a model that
did NOT see it in training -> honest P(business). Consumers with high OOF
P(business) are the actionable hidden-entrepreneur leads. Then SHAP for global
drivers and per-customer reason codes.

Run: .venv/Scripts/python.exe score_consumers.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap

from config import LEAD_THR, LEADS_FILE, ID_COL, LABEL_COL
from train_eval import make_lgbm, load_xy
from mdq_utils import oof_proba, shap_positive_values


def main():
    f, feats = load_xy()
    X, y = f[feats], f[LABEL_COL].values

    # ---- honest OOF P(business) for every card ----
    f["oof_p"] = oof_proba(make_lgbm, X, y)
    con = f[f[LABEL_COL] == 0].copy()
    biz = f[f[LABEL_COL] == 1].copy()
    print(f"consumers={len(con):,}  businesses={len(biz):,}")

    # ---- distribution of consumer scores: do hidden entrepreneurs exist? ----
    print("\n=== consumer OOF P(business) distribution ===")
    print(con["oof_p"].describe(percentiles=[.5, .9, .99, .999]).round(4).to_string())
    for t in [0.1, 0.2, 0.3, 0.5, 0.8, 0.9]:
        n = int((con["oof_p"] >= t).sum())
        print(f"  consumers with P>= {t:.2f}: {n:,} ({n/len(con)*100:.3f}%)")

    # ---- rank hidden entrepreneurs (actionable leads above threshold) ----
    he = con[con["oof_p"] >= LEAD_THR].sort_values("oof_p", ascending=False)
    if len(he) < 10:  # ensure a minimum review batch even if signal is sparse
        he = con.sort_values("oof_p", ascending=False).head(25)
    print(f"\n=== {len(he)} hidden-entrepreneur leads (P>={LEAD_THR}): profile vs baselines ===")
    cmp_cols = ["oof_p", "amt_sum", "amt_median", "online_share", "recurring_share",
                "b2b_mcc_share", "b2b_amt_share", "tokenized_share", "bizhours_share",
                "weekend_share", "evening_share", "merchant_top_ratio", "foreign_share"]
    profile = pd.DataFrame({
        "typical_consumer": con[cmp_cols].median(),
        "hidden_entrepreneur": he[cmp_cols].median(),
        "typical_business": biz[cmp_cols].median(),
    })
    print(profile.round(3).to_string())

    he[[ID_COL, "oof_p"] + cmp_cols[1:]].to_csv(LEADS_FILE, index=False)
    print(f"\nsaved {LEADS_FILE} ({len(he)} rows)")

    # ---- SHAP: global drivers (on a sample) ----
    model = make_lgbm().fit(X, y)
    expl = shap.TreeExplainer(model)
    samp = X.sample(min(4000, len(X)), random_state=42)
    glob = pd.Series(np.abs(shap_positive_values(expl, samp)).mean(0),
                     index=feats).sort_values(ascending=False)
    print("\n=== SHAP global mean|impact| (top 15) ===")
    print(glob.head(15).round(4).to_string())

    # ---- SHAP reason codes for the single highest-scoring hidden entrepreneur ----
    top1 = he.iloc[[0]][feats]
    contrib = pd.Series(shap_positive_values(expl, top1)[0],
                        index=feats).sort_values(key=np.abs, ascending=False)
    print(f"\n=== reason codes: top lead card={he.iloc[0][ID_COL]} P={he.iloc[0]['oof_p']:.3f} ===")
    print("(positive SHAP pushes toward 'business')")
    print(contrib.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
