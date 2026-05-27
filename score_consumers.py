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

    # ---- SHAP: global drivers (on a sample) + per-lead reason codes ----
    model = make_lgbm().fit(X, y)
    expl = shap.TreeExplainer(model)
    samp = X.sample(min(4000, len(X)), random_state=42)
    glob = pd.Series(np.abs(shap_positive_values(expl, samp)).mean(0),
                     index=feats).sort_values(ascending=False)
    print("\n=== SHAP global mean|impact| (top 15) ===")
    print(glob.head(15).round(4).to_string())

    # Full-model SHAP is used only for explanations. The lead scores above remain OOF.
    lead_sv = shap_positive_values(expl, he[feats])
    reason_cols = {
        "reason_1": [],
        "reason_2": [],
        "reason_3": [],
        "why_business": [],
    }
    for i, (_, row) in enumerate(he.iterrows()):
        contrib = pd.Series(lead_sv[i], index=feats)
        top3 = contrib.abs().sort_values(ascending=False).head(3).index
        top_business = contrib[contrib > 0].sort_values(ascending=False).head(3).index
        reasons = []
        for feat in top3:
            direction = "business" if contrib[feat] > 0 else "consumer"
            reasons.append(
                f"{feat}={row[feat]:.4g}; SHAP={contrib[feat]:+.4f}; pushes_to={direction}"
            )
        business_reasons = [
            f"{feat}={row[feat]:.4g}; SHAP={contrib[feat]:+.4f}"
            for feat in top_business
        ]
        for col, value in zip(["reason_1", "reason_2", "reason_3"], reasons):
            reason_cols[col].append(value)
        reason_cols["why_business"].append(" | ".join(business_reasons))

    for col, values in reason_cols.items():
        he[col] = values

    export_cols = [ID_COL, "oof_p"] + cmp_cols[1:] + [
        "reason_1", "reason_2", "reason_3", "why_business"
    ]
    he[export_cols].to_csv(LEADS_FILE, index=False)
    print(f"\nsaved {LEADS_FILE} ({len(he)} rows, with SHAP reason codes)")

    print("\n=== SHAP reason codes for all hidden-entrepreneur leads ===")
    print("(positive SHAP pushes toward 'business')")
    for _, row in he.iterrows():
        print(f"\ncard={row[ID_COL]} P={row['oof_p']:.3f}")
        print(f"  1) {row['reason_1']}")
        print(f"  2) {row['reason_2']}")
        print(f"  3) {row['reason_3']}")


if __name__ == "__main__":
    main()
