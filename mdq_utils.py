"""Metric, threshold, OOF and SHAP helpers shared by the scripts and the notebook."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             precision_recall_curve)

from config import SEED, N_SPLITS


def ranking_metrics(y, p) -> dict:
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p)}


def metrics_at_threshold(y, p, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    return {"threshold": float(thr),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall":    recall_score(y, yhat, zero_division=0),
            "f1":        f1_score(y, yhat, zero_division=0),
            "confusion": confusion_matrix(y, yhat)}


def tune_thresholds(y, proba, min_precision_outreach: float = 0.50) -> dict:
    # f1: strict cutoff for automated migration.
    # outreach: smallest cutoff whose precision is still acceptable, so the lead pool
    # stays as large as possible (a missed SME costs more than a cheap call).
    # Always tune on train / OOF, never on the held-out set.
    prec, rec, thr = precision_recall_curve(y, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    thr_f1 = float(thr[np.nanargmax(f1s[:-1])])
    mask = prec[:-1] >= min_precision_outreach
    thr_outreach = float(thr[mask].min()) if mask.any() else thr_f1
    return {"f1": thr_f1, "outreach": thr_outreach}


def oof_proba(make_estimator, X, y, seed: int = SEED, n_splits: int = N_SPLITS):
    # `make_estimator` is a zero-arg factory so each fold gets a fresh, unfitted model.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(make_estimator(), X, y, cv=skf,
                             method="predict_proba", n_jobs=-1)[:, 1]


def shap_positive_values(explainer, X) -> np.ndarray:
    # SHAP / LightGBM versions differ in shape; this normalizes to (n, features) for class 1.
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def plot_confusion(cm, ax, labels=("consumer", "business"), title="confusion"):
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    return im
