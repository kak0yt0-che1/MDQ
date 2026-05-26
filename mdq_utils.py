"""Shared helpers used by both the scripts and the notebook, so metric,
threshold, out-of-fold, and SHAP logic lives in exactly one place.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

from config import SEED, N_SPLITS


def ranking_metrics(y, p) -> dict:
    """Threshold-independent quality of a probability ranking."""
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p)}


def metrics_at_threshold(y, p, thr: float) -> dict:
    """Point metrics + confusion matrix at a given probability cutoff."""
    yhat = (p >= thr).astype(int)
    return {"threshold": float(thr),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0),
            "confusion": confusion_matrix(y, yhat)}


def tune_thresholds(y, proba, min_recall: float = 0.95) -> dict:
    """Pick two operating points from a precision/recall sweep:
    - 'f1'     : threshold maximising F1 (balanced) -> use for auto-decisions
    - 'recall' : highest-precision threshold with recall >= min_recall
                 -> use for outreach lists (a miss costs more than a false alarm)
    Tune this on TRAIN / out-of-fold scores, never on the held-out set.
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    thr_f1 = float(thr[np.nanargmax(f1s[:-1])])
    mask = rec[:-1] >= min_recall
    thr_recall = float(thr[mask][np.argmax(prec[:-1][mask])]) if mask.any() else thr_f1
    return {"f1": thr_f1, "recall": thr_recall}


def oof_proba(make_estimator, X, y, seed: int = SEED, n_splits: int = N_SPLITS):
    """Honest out-of-fold P(class=1): every row scored by a model not trained on it.
    `make_estimator` is a zero-arg factory so each fold gets a fresh, unfitted model.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(make_estimator(), X, y, cv=skf,
                             method="predict_proba", n_jobs=-1)[:, 1]


def shap_positive_values(explainer, X) -> np.ndarray:
    """Return the positive-class SHAP matrix, tolerant of the list-vs-array
    output difference across SHAP / LightGBM versions."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # [class0, class1]
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, classes)
        return sv[:, :, 1]
    return sv


def plot_confusion(cm, ax, labels=("consumer", "business"), title="confusion"):
    """Annotated confusion-matrix heatmap on a provided matplotlib axis."""
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    return im
