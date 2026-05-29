"""Математическое обоснование проекта MDQ — статистические тесты, визуализации, анализ моделей.

Генерирует ВСЕ статистические доказательства для обоснования проекта
Hidden Entrepreneur Detection по карточным транзакциям.

Секции:
  1. Статистическая значимость признаков (Mann-Whitney U, effect size)
  2. Анализ распределений (KS-тест, KDE, univariate AUC)
  3. Корреляция и избыточность (матрица, высококоррелированные пары)
  4. Детальные метрики моделей (ROC, PR, bootstrap CI)
  5. Анализ порогов (P/R/F1, cost-sensitive)
  6. Калибровка (reliability diagram, Platt / isotonic)
  7. Анализ ошибок (FP/FN профили, ожидаемые потери)
  8. SHAP reason-codes для ВСЕХ leads

Запуск:
  .venv/Scripts/python.exe math_justification.py
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")                              # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    classification_report, brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import shap

from config import (SEED, FEAT_FILE, MODEL_FILE, LEADS_FILE,
                    ID_COL, LABEL_COL, LEAD_THR, TEST_SIZE)
from train_eval import load_xy, make_lgbm, make_logreg, make_rf
from mdq_utils import (ranking_metrics, metrics_at_threshold,
                       tune_thresholds, oof_proba, shap_positive_values)

# ── global plot settings ──────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

COL_CONSUMER = "#2196F3"
COL_BUSINESS = "#FF5722"
PLOT_DIR = os.path.join(os.path.dirname(__file__) or ".", "plots")

TOP6 = ["b2b_mcc_share", "b2b_amt_share", "b2b_unique_merchants",
        "recurring_share", "merchant_hhi", "amt_median"]

# Pretty Russian labels for the top-6 features
RU_LABELS = {
    "b2b_mcc_share":       "Доля B2B MCC",
    "b2b_amt_share":       "Доля B2B оборота",
    "b2b_unique_merchants":"B2B уник. мерчантов",
    "recurring_share":     "Доля рекуррентных",
    "merchant_hhi":        "HHI мерчантов",
    "amt_median":          "Медиана суммы (₸)",
}


def _ensure_dirs():
    os.makedirs(PLOT_DIR, exist_ok=True)


def _savefig(fig, name: str):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")


def _header(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def _load_data():
    """Load features + split into consumer / business arrays."""
    df, feats = load_xy()
    X = df[feats]
    y = df[LABEL_COL].values
    return df, feats, X, y


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Feature Statistical Significance
# ═══════════════════════════════════════════════════════════════════════════════
def section1_feature_significance(df=None, feats=None, X=None, y=None):
    _header("СЕКЦИЯ 1 — Статистическая значимость признаков")
    if df is None:
        df, feats, X, y = _load_data()

    con_mask = y == 0
    biz_mask = y == 1

    rows = []
    for f in feats:
        xc = X.loc[con_mask, f].dropna().values
        xb = X.loc[biz_mask, f].dropna().values
        # Mann-Whitney U
        stat, pval = stats.mannwhitneyu(xc, xb, alternative="two-sided")
        n1, n2 = len(xc), len(xb)
        # rank-biserial correlation r = 1 - 2U / (n1*n2)
        r_rb = 1.0 - 2.0 * stat / (n1 * n2)
        # Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * xc.std()**2 + (n2 - 1) * xb.std()**2) / (n1 + n2 - 2))
        d = (xb.mean() - xc.mean()) / pooled_std if pooled_std > 0 else 0.0
        rows.append({
            "feature": f,
            "consumer_median": np.median(xc),
            "business_median": np.median(xb),
            "consumer_mean": xc.mean(),
            "business_mean": xb.mean(),
            "U_stat": stat,
            "p_value": pval,
            "rank_biserial_r": r_rb,
            "cohens_d": d,
        })

    res = pd.DataFrame(rows).sort_values("p_value")
    print("\nТаблица: Mann-Whitney U, ранжировано по p-value")
    print(res.to_string(index=False, float_format="%.6g"))

    # ---- grouped bar chart: top-15 features by |Cohen's d| ----
    top15 = res.nlargest(15, "cohens_d", keep="first")
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(top15))
    w = 0.35
    ax.bar(x_pos - w/2, top15["consumer_median"].values, w,
           label="Потребительские", color=COL_CONSUMER, edgecolor="white")
    ax.bar(x_pos + w/2, top15["business_median"].values, w,
           label="Бизнес", color=COL_BUSINESS, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top15["feature"].values, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Медиана")
    ax.set_title("Медианы признаков: Потребительские vs Бизнес (топ-15 по Cohen's d)")
    ax.legend()
    _savefig(fig, "s1_grouped_bar_median.png")

    # ---- violin plots for top-6 features ----
    present = [f for f in TOP6 if f in feats]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, feat_name in zip(axes.flat, present):
        subset = pd.DataFrame({
            "value": X[feat_name].values,
            "class": np.where(y == 0, "Потребитель", "Бизнес"),
        })
        sns.violinplot(data=subset, x="class", y="value", ax=ax,
                       palette=[COL_CONSUMER, COL_BUSINESS], inner="quartile",
                       cut=0, linewidth=0.8)
        ax.set_title(RU_LABELS.get(feat_name, feat_name), fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.flat[len(present):]:
        ax.set_visible(False)
    fig.suptitle("Violin-графики топ-6 признаков", fontsize=14, y=1.01)
    fig.tight_layout()
    _savefig(fig, "s1_violin_top6.png")

    return res


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Feature Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def section2_distribution_analysis(df=None, feats=None, X=None, y=None):
    _header("СЕКЦИЯ 2 — Анализ распределений признаков")
    if df is None:
        df, feats, X, y = _load_data()

    con_mask = y == 0
    biz_mask = y == 1
    present = [f for f in TOP6 if f in feats]

    # ---- KS tests for top-6 ----
    print("\nKolmogorov-Smirnov тесты (top-6 признаков):")
    ks_rows = []
    for f in present:
        stat, pval = stats.ks_2samp(X.loc[con_mask, f].dropna().values,
                                    X.loc[biz_mask, f].dropna().values)
        ks_rows.append({"feature": f, "KS_stat": stat, "p_value": pval})
    ks_df = pd.DataFrame(ks_rows).sort_values("KS_stat", ascending=False)
    print(ks_df.to_string(index=False, float_format="%.6g"))

    # ---- KDE overlapping density plots ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, feat_name in zip(axes.flat, present):
        xc = X.loc[con_mask, feat_name].dropna()
        xb = X.loc[biz_mask, feat_name].dropna()
        xc.plot.kde(ax=ax, color=COL_CONSUMER, label="Потребитель", lw=2, bw_method=0.3)
        xb.plot.kde(ax=ax, color=COL_BUSINESS, label="Бизнес", lw=2, bw_method=0.3)
        ax.set_title(f"KDE: {RU_LABELS.get(feat_name, feat_name)}", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylabel("Плотность")
    for ax in axes.flat[len(present):]:
        ax.set_visible(False)
    fig.suptitle("Плотности распределений (KDE) — Потребитель vs Бизнес", fontsize=14, y=1.01)
    fig.tight_layout()
    _savefig(fig, "s2_kde_top6.png")

    # ---- univariate AUC for every feature ----
    print("\nUnivariate AUC (как хорошо один признак разделяет классы):")
    auc_rows = []
    for f in feats:
        vals = X[f].fillna(0).values
        auc = roc_auc_score(y, vals)
        auc = max(auc, 1 - auc)  # direction-invariant
        auc_rows.append({"feature": f, "univariate_AUC": auc})
    auc_df = pd.DataFrame(auc_rows).sort_values("univariate_AUC", ascending=False)
    print(auc_df.to_string(index=False, float_format="%.4f"))

    return ks_df, auc_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Correlation and Redundancy
# ═══════════════════════════════════════════════════════════════════════════════
def section3_correlation(df=None, feats=None, X=None, y=None):
    _header("СЕКЦИЯ 3 — Корреляция и избыточность признаков")
    if df is None:
        df, feats, X, y = _load_data()

    corr = X.corr()

    # ---- find highly correlated pairs ----
    pairs = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            r = corr.iloc[i, j]
            if abs(r) > 0.80:
                pairs.append((feats[i], feats[j], r))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    print(f"\nВысококоррелированные пары (|r| > 0.80): {len(pairs)}")
    for a, b, r in pairs:
        print(f"  {a:30s} <-> {b:30s}  r={r:+.4f}")

    # ---- rationale for keeping/dropping ----
    print("\nОбоснование (сохранены / удалены):")
    kept_rationale = {
        ("amt_sum", "amt_mean"):
            "amt_sum отражает объём, amt_mean — средний чек; оба информативны для модели.",
        ("amt_sum", "amt_max"):
            "amt_max ловит выбросы; amt_sum — общий оборот. Разная семантика.",
        ("gap_mean", "gap_std"):
            "Объединены через burstiness = (std-mean)/(std+mean); gap_cv уже удалён.",
        ("b2b_mcc_share", "b2b_amt_share"):
            "Доля по числу tx vs доля по сумме — дополняют друг друга (один в штуках, другой в ₸).",
        ("merchant_hhi", "merchant_top_ratio"):
            "HHI — общая концентрация, top_ratio — доминирование одного мерчанта. Оба нужны.",
    }
    for (a, b), reason in kept_rationale.items():
        if any(p[0] == a and p[1] == b for p in pairs) or any(p[0] == b and p[1] == a for p in pairs):
            print(f"  [{a} & {b}]: СОХРАНЕНЫ — {reason}")

    # ---- heatmap ----
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Корреляционная матрица признаков", fontsize=14)
    fig.tight_layout()
    _savefig(fig, "s3_correlation_heatmap.png")

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Model Metrics Detailed Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def section4_model_metrics(df=None, feats=None, X=None, y=None):
    _header("СЕКЦИЯ 4 — Детальные метрики моделей")
    if df is None:
        df, feats, X, y = _load_data()

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    models = {
        "LogReg": make_logreg(feats),
        "LightGBM": make_lgbm(),
        "RandomForest": make_rf(),
    }
    fitted = {}
    probas = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        probas[name] = m.predict_proba(Xva)[:, 1]
        fitted[name] = m

    # ---- ROC curves ----
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"LogReg": "#9C27B0", "LightGBM": COL_BUSINESS, "RandomForest": "#4CAF50"}
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(yva, p)
        auc = roc_auc_score(yva, p)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.4f})", color=colors[name], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR (ложноположительная доля)")
    ax.set_ylabel("TPR (чувствительность)")
    ax.set_title("ROC-кривые моделей")
    ax.legend(fontsize=10)
    _savefig(fig, "s4_roc_curves.png")

    # ---- PR curves ----
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(yva, p)
        ap = average_precision_score(yva, p)
        ax.plot(rec, prec, label=f"{name}  (AP={ap:.4f})", color=colors[name], lw=2)
    ax.set_xlabel("Полнота (Recall)")
    ax.set_ylabel("Точность (Precision)")
    ax.set_title("Precision-Recall кривые моделей")
    ax.legend(fontsize=10)
    _savefig(fig, "s4_pr_curves.png")

    # ---- classification report at multiple thresholds ----
    # Tune thresholds on OOF from training set
    oof = oof_proba(make_lgbm, Xtr, ytr)
    thr = tune_thresholds(ytr, oof)
    thr_f1 = thr["f1"]
    thr_recall = thr["recall"]

    pva_lgbm = probas["LightGBM"]
    thresholds_to_check = {
        "thr=0.30": 0.30,
        "thr=0.50": 0.50,
        f"F1-max (thr={thr_f1:.3f})": thr_f1,
        f"Recall>=0.95 (thr={thr_recall:.3f})": thr_recall,
    }
    print("\nClassification Reports @ разные пороги (LightGBM, validation set):")
    for label, t in thresholds_to_check.items():
        m = metrics_at_threshold(yva, pva_lgbm, t)
        print(f"\n--- {label} ---")
        print(f"  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}")
        yhat = (pva_lgbm >= t).astype(int)
        print(classification_report(yva, yhat,
              target_names=["потребитель", "бизнес"], digits=4))

    # ---- bootstrap confidence intervals for AUC ----
    n_boot = 100
    rng = np.random.RandomState(SEED)
    roc_boots = []
    pr_boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(yva), len(yva), replace=True)
        if len(np.unique(yva[idx])) < 2:
            continue
        roc_boots.append(roc_auc_score(yva[idx], pva_lgbm[idx]))
        pr_boots.append(average_precision_score(yva[idx], pva_lgbm[idx]))

    roc_boots = np.array(roc_boots)
    pr_boots = np.array(pr_boots)
    print(f"\nBootstrap 95% CI (n={n_boot}):")
    print(f"  ROC-AUC: {roc_boots.mean():.4f}  [{np.percentile(roc_boots, 2.5):.4f}, {np.percentile(roc_boots, 97.5):.4f}]")
    print(f"  PR-AUC : {pr_boots.mean():.4f}  [{np.percentile(pr_boots, 2.5):.4f}, {np.percentile(pr_boots, 97.5):.4f}]")

    return fitted, probas, thr


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Threshold Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def section5_threshold_analysis(df=None, feats=None, X=None, y=None,
                                fitted=None, probas=None, thr=None):
    _header("СЕКЦИЯ 5 — Анализ порогов")
    if df is None:
        df, feats, X, y = _load_data()

    # Re-train if needed
    if probas is None or thr is None:
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
        lgbm = make_lgbm()
        lgbm.fit(Xtr, ytr)
        pva = lgbm.predict_proba(Xva)[:, 1]
        oof = oof_proba(make_lgbm, Xtr, ytr)
        thr = tune_thresholds(ytr, oof)
    else:
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
        pva = probas["LightGBM"]

    thr_f1 = thr["f1"]
    thr_recall = thr["recall"]

    # ---- P/R/F1 vs threshold ----
    thresholds = np.linspace(0.01, 0.99, 200)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        yhat = (pva >= t).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precs.append(precision_score(yva, yhat, zero_division=0))
        recs.append(recall_score(yva, yhat, zero_division=0))
        f1s.append(f1_score(yva, yhat, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precs, label="Точность (Precision)", color=COL_CONSUMER, lw=2)
    ax.plot(thresholds, recs,  label="Полнота (Recall)",    color=COL_BUSINESS, lw=2)
    ax.plot(thresholds, f1s,   label="F1-мера",             color="#4CAF50", lw=2.5)
    # mark operating points
    ax.axvline(thr_f1, ls="--", color="#4CAF50", alpha=0.7, label=f"F1-max ({thr_f1:.3f})")
    ax.axvline(thr_recall, ls=":", color="#9C27B0", alpha=0.7, label=f"Recall≥0.95 ({thr_recall:.3f})")
    ax.set_xlabel("Порог (threshold)")
    ax.set_ylabel("Значение метрики")
    ax.set_title("Precision / Recall / F1 в зависимости от порога")
    ax.legend(fontsize=10)
    _savefig(fig, "s5_threshold_prf1.png")

    # ---- cost-sensitive threshold analysis ----
    print("\nCost-sensitive анализ: оптимальный порог для разных C_FP/C_FN:")
    cost_ratios = [(1, 1), (1, 5), (1, 10), (1, 20)]
    cost_rows = []
    for c_fp, c_fn in cost_ratios:
        best_t, best_cost = 0.5, np.inf
        for t in thresholds:
            yhat = (pva >= t).astype(int)
            fp = ((yhat == 1) & (yva == 0)).sum()
            fn = ((yhat == 0) & (yva == 1)).sum()
            cost = c_fp * fp + c_fn * fn
            if cost < best_cost:
                best_cost = cost
                best_t = t
        m = metrics_at_threshold(yva, pva, best_t)
        cost_rows.append({
            "C_FP:C_FN": f"{c_fp}:{c_fn}",
            "optimal_threshold": f"{best_t:.3f}",
            "precision": f"{m['precision']:.4f}",
            "recall": f"{m['recall']:.4f}",
            "f1": f"{m['f1']:.4f}",
            "total_cost": int(best_cost),
        })
    cost_df = pd.DataFrame(cost_rows)
    print(cost_df.to_string(index=False))

    return cost_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Calibration Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def section6_calibration(df=None, feats=None, X=None, y=None):
    _header("СЕКЦИЯ 6 — Анализ калибровки модели")
    if df is None:
        df, feats, X, y = _load_data()

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    # base LightGBM
    lgbm = make_lgbm()
    lgbm.fit(Xtr, ytr)
    p_raw = lgbm.predict_proba(Xva)[:, 1]

    # Platt scaling (sigmoid). Newer scikit-learn versions removed cv="prefit",
    # so calibration is fitted with internal CV on the training fold.
    cal_platt = CalibratedClassifierCV(make_lgbm(), method="sigmoid", cv=3)
    cal_platt.fit(Xtr, ytr)
    p_platt = cal_platt.predict_proba(Xva)[:, 1]

    # isotonic regression
    cal_iso = CalibratedClassifierCV(make_lgbm(), method="isotonic", cv=3)
    cal_iso.fit(Xtr, ytr)
    p_iso = cal_iso.predict_proba(Xva)[:, 1]

    # Brier scores
    brier_raw   = brier_score_loss(yva, p_raw)
    brier_platt = brier_score_loss(yva, p_platt)
    brier_iso   = brier_score_loss(yva, p_iso)
    print(f"\nBrier score (ниже — лучше):")
    print(f"  Без калибровки : {brier_raw:.6f}")
    print(f"  Platt (sigmoid): {brier_platt:.6f}")
    print(f"  Isotonic       : {brier_iso:.6f}")

    # ---- reliability diagram ----
    n_bins = 10
    fig, ax = plt.subplots(figsize=(8, 7))
    for preds, label, color, ls in [
        (p_raw,   f"Без калибровки (Brier={brier_raw:.4f})",   COL_BUSINESS, "-"),
        (p_platt, f"Platt (Brier={brier_platt:.4f})",          "#9C27B0",    "--"),
        (p_iso,   f"Isotonic (Brier={brier_iso:.4f})",         "#4CAF50",    "-."),
    ]:
        frac_pos, mean_pred = calibration_curve(yva, preds, n_bins=n_bins,
                                                 strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", label=label, color=color,
                lw=2, ls=ls)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Идеальная калибровка")
    ax.set_xlabel("Средняя предсказанная вероятность")
    ax.set_ylabel("Доля положительных (факт)")
    ax.set_title("Диаграмма надёжности (Reliability Diagram)")
    ax.legend(fontsize=9)
    _savefig(fig, "s6_calibration_curve.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Error Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def section7_error_analysis(df=None, feats=None, X=None, y=None):
    _header("СЕКЦИЯ 7 — Анализ ошибок модели")
    if df is None:
        df, feats, X, y = _load_data()

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    lgbm = make_lgbm()
    lgbm.fit(Xtr, ytr)
    pva = lgbm.predict_proba(Xva)[:, 1]

    oof = oof_proba(make_lgbm, Xtr, ytr)
    thr = tune_thresholds(ytr, oof)
    thr_f1 = thr["f1"]

    yhat = (pva >= thr_f1).astype(int)

    # classify errors
    tp_mask = (yhat == 1) & (yva == 1)
    tn_mask = (yhat == 0) & (yva == 0)
    fp_mask = (yhat == 1) & (yva == 0)
    fn_mask = (yhat == 0) & (yva == 1)

    Xva_df = Xva.copy()
    Xva_df = Xva_df.reset_index(drop=True)

    groups = {
        "True Positive": Xva_df.loc[tp_mask],
        "True Negative": Xva_df.loc[tn_mask],
        "False Positive": Xva_df.loc[fp_mask],
        "False Negative": Xva_df.loc[fn_mask],
    }

    print(f"\nПорог F1-max: {thr_f1:.3f}")
    print(f"  TP={tp_mask.sum()}  TN={tn_mask.sum()}  FP={fp_mask.sum()}  FN={fn_mask.sum()}")

    # feature profiles comparison
    profile_cols = [f for f in TOP6 if f in feats]
    extra = ["tx_count", "amt_sum", "online_share", "bizhours_share"]
    profile_cols += [c for c in extra if c in feats and c not in profile_cols]

    print(f"\nПрофили ошибок (медианы ключевых признаков):")
    profile = pd.DataFrame({
        name: grp[profile_cols].median() if len(grp) > 0 else pd.Series(dtype=float)
        for name, grp in groups.items()
    })
    print(profile.round(4).to_string())

    # ---- expected loss for different cost ratios ----
    print("\nОжидаемые потери при различных соотношениях C_FP:C_FN:")
    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    cost_ratios = [(1, 1), (1, 5), (1, 10), (1, 20)]
    loss_rows = []
    for c_fp, c_fn in cost_ratios:
        loss = c_fp * fp_count + c_fn * fn_count
        loss_rows.append({
            "C_FP:C_FN": f"{c_fp}:{c_fn}",
            "FP": fp_count, "FN": fn_count,
            "Expected_Loss": loss,
        })
    loss_df = pd.DataFrame(loss_rows)
    print(loss_df.to_string(index=False))

    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SHAP for All Leads
# ═══════════════════════════════════════════════════════════════════════════════
def section8_shap_leads(df=None, feats=None, X=None, y=None):
    _header("СЕКЦИЯ 8 — SHAP reason-codes для ВСЕХ hidden entrepreneur leads")
    if df is None:
        df, feats, X, y = _load_data()

    # compute OOF probabilities for all cards
    oof_p = oof_proba(make_lgbm, X, y)
    df_scored = df.copy()
    df_scored["oof_p"] = oof_p

    # filter consumers above lead threshold
    con = df_scored[df_scored[LABEL_COL] == 0].copy()
    leads = con[con["oof_p"] >= LEAD_THR].sort_values("oof_p", ascending=False)
    if len(leads) < 10:
        leads = con.sort_values("oof_p", ascending=False).head(25)
    print(f"\nВсего leads: {len(leads)}")

    # train full model for SHAP
    lgbm_full = make_lgbm()
    lgbm_full.fit(X, y)
    explainer = shap.TreeExplainer(lgbm_full)

    # compute SHAP for all leads
    leads_X = leads[feats]
    sv = shap_positive_values(explainer, leads_X)

    print(f"\nTop-3 SHAP причины для каждого lead:")
    print("-" * 90)
    for i, (idx, row) in enumerate(leads.iterrows()):
        shap_vals = pd.Series(sv[i], index=feats)
        top3 = shap_vals.abs().nlargest(3)
        reasons = []
        for feat_name in top3.index:
            val = shap_vals[feat_name]
            direction = "→ бизнес" if val > 0 else "→ потребитель"
            reasons.append(f"{feat_name}={row[feat_name]:.4g} (SHAP={val:+.4f} {direction})")
        card_id = row[ID_COL]
        prob = row["oof_p"]
        print(f"  Lead #{i+1:3d}  card={card_id}  P(business)={prob:.3f}")
        for r in reasons:
            print(f"         {r}")

    print("-" * 90)
    print(f"Всего leads с SHAP reason-codes: {len(leads)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — run all sections, save summary CSV
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    _ensure_dirs()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Загрузка данных ...")
    df, feats, X, y = _load_data()
    print(f"  samples={len(df):,}  features={len(feats)}  positive_rate={y.mean():.4f}")

    # ── Section 1 ──
    sig_df = section1_feature_significance(df, feats, X, y)

    # ── Section 2 ──
    ks_df, auc_df = section2_distribution_analysis(df, feats, X, y)

    # ── Section 3 ──
    corr_pairs = section3_correlation(df, feats, X, y)

    # ── Section 4 ──
    fitted, probas, thr = section4_model_metrics(df, feats, X, y)

    # ── Section 5 ──
    cost_df = section5_threshold_analysis(df, feats, X, y, fitted, probas, thr)

    # ── Section 6 ──
    section6_calibration(df, feats, X, y)

    # ── Section 7 ──
    section7_error_analysis(df, feats, X, y)

    # ── Section 8 ──
    section8_shap_leads(df, feats, X, y)

    # ── Save summary CSV ──
    _header("Сводная таблица статистических тестов → CSV")
    summary = sig_df[["feature", "p_value", "rank_biserial_r", "cohens_d"]].copy()
    summary = summary.merge(auc_df, on="feature", how="left")
    summary_path = os.path.join(PLOT_DIR, "statistical_tests_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  [saved] {summary_path}")

    _header("ГОТОВО — все секции выполнены")
    print(f"  Графики: {PLOT_DIR}")
    print(f"  Сводная CSV: {summary_path}")


if __name__ == "__main__":
    main()
