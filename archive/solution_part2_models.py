"""
=============================================================================
PART 2: MODEL TRAINING, VALIDATION & INTERPRETATION
Hidden Entrepreneur Detection — MDQ Competition
=============================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

DATA_DIR = r"c:\Users\olzha\OneDrive\Рабочий_стол\MDQ"
SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. LOAD FEATURES
# ============================================================
print("Loading features...")
features_df = pd.read_parquet(f"{DATA_DIR}/features_df.parquet")
print(f"Shape: {features_df.shape}")
print(f"Class balance: {features_df['label'].value_counts().to_dict()}")

# Feature columns (exclude card_number and label)
feature_cols = [c for c in features_df.columns if c not in ["card_number", "label"]]
print(f"Number of features: {len(feature_cols)}")

X = features_df[feature_cols].values
y = features_df["label"].values

# Handle any NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================
# 2. TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Train class balance: {np.bincount(y_train)}")
print(f"Test class balance:  {np.bincount(y_test)}")

# ============================================================
# 3. BASELINE — LOGISTIC REGRESSION
# ============================================================
print("\n" + "="*60)
print("MODEL 1: Logistic Regression (Baseline)")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=["Consumer", "Business"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_prob_lr):.4f}")

# ============================================================
# 4. CATBOOST
# ============================================================
print("\n" + "="*60)
print("MODEL 2: CatBoost")
print("="*60)

try:
    from catboost import CatBoostClassifier
    
    cb = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=SEED,
        verbose=100,
        eval_metric="F1",
        auto_class_weights="Balanced",
        early_stopping_rounds=50,
    )
    cb.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)
    y_pred_cb = cb.predict(X_test)
    y_prob_cb = cb.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_cb, target_names=["Consumer", "Business"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_cb):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_prob_cb):.4f}")
    HAS_CATBOOST = True
except ImportError:
    print("CatBoost not installed, skipping...")
    HAS_CATBOOST = False

# ============================================================
# 5. LIGHTGBM
# ============================================================
print("\n" + "="*60)
print("MODEL 3: LightGBM")
print("="*60)

try:
    import lightgbm as lgb
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        is_unbalance=True,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    y_pred_lgb = lgb_model.predict(X_test)
    y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lgb, target_names=["Consumer", "Business"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lgb):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_prob_lgb):.4f}")
    HAS_LGBM = True
except ImportError:
    print("LightGBM not installed, skipping...")
    HAS_LGBM = False

# ============================================================
# 6. RANDOM FOREST
# ============================================================
print("\n" + "="*60)
print("MODEL 4: Random Forest")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=500, max_depth=12, min_samples_leaf=10,
    class_weight="balanced", random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Consumer", "Business"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_prob_rf):.4f}")

# ============================================================
# 7. MODEL COMPARISON TABLE
# ============================================================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

results = []
models_data = [("Logistic Regression", y_pred_lr, y_prob_lr)]
if HAS_CATBOOST:
    models_data.append(("CatBoost", y_pred_cb, y_prob_cb))
if HAS_LGBM:
    models_data.append(("LightGBM", y_pred_lgb, y_prob_lgb))
models_data.append(("Random Forest", y_pred_rf, y_prob_rf))

for name, y_pred, y_prob in models_data:
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format="%.4f"))
results_df.to_csv(f"{DATA_DIR}/model_comparison.csv", index=False)

# ============================================================
# 8. SELECT BEST MODEL & THRESHOLD TUNING
# ============================================================
# Select best model by F1
best_idx = results_df["F1"].idxmax()
best_name = results_df.loc[best_idx, "Model"]
print(f"\nBest model by F1: {best_name}")

# Use best model probabilities for threshold tuning
if best_name == "CatBoost" and HAS_CATBOOST:
    y_prob_best = y_prob_cb
    best_model = cb
elif best_name == "LightGBM" and HAS_LGBM:
    y_prob_best = y_prob_lgb
    best_model = lgb_model
elif best_name == "Random Forest":
    y_prob_best = y_prob_rf
    best_model = rf
else:
    y_prob_best = y_prob_lr
    best_model = lr

print("\n--- Threshold Tuning ---")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_best)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
print(f"Optimal threshold: {best_threshold:.4f}")
print(f"At this threshold -> Precision: {precisions[best_threshold_idx]:.4f}, Recall: {recalls[best_threshold_idx]:.4f}, F1: {f1_scores[best_threshold_idx]:.4f}")

y_pred_tuned = (y_prob_best >= best_threshold).astype(int)
print("\nClassification Report (tuned threshold):")
print(classification_report(y_test, y_pred_tuned, target_names=["Consumer", "Business"]))

# ============================================================
# 9. VISUALIZATIONS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Hidden Entrepreneur Detection — Model Results", fontsize=16, fontweight="bold")

# 9a. Confusion Matrix (default threshold)
ax = axes[0, 0]
cm = confusion_matrix(y_test, (y_prob_best >= 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Consumer", "Business"], yticklabels=["Consumer", "Business"])
ax.set_title(f"Confusion Matrix\n({best_name}, threshold=0.5)")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")

# 9b. Confusion Matrix (tuned threshold)
ax = axes[0, 1]
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Greens", ax=ax,
            xticklabels=["Consumer", "Business"], yticklabels=["Consumer", "Business"])
ax.set_title(f"Confusion Matrix\n({best_name}, threshold={best_threshold:.3f})")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")

# 9c. ROC Curves
ax = axes[0, 2]
for name, _, y_prob in models_data:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_title("ROC Curves")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8)

# 9d. Precision-Recall Curve
ax = axes[1, 0]
for name, _, y_prob in models_data:
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
ax.set_title("Precision-Recall Curves")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(fontsize=8)

# 9e. F1 vs Threshold
ax = axes[1, 1]
ax.plot(thresholds, f1_scores[:-1], "b-", linewidth=2)
ax.axvline(best_threshold, color="r", linestyle="--", label=f"Best: {best_threshold:.3f}")
ax.set_title("F1 Score vs Threshold")
ax.set_xlabel("Threshold")
ax.set_ylabel("F1 Score")
ax.legend()

# 9f. Feature Importance (top 20)
ax = axes[1, 2]
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "get_feature_importance"):
    importances = best_model.get_feature_importance()
else:
    importances = np.abs(best_model.coef_[0])

fi = pd.Series(importances, index=feature_cols).sort_values(ascending=True).tail(20)
fi.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title(f"Top 20 Feature Importance\n({best_name})")
ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlots saved to {DATA_DIR}/model_results.png")

# ============================================================
# 10. SHAP ANALYSIS
# ============================================================
print("\n" + "="*60)
print("SHAP ANALYSIS")
print("="*60)

try:
    import shap
    
    if HAS_CATBOOST and best_name == "CatBoost":
        explainer = shap.TreeExplainer(best_model)
    elif HAS_LGBM and best_name == "LightGBM":
        explainer = shap.TreeExplainer(best_model)
    elif best_name == "Random Forest":
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.LinearExplainer(best_model, X_train_scaled)
    
    # Use a sample for SHAP (full dataset too slow)
    sample_idx = np.random.choice(len(X_test), min(1000, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]
    
    shap_values = explainer.shap_values(X_sample)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # class 1 (business)
    else:
        shap_vals = shap_values
    
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_cols, show=False, max_display=20)
    plt.title("SHAP Summary Plot — Feature Impact on Business Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"SHAP plot saved to {DATA_DIR}/shap_summary.png")
    
except ImportError:
    print("SHAP not installed. Install with: pip install shap")
except Exception as e:
    print(f"SHAP error: {e}")

# ============================================================
# 11. FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Best Model: {best_name}")
print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"Test Set Metrics (tuned threshold):")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_tuned):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred_tuned):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob_best):.4f}")

print("\nConfusion Matrix (tuned):")
print(cm_tuned)
print(f"\nTop 10 Features:")
fi_sorted = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
for i, (feat, imp) in enumerate(fi_sorted.head(10).items()):
    print(f"  {i+1}. {feat}: {imp:.4f}")

print("\n✅ Done! All results saved.")
