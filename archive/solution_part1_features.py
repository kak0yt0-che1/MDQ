"""
PART 1 (ULTRA-FAST): Fully vectorized feature engineering.
No .apply(), no lambdas — pure pandas groupby.agg() and merges.
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
import warnings, time
warnings.filterwarnings("ignore")

t0 = time.time()
DATA_DIR = r"c:\Users\olzha\OneDrive\Рабочий_стол\MDQ"

# ===== LOAD =====
print("Loading data...")
biz = pd.read_parquet(f"{DATA_DIR}/business_cards_MDQ.parquet")
con = pd.read_parquet(f"{DATA_DIR}/consumer_cards_MDQ.parquet")
mer = pd.read_parquet(f"{DATA_DIR}/merchants_reference.parquet")
print(f"Biz: {biz.shape[0]:,} | Con: {con.shape[0]:,} | Loaded in {time.time()-t0:.1f}s")

# ===== LABEL & COMBINE =====
biz["label"] = 1
con["label"] = 0
df = pd.concat([biz, con], ignore_index=True)
del biz, con  # free RAM

# ===== PREPROCESS =====
print("Preprocessing...")
df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
df["hour"] = df["transaction_timestamp"].dt.hour
df["dayofweek"] = df["transaction_timestamp"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
df["is_bh"] = ((df["hour"] >= 9) & (df["hour"] < 18)).astype(np.int8)
df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(np.int8)
df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(np.int8)
df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(np.int8)
df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] < 24)).astype(np.int8)
df["month"] = df["transaction_timestamp"].dt.month
df["date_int"] = df["transaction_timestamp"].dt.date.apply(lambda x: x.toordinal())
df["week"] = df["transaction_timestamp"].dt.isocalendar().week.astype(int)
df["is_international"] = (df["country"] != "Kazakhstan").astype(np.int8)
df["is_online"] = (df["channel"] == "online").astype(np.int8)

# B2B flags
B2B_MCC = {"7311","7372","7379","4816","5045","5046","7392","8999","5065","5044","7399","5111","7375","4814","7333","7338"}
B2B_KW = ["google ads","meta ads","tiktok ads","yandex direct","linkedin ads",
    "instagram promote","amazon web services","microsoft azure","google cloud",
    "salesforce","hubspot","atlassian","shopify","zoom","slack",
    "digitalocean","cloudflare","godaddy","hetzner","dropbox","mailchimp","stripe","notion","figma","canva pro","github"]
mer["is_b2b"] = ((mer["mcc"].isin(B2B_MCC)) | (mer["merchant_name"].str.lower().apply(lambda x: any(k in str(x) for k in B2B_KW)))).astype(np.int8)
print(f"B2B merchants: {mer['is_b2b'].sum()}/{len(mer)}")

df = df.merge(mer[["merchant_id","is_b2b","recurring_capable"]], on="merchant_id", how="left")
df["is_b2b"] = df["is_b2b"].fillna(0).astype(np.int8)
df["recurring_capable"] = df["recurring_capable"].fillna(False).astype(np.int8)
df["b2b_amount"] = df["transaction_amount_kzt"] * df["is_b2b"]
df["is_recurring_int"] = df["is_recurring"].astype(np.int8)
df["tokenized_int"] = df["tokenized"].astype(np.int8)

print(f"Preprocessing done in {time.time()-t0:.1f}s")

# ===== FEATURE ENGINEERING (all via groupby.agg) =====
print("Building features...")

# --- BLOCK 1: Simple agg on numeric/flag columns ---
agg1 = df.groupby("card_number").agg(
    label=("label", "first"),
    txn_count=("transaction_amount_kzt", "count"),
    total_spend=("transaction_amount_kzt", "sum"),
    avg_amount=("transaction_amount_kzt", "mean"),
    median_amount=("transaction_amount_kzt", "median"),
    std_amount=("transaction_amount_kzt", "std"),
    min_amount=("transaction_amount_kzt", "min"),
    max_amount=("transaction_amount_kzt", "max"),
    q25_amount=("transaction_amount_kzt", lambda x: x.quantile(0.25)),
    q75_amount=("transaction_amount_kzt", lambda x: x.quantile(0.75)),
    unique_merchants=("merchant_id", "nunique"),
    unique_mcc=("mcc", "nunique"),
    unique_countries=("country", "nunique"),
    unique_banks=("bank_name", "nunique"),
    b2b_txn_count=("is_b2b", "sum"),
    b2b_txn_share=("is_b2b", "mean"),
    b2b_spend=("b2b_amount", "sum"),
    recurring_share=("is_recurring_int", "mean"),
    recurring_capable_share=("recurring_capable", "mean"),
    tokenized_share=("tokenized_int", "mean"),
    online_share=("is_online", "mean"),
    weekend_share=("is_weekend", "mean"),
    business_hours_share=("is_bh", "mean"),
    night_share=("is_night", "mean"),
    morning_share=("is_morning", "mean"),
    afternoon_share=("is_afternoon", "mean"),
    evening_share=("is_evening", "mean"),
    international_share=("is_international", "mean"),
    active_days=("date_int", "nunique"),
    active_weeks=("week", "nunique"),
)
print(f"  Block 1 done in {time.time()-t0:.1f}s")

features = agg1.copy()
features["std_amount"] = features["std_amount"].fillna(0)
features["cv_amount"] = features["std_amount"] / features["avg_amount"].replace(0, np.nan)
features["cv_amount"] = features["cv_amount"].fillna(0)
features["iqr_amount"] = features["q75_amount"] - features["q25_amount"]
features["log_total_spend"] = np.log1p(features["total_spend"])
features["log_avg_amount"] = np.log1p(features["avg_amount"])
features["b2b_spend_share"] = features["b2b_spend"] / features["total_spend"].replace(0, np.nan)
features["b2b_spend_share"] = features["b2b_spend_share"].fillna(0)
features["pos_share"] = 1 - features["online_share"]
features["txn_per_active_day"] = features["txn_count"] / features["active_days"].replace(0, 1)

# --- BLOCK 2: Entropy and concentration (need custom agg) ---
print("  Computing entropy & concentration (chunked)...")

# Pre-compute merchant counts per card
mc = df.groupby(["card_number","merchant_id"]).size().reset_index(name="cnt")
mc_total = mc.groupby("card_number")["cnt"].transform("sum")
mc["share"] = mc["cnt"] / mc_total

# Merchant entropy
mer_entropy = mc.groupby("card_number").apply(lambda g: scipy_entropy(g["share"]))
features["merchant_entropy"] = mer_entropy
print(f"  Merchant entropy done in {time.time()-t0:.1f}s")

# MCC counts per card
mcc_c = df.groupby(["card_number","mcc"]).size().reset_index(name="cnt")
mcc_total = mcc_c.groupby("card_number")["cnt"].transform("sum")
mcc_c["share"] = mcc_c["cnt"] / mcc_total

# MCC entropy
mcc_entropy = mcc_c.groupby("card_number").apply(lambda g: scipy_entropy(g["share"]))
features["mcc_entropy"] = mcc_entropy
print(f"  MCC entropy done in {time.time()-t0:.1f}s")

# Top-N merchant concentration
mc_sorted = mc.sort_values(["card_number","cnt"], ascending=[True, False])

def topn_conc(n):
    top = mc_sorted.groupby("card_number").head(n)
    return top.groupby("card_number")["share"].sum()

features["top1_merchant_conc"] = topn_conc(1)
features["top3_merchant_conc"] = topn_conc(3)
features["top5_merchant_conc"] = topn_conc(5)
print(f"  Merchant concentration done in {time.time()-t0:.1f}s")

# Top-N MCC concentration
mcc_sorted = mcc_c.sort_values(["card_number","cnt"], ascending=[True, False])

def topn_mcc_conc(n):
    top = mcc_sorted.groupby("card_number").head(n)
    return top.groupby("card_number")["share"].sum()

features["top1_mcc_conc"] = topn_mcc_conc(1)
features["top3_mcc_conc"] = topn_mcc_conc(3)
print(f"  MCC concentration done in {time.time()-t0:.1f}s")

# HHI
features["merchant_hhi"] = mc.groupby("card_number").apply(lambda g: (g["share"]**2).sum())
print(f"  HHI done in {time.time()-t0:.1f}s")

# --- BLOCK 3: B2B unique merchants ---
b2b_uniq = df[df["is_b2b"]==1].groupby("card_number")["merchant_id"].nunique()
features["b2b_unique_merchants"] = b2b_uniq.reindex(features.index).fillna(0).astype(int)
b2b_avg = df[df["is_b2b"]==1].groupby("card_number")["transaction_amount_kzt"].mean()
features["b2b_avg_amount"] = b2b_avg.reindex(features.index).fillna(0)

# --- BLOCK 4: Monthly stability ---
print("  Computing monthly stability...")
ms = df.groupby(["card_number","month"])["transaction_amount_kzt"].agg(["sum","count"])
ms_stats = ms.groupby(level=0).agg(
    monthly_spend_mean=("sum","mean"),
    monthly_spend_std=("sum","std"),
    monthly_count_mean=("count","mean"),
    monthly_count_std=("count","std"),
)
ms_stats = ms_stats.fillna(0)
features["monthly_spend_cv"] = ms_stats["monthly_spend_std"] / ms_stats["monthly_spend_mean"].replace(0, np.nan)
features["monthly_spend_cv"] = features["monthly_spend_cv"].fillna(0)
features["monthly_count_cv"] = ms_stats["monthly_count_std"] / ms_stats["monthly_count_mean"].replace(0, np.nan)
features["monthly_count_cv"] = features["monthly_count_cv"].fillna(0)
print(f"  Monthly stability done in {time.time()-t0:.1f}s")

# --- BLOCK 5: Burstiness (simplified - just mean interval) ---
print("  Computing intervals...")
# Sort by card and timestamp, compute diff within card
df_sorted = df[["card_number","transaction_timestamp"]].sort_values(["card_number","transaction_timestamp"])
df_sorted["prev_ts"] = df_sorted.groupby("card_number")["transaction_timestamp"].shift(1)
df_sorted["interval_s"] = (df_sorted["transaction_timestamp"] - df_sorted["prev_ts"]).dt.total_seconds()
df_sorted = df_sorted.dropna(subset=["interval_s"])

interval_stats = df_sorted.groupby("card_number")["interval_s"].agg(["mean","std"])
interval_stats = interval_stats.fillna(0)
features["mean_interval_hours"] = interval_stats["mean"] / 3600
features["std_interval_hours"] = interval_stats["std"] / 3600
denom = interval_stats["std"] + interval_stats["mean"]
features["burstiness"] = ((interval_stats["std"] - interval_stats["mean"]) / denom.replace(0, np.nan)).fillna(0)
print(f"  Intervals done in {time.time()-t0:.1f}s")

# --- BLOCK 6: Day-of-week shares ---
features["monday_share"] = df[df["dayofweek"]==0].groupby("card_number").size() / features["txn_count"]
features["friday_share"] = df[df["dayofweek"]==4].groupby("card_number").size() / features["txn_count"]
features["monday_share"] = features["monday_share"].fillna(0)
features["friday_share"] = features["friday_share"].fillna(0)

# ===== CLEANUP & SAVE =====
features = features.reset_index()
features = features.fillna(0).replace([np.inf, -np.inf], 0)

feat_cols = [c for c in features.columns if c not in ["card_number","label"]]
print(f"\nFeature matrix: {features.shape[0]} rows x {len(feat_cols)} features")
print(f"Class: {features['label'].value_counts().to_dict()}")
print(f"Total time: {time.time()-t0:.1f}s")

features.to_parquet(f"{DATA_DIR}/features_df.parquet", index=False)
print(f"Saved to features_df.parquet")
print(f"Feature names: {sorted(feat_cols)}")
