"""Card-level feature engineering for hidden-entrepreneur detection.

Aggregates raw transactions (one row per transaction) into one row per card.
Importable from the notebook:  from features import build_card_features, load_labeled
Or run standalone to materialize features_card_level.parquet:
    .venv/Scripts/python.exe features.py

Leakage policy (see EDA): card_tier is dropped (perfect giveaway); raw mcc /
merchant_id are NOT one-hot encoded (class-exclusive values are coverage
artifacts). B2B exposure is measured via a *domain-curated* MCC list below, not
via which codes happen to be business-exclusive in this sample.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import (CONSUMER_FILE, BUSINESS_FILE, MERCHANT_FILE, FEAT_FILE,
                    B2B_MCC, ID_COL, LABEL_COL)


def load_labeled() -> pd.DataFrame:
    """Load consumer+business transactions into one frame with a binary label.
    label = 1 -> business card, 0 -> consumer card."""
    con = pd.read_parquet(CONSUMER_FILE)
    biz = pd.read_parquet(BUSINESS_FILE)
    con[LABEL_COL] = 0
    biz[LABEL_COL] = 1
    df = pd.concat([con, biz], ignore_index=True)
    return df


def _add_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-transaction helper columns used by the aggregations."""
    ts = pd.to_datetime(df["transaction_timestamp"])
    df["_hour"] = ts.dt.hour.astype("int16")
    dow = ts.dt.dayofweek.astype("int16")
    df["_is_weekend"] = (dow >= 5)
    df["_is_evening"] = df["_hour"].between(18, 23)
    df["_is_bizhours"] = (~df["_is_weekend"]) & df["_hour"].between(9, 17)
    df["_is_online"] = (df["channel"] == "online")
    df["_is_foreign"] = (df["country"] != "Kazakhstan")
    df["_is_b2b_mcc"] = df["mcc"].isin(B2B_MCC)
    df["_amt"] = df["transaction_amount_kzt"].astype("float64")
    df["_date"] = ts.dt.normalize()
    # calendar week as an absolute year-week period so distinct weeks never
    # collide across the 2025->2026 year boundary (ISO weeknum alone would).
    df["_week"] = ts.dt.to_period("W").astype("string")
    df["_month"] = ts.dt.to_period("M").astype("string")
    return df


def _dist_features(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Concentration (HHI), top-1 share, and entropy of a card's distribution over
    `key` (e.g. merchant_id), computed from per-(card,key) tx counts."""
    g = df.groupby([ID_COL, key], observed=True).size().rename("n")
    total = g.groupby(level=0, observed=True).transform("sum")
    p = g / total
    return pd.DataFrame({
        f"{prefix}_hhi": (p * p).groupby(level=0, observed=True).sum(),
        f"{prefix}_top_ratio": p.groupby(level=0, observed=True).max(),
        f"{prefix}_entropy": (-(p * np.log(p))).groupby(level=0, observed=True).sum(),
    })


def build_card_features(df: pd.DataFrame, merchants: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate a labeled transaction frame into one row per card_number."""
    df = _add_helpers(df)
    obs_end = df["_date"].max()  # global end of observation window

    # merchant-reference join: does the card transact at recurring-capable
    # merchants? (distinct from is_recurring, which is per-transaction)
    if merchants is None:
        merchants = pd.read_parquet(MERCHANT_FILE)
    rec_cap = merchants.drop_duplicates("merchant_id").set_index("merchant_id")["recurring_capable"]
    df["_rec_capable"] = df["merchant_id"].map(rec_cap).fillna(False)

    grp = df.groupby("card_number", observed=True)
    feat = grp.agg(
        label=("label", "first"),
        tx_count=("_amt", "size"),
        amt_sum=("_amt", "sum"),
        amt_mean=("_amt", "mean"),
        amt_median=("_amt", "median"),
        amt_std=("_amt", "std"),
        amt_max=("_amt", "max"),
        amt_min=("_amt", "min"),
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

    # B2B amount share (share of spend, not just count, in B2B MCCs)
    feat["b2b_amt_share"] = (
        df.assign(_b2b_amt=df["_amt"].where(df["_is_b2b_mcc"], 0.0))
          .groupby("card_number", observed=True)["_b2b_amt"].sum()
        / feat["amt_sum"]
    )

    # derived ratios / cadence
    feat["amt_cv"] = feat["amt_std"] / feat["amt_mean"].replace(0, np.nan)
    feat["span_days"] = (feat["last_date"] - feat["first_date"]).dt.days + 1
    feat["tx_per_active_day"] = feat["tx_count"] / feat["active_days"]
    feat["recency_days"] = (obs_end - feat["last_date"]).dt.days
    feat["merchants_per_tx"] = feat["n_unique_merchants"] / feat["tx_count"]

    # supplier concentration / diversity (merchant-level only; mcc-level was an
    # r~0.99 duplicate so it is intentionally dropped).
    feat = feat.join(_dist_features(df, "merchant_id", "merchant"))

    # B2B supplier breadth: how many distinct B2B merchants the card uses
    b2b_u = (df[df["_is_b2b_mcc"]].groupby("card_number", observed=True)["merchant_id"]
             .nunique())
    feat["b2b_unique_merchants"] = b2b_u.reindex(feat.index).fillna(0).astype("int64")

    # month-to-month stability: CV of monthly spend (monthly_count_cv dropped as
    # near-random, AUC~0.56).
    ms = df.groupby(["card_number", "_month"], observed=True)["_amt"].sum()
    msc = ms.groupby(level=0, observed=True).agg(sp_mean="mean", sp_std="std")
    feat["monthly_spend_cv"] = msc["sp_std"] / msc["sp_mean"].replace(0, np.nan)

    # burstiness from inter-arrival gaps (hours between consecutive tx)
    d = df[["card_number", "transaction_timestamp"]].sort_values(
        ["card_number", "transaction_timestamp"])
    gap_h = (d.groupby("card_number", observed=True)["transaction_timestamp"]
               .diff().dt.total_seconds() / 3600.0)
    d = d.assign(_gap=gap_h)
    gstat = d.groupby("card_number", observed=True)["_gap"].agg(
        gap_mean="mean", gap_std="std")
    # Goh-Barabasi burstiness in [-1, 1]: >0 bursty, <0 regular (gap_cv was an
    # r~0.99 duplicate of this, so it is dropped).
    gstat["burstiness"] = ((gstat["gap_std"] - gstat["gap_mean"]) /
                           (gstat["gap_std"] + gstat["gap_mean"]).replace(0, np.nan))
    feat = feat.join(gstat)

    # tidy: drop raw date helpers, fill structural NaNs (single-tx cards have no gap/std)
    feat = feat.drop(columns=["first_date", "last_date"])
    feat = feat.fillna({
        "amt_std": 0.0, "amt_cv": 0.0,
        "gap_mean": 0.0, "gap_std": 0.0, "burstiness": 0.0,
        "monthly_spend_cv": 0.0,  # single-month cards
    })
    return feat.reset_index()


# Leakage-excluded columns kept out of the model on purpose (see EDA).
LEAKY_OR_EXCLUDED = ["card_tier", "bank_name"]


def main():
    print("loading labeled transactions ...")
    df = load_labeled()
    print(f"  rows={len(df):,}  cards={df[ID_COL].nunique():,}")
    print("building card-level features ...")
    feat = build_card_features(df)
    feat.to_parquet(FEAT_FILE, index=False)
    print(f"saved {FEAT_FILE}  shape={feat.shape}")
    print(f"positive rate (business): {feat[LABEL_COL].mean():.4f}")
    print("\nfeature columns:")
    print([c for c in feat.columns if c not in (ID_COL, LABEL_COL)])
    print("\nnull counts:")
    print(feat.isna().sum().loc[lambda s: s > 0].to_string() or "  none")


if __name__ == "__main__":
    main()
