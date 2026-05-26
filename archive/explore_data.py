import pandas as pd
import numpy as np

# Load data
biz = pd.read_parquet(r"c:\Users\olzha\OneDrive\Рабочий_стол\MDQ\business_cards_MDQ.parquet")
con = pd.read_parquet(r"c:\Users\olzha\OneDrive\Рабочий_стол\MDQ\consumer_cards_MDQ.parquet")
mer = pd.read_parquet(r"c:\Users\olzha\OneDrive\Рабочий_стол\MDQ\merchants_reference.parquet")

print("=== UNIQUE CARDHOLDERS ===")
print(f"Business unique cards: {biz['card_number'].nunique()}")
print(f"Consumer unique cards: {con['card_number'].nunique()}")

print("\n=== CARD TIER DISTRIBUTION ===")
print("Business tiers:")
print(biz["card_tier"].value_counts())
print("\nConsumer tiers:")
print(con["card_tier"].value_counts())

print("\n=== CHANNEL VALUES ===")
print("Business channels:")
print(biz["channel"].value_counts())
print("\nConsumer channels:")
print(con["channel"].value_counts())

print("\n=== COUNTRY VALUES ===")
print("Business countries:")
print(biz["country"].value_counts().head(15))
print("\nConsumer countries:")
print(con["country"].value_counts().head(15))

print("\n=== TOKENIZED ===")
print("Business tokenized:")
print(biz["tokenized"].value_counts())
print("\nConsumer tokenized:")
print(con["tokenized"].value_counts())

print("\n=== IS_RECURRING ===")
print("Business:")
print(biz["is_recurring"].value_counts())
print("\nConsumer:")
print(con["is_recurring"].value_counts())

print("\n=== BANKNAME ===")
print("Business banks:")
print(biz["bank_name"].value_counts().head(15))
print("\nConsumer banks:")
print(con["bank_name"].value_counts().head(15))

print("\n=== DATE RANGE ===")
print(f"Business: {biz['transaction_date'].min()} to {biz['transaction_date'].max()}")
print(f"Consumer: {con['transaction_date'].min()} to {con['transaction_date'].max()}")

print("\n=== MCC in business (top 20) ===")
print(biz["mcc"].value_counts().head(20))
print("\n=== MCC in consumer (top 20) ===")
print(con["mcc"].value_counts().head(20))

# B2B merchants
print("\n=== ALL MERCHANTS ===")
for idx, row in mer.iterrows():
    print(f"  {row['merchant_id']} | {row['merchant_name']} | MCC: {row['mcc']} | {row['merchant_country']} | Recurring: {row['recurring_capable']}")

print("\n=== ALL UNIQUE MCC CODES (sorted) ===")
all_mcc_biz = set(biz["mcc"].unique())
all_mcc_con = set(con["mcc"].unique())
all_mcc = sorted(all_mcc_biz | all_mcc_con)
print(f"Total unique MCC: {len(all_mcc)}")
print(all_mcc)
print(f"\nMCC only in business: {sorted(all_mcc_biz - all_mcc_con)}")
print(f"MCC only in consumer: {sorted(all_mcc_con - all_mcc_biz)}")

# Overlap check: same card_number in both?
biz_cards = set(biz["card_number"].unique())
con_cards = set(con["card_number"].unique())
overlap = biz_cards & con_cards
print(f"\n=== CARD OVERLAP ===")
print(f"Business cards: {len(biz_cards)}")
print(f"Consumer cards: {len(con_cards)}")
print(f"Overlap: {len(overlap)}")

# Merchant usage by label
print("\n=== TOP MERCHANTS IN BUSINESS ===")
biz_merchants = biz.groupby("merchant_id")["transaction_amount_kzt"].agg(["count","sum"]).sort_values("sum", ascending=False).head(20)
biz_merchants = biz_merchants.merge(mer[["merchant_id","merchant_name"]], left_index=True, right_on="merchant_id", how="left")
print(biz_merchants.to_string())

print("\n=== TOP MERCHANTS IN CONSUMER ===")
con_merchants = con.groupby("merchant_id")["transaction_amount_kzt"].agg(["count","sum"]).sort_values("sum", ascending=False).head(20)
con_merchants = con_merchants.merge(mer[["merchant_id","merchant_name"]], left_index=True, right_on="merchant_id", how="left")
print(con_merchants.to_string())
