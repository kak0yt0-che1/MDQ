"""Central configuration: one place for the seed, file paths, and the B2B MCC list.
Every other module imports from here so paths/seeds are never duplicated.
"""
from __future__ import annotations

SEED = 42

# ---- data / artifact paths (relative to the project root) ----
CONSUMER_FILE = "consumer_cards_MDQ.parquet"
BUSINESS_FILE = "business_cards_MDQ.parquet"
MERCHANT_FILE = "merchants_reference.parquet"
FEAT_FILE = "features_card_level.parquet"
MODEL_FILE = "model_lgbm.joblib"
LEADS_FILE = "hidden_entrepreneurs.csv"

# ---- column roles ----
ID_COL = "card_number"
LABEL_COL = "label"

# ---- scoring ----
LEAD_THR = 0.30          # P(business) cutoff for an actionable review lead
N_SPLITS = 5             # folds for out-of-fold scoring / threshold tuning
TEST_SIZE = 0.20         # validation fraction

# Domain-curated B2B / wholesale / professional-services MCCs.
# Chosen from MCC *semantics*, not this dataset's class frequencies.
# NOTE: 5122 (wholesale drugs) was deliberately removed - it is business-exclusive
# in this sample, so including it would leak the label into b2b_*_share. See README.
B2B_MCC = {
    # wholesale / distribution
    "5044", "5045", "5046", "5047", "5051", "5065", "5072", "5074", "5085",
    "5111", "5131", "5137", "5139", "5169", "5172", "5192", "5198", "5199",
    # professional / business services
    "7311", "7321", "7333", "7338", "7339", "7342", "7349", "7361", "7372",
    "7375", "7379", "7392", "7393", "7394", "7399", "8742", "8911", "8931",
    "2741", "2791", "2842",
    # freight / logistics / network services
    "4214", "4215", "4225", "4816",
}
