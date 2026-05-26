# Archived: earlier monolithic solution

These files are an earlier, parallel solution kept for reference only. The project
consolidated on the modular pipeline at the repo root (`config.py`, `features.py`,
`mdq_utils.py`, `train_eval.py`, `score_consumers.py`, `notebook.ipynb`).

Why it was superseded (not wrong, just less complete / less rigorous):
- **No hidden-entrepreneur step** — it classifies business-vs-consumer and reports
  test metrics, but never scores the consumer pool out-of-fold to surface leads,
  which is the actual business goal.
- **Threshold tuned on the test set** (`precision_recall_curve(y_test, ...)`) — optimistic.
- **Early stopping uses `eval_set = X_test`** — the test set leaks into tree-count selection.
- **B2B list mixed in consumer-heavy MCCs** (e.g. `4814` telecom) plus brittle merchant-name keywords.

Its genuinely useful features were ported into the root `features.py`:
`recurring_capable_share`, `monthly_spend_cv`, `b2b_unique_merchants`.
