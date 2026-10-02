# Sanity check

Test PR-AUC, 24h window. Random guessing = 0.0182.

| Trained on | Test PR-AUC |
|---|---|
| All features | 0.9984 |
| Error counts only | 0.6861 |
| Maintenance history only | 0.2988 |
| Sensors only (raw, rolling, trend) | 0.2665 |
| Machine age + model only | 0.0243 |
| All features, **shuffled** training labels | 0.0145 |

Same pipeline with a shorter or longer warning window:

| Warning window | Positive rows | Test PR-AUC |
|---|---|---|
| 12h | 0.90% | 0.9633 |
| 24h | 1.82% | 0.9984 |
| 48h | 3.61% | 0.9606 |
| 72h | 5.40% | 0.8125 |
