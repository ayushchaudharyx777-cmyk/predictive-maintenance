# Sanity check

Test PR-AUC. Random guessing = 0.0182.

| Trained on | Test PR-AUC |
|---|---|
| All features | 0.9984 |
| Sensors only (raw, rolling, trend) | 0.2665 |
| Error counts only | 0.6861 |
| Maintenance history only | 0.2988 |
| Machine age + model only | 0.0243 |
| All features, SHUFFLED training labels | 0.0145 |
