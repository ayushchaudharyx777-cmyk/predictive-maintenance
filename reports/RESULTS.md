# Results (version 20261003-043053)

Question: will this machine fail in the next **24 hours**?
Split by time: train until 2015-08-08, validation until 2015-10-19, test after.
Rows: 173000 / 57100 / 58000.

## Model selection (validation PR-AUC)

| Model | PR-AUC |
|---|---|
| Logistic Regression | 0.8259 |
| Gradient Boosting | 0.9996 |

Selected: **Gradient Boosting**. Alert threshold 0.454 (best F2 on validation).

## Test period

| Metric | Model | Baseline: alert on any error in last 24h |
|---|---|---|
| PR-AUC (random = 0.018) | 0.9984 | - |
| ROC-AUC | 1.0000 | - |
| Recall (rows) | 99.2% | 100.0% |
| Precision (rows) | 98.7% | 19.9% |
| False alarms (rows) | 14 | 4254 |
| Failures caught | 133 of 133 | 133 of 133 |
| False-alarm days per week (100 machines) | 0.6 | 94.7 |

## Top features (permutation importance, drop in PR-AUC)

| Feature | Importance |
|---|---|
| error4_count_24h | 0.1121 |
| error5_count_24h | 0.0907 |
| error1_count_24h | 0.0533 |
| vibration_mean_24h | 0.0422 |
| volt_mean_24h | 0.0336 |
| days_since_comp1 | 0.0281 |
| rotate_mean_24h | 0.0212 |
| error3_count_24h | 0.0171 |
| error2_count_24h | 0.0159 |
| days_since_comp3 | 0.0142 |

![PR curve](pr_curve.png)
