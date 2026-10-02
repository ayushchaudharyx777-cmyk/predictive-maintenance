# Results

Question: will this machine fail in the next **24 hours**?
Split by time: train until 2015-08-08, validation until 2015-10-19, test after.
Rows: 173000 / 57100 / 58000.
Alert threshold 0.50, chosen on validation by `cost`.

Test period = the last ~20% of the timeline, never seen during training or threshold selection.
Model version `20261003-050220`. 

| Metric (test period) | Value |
|---|---|
| Selected model | Gradient Boosting (lr=0.1, leaves=15) |
| PR-AUC | 0.9993, 95% CI [0.9976, 1.0000] (random guessing = 0.018) |
| Recall / Precision at the alert threshold (0.50) | 100.0% / 99.2% |
| Failures caught | 133 of 133, median warning 24h ahead |
| False-alarm days | 0.4 per week across 100 machines |
| Which component will fail (top-1) | 100.0% of 1054 warning rows (most-overdue-component rule: 51.1%) |

**Cost of each policy over the test period** (ASSUMED: breakdown 10,000, planned repair
2,500, inspection after a false alarm 500):

| Policy | Caught | Missed | False-alarm days | Cost | Saving vs run-to-failure |
|---|---|---|---|---|---|
| Run to failure (no alerts) | 0 | 133 | 0 | 1,330,000 | 0 (0%) |
| Rule: alert on any error in last 24h | 133 | 0 | 979 | 822,000 | 508,000 (38%) |
| Model (threshold 0.50) | 133 | 0 | 4 | 334,500 | 995,500 (75%) |

The model's alerts stay worthwhile until one false-alarm inspection costs more than 249,375.
Model saving: **995,500** (75%).

![PR curve](pr_curve.png)

## Row-level detail

| Metric | Model | Rule: alert on any error in last 24h |
|---|---|---|
| ROC-AUC | 1.0000 | - |
| Recall | 100.0% | 100.0% |
| Precision | 99.2% | 19.9% |
| False alarms (rows) | 9 | 4254 |

## Candidates (validation PR-AUC)

| Model | PR-AUC |
|---|---|
| Logistic Regression | 0.8259 |
| Gradient Boosting (lr=0.1, leaves=31) | 0.9996 |
| Gradient Boosting (lr=0.05, leaves=31) | 0.9997 |
| Gradient Boosting (lr=0.1, leaves=15) | 0.9997 |
| Gradient Boosting (lr=0.05, leaves=63) | 0.9994 |

## Top features (permutation importance, drop in PR-AUC)

| Feature | Importance |
|---|---|
| error4_count_24h | 0.0670 |
| error5_count_24h | 0.0499 |
| error1_count_24h | 0.0324 |
| vibration_mean_24h | 0.0206 |
| days_since_comp3 | 0.0142 |
| volt_mean_24h | 0.0125 |
| error2_count_24h | 0.0096 |
| days_since_comp4 | 0.0079 |
| error3_count_24h | 0.0073 |
| days_since_comp1 | 0.0067 |
