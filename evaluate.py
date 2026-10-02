"""Metrics. Two views: per row (every scored hour) and per event (each real failure, each false-alarm day)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def pick_threshold(y_true, risk, beta: float = 2.0) -> float:
    """Threshold with the best F-beta on the validation set (beta > 1 favours recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, risk)
    precision, recall = precision[:-1], recall[:-1]
    b2 = beta**2
    f = (1 + b2) * precision * recall / np.maximum(b2 * precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f))])


def row_metrics(y_true, risk, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    alert = np.asarray(risk) >= threshold
    tp = int((alert & (y_true == 1)).sum())
    fp = int((alert & (y_true == 0)).sum())
    fn = int((~alert & (y_true == 1)).sum())
    return {
        "pr_auc": float(average_precision_score(y_true, risk)),
        "roc_auc": float(roc_auc_score(y_true, risk)),
        "prevalence": float(y_true.mean()),   # PR-AUC of random guessing
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "true_alerts": tp,
        "false_alarms": fp,
        "missed": fn,
    }


def event_metrics(scored: pd.DataFrame, threshold: float) -> dict:
    """scored needs: machineID, datetime, risk, label, hours_to_failure.

    A failure counts as caught if at least one alert fired inside its warning window.
    A false-alarm day is a machine-day with an alert that was not followed by a failure within the window.
    """
    alert = scored["risk"] >= threshold
    pos = scored[scored["label"] == 1].assign(
        failure_time=lambda d: d["datetime"] + pd.to_timedelta(d["hours_to_failure"], unit="h"),
        alert=alert[scored["label"] == 1],
    )
    caught = pos.groupby(["machineID", "failure_time"])["alert"].any()
    false = scored[alert & (scored["label"] == 0)]
    fa_days = len(false.assign(day=false["datetime"].dt.date).drop_duplicates(["machineID", "day"]))
    weeks = max((scored["datetime"].max() - scored["datetime"].min()).total_seconds() / (7 * 86400), 1e-9)
    return {
        "failures": len(caught),
        "failures_caught": int(caught.sum()),
        "failure_recall": float(caught.mean()) if len(caught) else float("nan"),
        "false_alarm_days": int(fa_days),
        "false_alarm_days_per_week": fa_days / weeks,
        "machines": int(scored["machineID"].nunique()),
    }
