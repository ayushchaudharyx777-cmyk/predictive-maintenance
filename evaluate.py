"""Metrics. Two views: per row (every scored hour) and per event (each real failure, each false-alarm day)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from policy import total_cost


def fbeta_threshold(y_true, risk, beta: float = 2.0) -> float:
    """Threshold with the best F-beta (beta > 1 favours recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, risk)
    precision, recall = precision[:-1], recall[:-1]
    b2 = beta**2
    f = (1 + b2) * precision * recall / np.maximum(b2 * precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f))])


def cost_threshold(scored: pd.DataFrame, costs: dict) -> float:
    """Cheapest threshold on a grid. Thresholds within 1% of the best cost count as tied, and the middle one
    of those is taken: a threshold picked from a tiny cost difference at the edge of the grid would not be robust."""
    grid = np.round(np.linspace(0.05, 0.95, 19), 2)
    cost = np.array([total_cost(event_metrics(scored, t), costs) for t in grid])
    tied = grid[cost <= cost.min() * 1.01]
    return float(tied[len(tied) // 2])


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
    Lead time = how long before the failure the first alert inside the window fired.
    """
    alert = scored["risk"] >= threshold
    pos = scored[scored["label"] == 1].assign(
        failure_time=lambda d: d["datetime"] + pd.to_timedelta(d["hours_to_failure"], unit="h"),
        alert=alert[scored["label"] == 1],
    )
    caught = pos.groupby(["machineID", "failure_time"])["alert"].any()
    lead = pos[pos["alert"]].groupby(["machineID", "failure_time"])["hours_to_failure"].max()
    false = scored[alert & (scored["label"] == 0)]
    fa_days = len(false.assign(day=false["datetime"].dt.date).drop_duplicates(["machineID", "day"]))
    weeks = max((scored["datetime"].max() - scored["datetime"].min()).total_seconds() / (7 * 86400), 1e-9)
    return {
        "failures": len(caught),
        "failures_caught": int(caught.sum()),
        "failure_recall": float(caught.mean()) if len(caught) else float("nan"),
        "median_lead_hours": float(lead.median()) if len(lead) else float("nan"),
        "false_alarm_days": int(fa_days),
        "false_alarm_days_per_week": fa_days / weeks,
        "machines": int(scored["machineID"].nunique()),
    }


def bootstrap_pr_auc(scored: pd.DataFrame, n: int = 200, seed: int = 0) -> list[float]:
    """95% interval for PR-AUC, resampling whole machines (rows of one machine are not independent)."""
    groups = [g[["label", "risk"]].to_numpy() for _, g in scored.groupby("machineID")]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        arr = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if arr[:, 0].sum() > 0:
            vals.append(average_precision_score(arr[:, 0], arr[:, 1]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def alert_episodes(scored: pd.DataFrame, threshold: float, max_gap_hours: int = 6) -> pd.DataFrame:
    """Group consecutive alert rows of a machine into one episode and say how each one ended."""
    a = scored[scored["risk"] >= threshold].sort_values(["machineID", "datetime"]).copy()
    cols = ["machineID", "start", "end", "peak_risk", "component", "reasons", "outcome", "lead_hours"]
    if a.empty:
        return pd.DataFrame(columns=cols)
    gap = a.groupby("machineID")["datetime"].diff()
    a["episode"] = (gap.isna() | (gap > pd.Timedelta(hours=max_gap_hours))).cumsum()
    a["failure_time"] = (a["datetime"] + pd.to_timedelta(a["hours_to_failure"], unit="h")).where(a["label"] == 1)
    for col in ("component", "reasons"):
        if col not in a:
            a[col] = ""
    ep = a.groupby("episode").agg(
        machineID=("machineID", "first"), start=("datetime", "min"), end=("datetime", "max"),
        peak_risk=("risk", "max"), component=("component", "first"), reasons=("reasons", "first"),
        failure_time=("failure_time", "min"),
    ).reset_index(drop=True)
    ep["lead_hours"] = (ep["failure_time"] - ep["start"]).dt.total_seconds() / 3600
    ep["outcome"] = np.where(ep["failure_time"].notna(), "Failure followed", "False alarm")
    return ep.sort_values("start", ascending=False)[cols].reset_index(drop=True)
