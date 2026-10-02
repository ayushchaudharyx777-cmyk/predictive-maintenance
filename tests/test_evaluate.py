import pandas as pd

from evaluate import event_metrics, pick_threshold, row_metrics


def _scored():
    t = pd.Timestamp("2015-06-01 00:00:00")
    rows = [
        # machine 1 fails at t+6h: two warning rows, one of them alerts -> caught
        (1, t, 0.9, 1, 6.0), (1, t + pd.Timedelta(hours=3), 0.2, 1, 3.0),
        # machine 2 fails at t+6h: no alert -> missed
        (2, t, 0.1, 1, 6.0),
        # machine 3 healthy: two alerts on the same day -> one false-alarm day
        (3, t, 0.8, 0, None), (3, t + pd.Timedelta(hours=3), 0.7, 0, None),
        (3, t + pd.Timedelta(days=7), 0.1, 0, None),
    ]
    return pd.DataFrame(rows, columns=["machineID", "datetime", "risk", "label", "hours_to_failure"])


def test_event_metrics():
    m = event_metrics(_scored(), threshold=0.5)
    assert m["failures"] == 2 and m["failures_caught"] == 1
    assert m["false_alarm_days"] == 1
    assert abs(m["false_alarm_days_per_week"] - 1.0) < 1e-6


def test_row_metrics():
    s = _scored()
    m = row_metrics(s["label"], s["risk"], 0.5)
    assert (m["true_alerts"], m["false_alarms"], m["missed"]) == (1, 2, 2)
    assert abs(m["precision"] - 1 / 3) < 1e-9 and abs(m["recall"] - 1 / 3) < 1e-9


def test_pick_threshold_separable():
    thr = pick_threshold([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9])
    assert 0.3 < thr <= 0.8
