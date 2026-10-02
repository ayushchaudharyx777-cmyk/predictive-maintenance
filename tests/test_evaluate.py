import pandas as pd

from evaluate import alert_episodes, bootstrap_pr_auc, cost_threshold, event_metrics, fbeta_threshold, row_metrics


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


def test_fbeta_threshold_separable():
    thr = fbeta_threshold([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9])
    assert 0.3 < thr <= 0.8


def test_lead_time_is_first_alert_in_window():
    assert event_metrics(_scored(), threshold=0.5)["median_lead_hours"] == 6.0


def test_cost_threshold_avoids_false_alarms_and_misses():
    costs = {"unplanned_failure": 10000, "planned_repair": 2500, "inspection": 500}
    s = _scored()
    s.loc[s["machineID"] == 2, "risk"] = 0.95      # now both failures are detectable above 0.85
    thr = cost_threshold(s, costs)
    m = event_metrics(s, thr)
    assert m["failures_caught"] == 2 and m["false_alarm_days"] == 0


def test_alert_episodes_group_and_label():
    ep = alert_episodes(_scored(), threshold=0.5)
    assert len(ep) == 2                              # machine 1 (one alert row), machine 3 (two rows, one episode)
    by_machine = ep.set_index("machineID")
    assert by_machine.loc[1, "outcome"] == "Failure followed" and by_machine.loc[1, "lead_hours"] == 6
    assert by_machine.loc[3, "outcome"] == "False alarm"


def test_bootstrap_interval_contains_point_estimate():
    s = _scored()
    lo, hi = bootstrap_pr_auc(s, n=50)
    assert 0 <= lo <= hi <= 1
