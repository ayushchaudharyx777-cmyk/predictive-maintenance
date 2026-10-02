"""Policy maths, explanations, drift monitor, quality gate."""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

import monitor
import policy
from explain import GROUPS, reasons
from features import FEATURES

ROOT = Path(__file__).parent.parent
COSTS = {"unplanned_failure": 10000, "planned_repair": 2500, "inspection": 500}


def test_policy_costs():
    ev = {"failures": 10, "failures_caught": 8, "false_alarm_days": 4}
    assert policy.total_cost(ev, COSTS) == 8 * 2500 + 2 * 10000 + 4 * 500
    rows = policy.compare({"none": policy.no_alerts(ev), "model": ev}, COSTS)
    assert rows[0]["cost"] == 100000 and rows[0]["saving_vs_run_to_failure"] == 0
    assert rows[1]["saving_vs_run_to_failure"] == 100000 - 42000
    assert policy.breakeven_inspection_cost(ev, COSTS) == 8 * 7500 / 4
    assert policy.breakeven_inspection_cost(dict(ev, false_alarm_days=0), COSTS) == float("inf")


def test_every_feature_belongs_to_at_most_one_group():
    grouped = [f for cols in GROUPS.values() for f in cols]
    assert len(grouped) == len(set(grouped)) and set(grouped) <= set(FEATURES)


def test_reasons_are_ranked_and_readable(dirs):
    model = joblib.load(dirs / "models" / "model.joblib")
    meta = json.loads((dirs / "models" / "meta.json").read_text())
    ref = pd.read_csv(dirs / "models" / "reference.csv.gz")
    out = reasons(model, ref.head(50), meta["baseline"])
    assert len(out) == 50
    for r in out:
        assert len(r) <= 3 and [x["risk_drop"] for x in r] == sorted((x["risk_drop"] for x in r), reverse=True)
        assert all(isinstance(x["text"], str) and x["risk_drop"] >= 0.02 for x in r)


def test_psi_zero_for_same_data_and_large_for_shift():
    s = pd.Series(range(1000), dtype=float)
    assert monitor.psi(s, s) < 0.01
    assert monitor.psi(s, s + 500) > 0.25


def test_monitor_alerts_on_simulated_drift_only(dirs):
    ref = dirs / "models" / "reference.csv.gz"
    same = dirs / "same.csv"
    pd.read_csv(ref).to_csv(same, index=False)
    assert monitor.main(["--current", str(same)]) == 0
    assert monitor.main(["--simulate"]) == 1
    report = json.loads((dirs / "reports" / "drift_report.json").read_text())
    assert report["status"] == "ALERT" and report["features"][0]["feature"].startswith(("pressure", "vibration"))


def test_monitor_ignores_tiny_batches(dirs):
    tiny = dirs / "tiny.csv"
    pd.read_csv(dirs / "models" / "reference.csv.gz").head(5).to_csv(tiny, index=False)
    assert monitor.main(["--current", str(tiny)]) == 0


def _gate(env_extra):
    import os

    env = dict(os.environ, **env_extra)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "quality_gate.py")], env=env,
                          capture_output=True, text=True, check=False)


def test_quality_gate_passes_and_blocks(dirs):
    ok = _gate({"GATE_MIN_PR_AUC": "0.0", "GATE_MIN_FAILURE_RECALL": "0.0"})
    assert ok.returncode == 0, ok.stdout + ok.stderr
    blocked = _gate({"GATE_MIN_PR_AUC": "1.1"})
    assert blocked.returncode == 1 and "QUALITY GATE FAILED" in blocked.stdout
