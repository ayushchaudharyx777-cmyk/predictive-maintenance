import json

import pandas as pd

from features import add_labels, build_features
from train import time_split


def test_artifacts_and_model_beats_random(dirs):
    meta = json.loads((dirs / "models" / "meta.json").read_text())
    assert (dirs / "models" / "model.joblib").exists()
    assert (dirs / "reports" / "RESULTS.md").exists() and (dirs / "reports" / "pr_curve.png").exists()
    assert meta["synthetic_data"] is True
    rows = meta["test"]["rows"]
    assert rows["pr_auc"] > 5 * rows["prevalence"]
    assert 0 < meta["threshold"] < 1
    assert meta["pr_auc_ci"][0] <= rows["pr_auc"] <= meta["pr_auc_ci"][1] + 1e-9
    for name in ("component_model.joblib", "reference.csv.gz", "registry.json", "test_scores.csv.gz"):
        assert (dirs / "models" / name).exists()


def test_component_model_beats_the_overdue_rule(dirs):
    comp = json.loads((dirs / "models" / "meta.json").read_text())["component"]
    assert comp["accuracy"] > comp["baseline_accuracy"]


def test_policy_table_is_consistent(dirs):
    meta = json.loads((dirs / "models" / "meta.json").read_text())
    run_to_failure, _, model = meta["policy"]
    assert run_to_failure["saving_vs_run_to_failure"] == 0
    assert model["failures_caught"] + model["failures_missed"] == meta["test"]["events"]["failures"]
    assert model["saving_vs_run_to_failure"] > 0


def test_alerts_carry_component_and_reasons(dirs):
    meta = json.loads((dirs / "models" / "meta.json").read_text())
    scores = pd.read_csv(dirs / "models" / "test_scores.csv.gz", keep_default_na=False)
    alerts = scores[scores["risk"] >= meta["threshold"]]
    assert len(alerts) and (alerts["component"] != "").all()
    assert (alerts["reasons"] != "").mean() > 0.9


def test_split_is_chronological_with_gap(raw):
    horizon = 24
    df = add_labels(build_features(raw["telemetry"], raw["errors"], raw["maint"], raw["machines"]),
                    raw["failures"], horizon)
    train, valid, test, _ = time_split(df, 0.6, 0.2, horizon)
    gap = pd.Timedelta(hours=horizon)
    assert train["datetime"].max() + gap <= valid["datetime"].min()
    assert valid["datetime"].max() + gap <= test["datetime"].min()
    assert min(len(train), len(valid), len(test)) > 0
