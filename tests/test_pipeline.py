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


def test_split_is_chronological_with_gap(raw):
    horizon = 24
    df = add_labels(build_features(raw["telemetry"], raw["errors"], raw["maint"], raw["machines"]),
                    raw["failures"], horizon)
    train, valid, test, _ = time_split(df, 0.6, 0.2, horizon)
    gap = pd.Timedelta(hours=horizon)
    assert train["datetime"].max() + gap <= valid["datetime"].min()
    assert valid["datetime"].max() + gap <= test["datetime"].min()
    assert min(len(train), len(valid), len(test)) > 0
