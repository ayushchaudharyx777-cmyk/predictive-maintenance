import numpy as np
import pandas as pd

from features import FEATURES, MIN_HISTORY, add_labels, build_features


def _feats(raw, telemetry=None):
    tel = raw["telemetry"] if telemetry is None else telemetry
    return build_features(tel, raw["errors"], raw["maint"], raw["machines"])


def test_all_features_present_and_warmup_dropped(raw):
    f = _feats(raw)
    assert list(f.columns) == ["machineID", "datetime", *FEATURES]
    first = f.groupby("machineID")["datetime"].min() - raw["telemetry"].groupby("machineID")["datetime"].min()
    assert (first == pd.Timedelta(hours=MIN_HISTORY - 1)).all()
    assert not f[[c for c in FEATURES if "_mean_" in c or "_std_" in c or "_trend_" in c]].isna().any().any()


def test_no_future_leakage(raw):
    """Corrupt every reading after a cut-off: features at or before the cut-off must not change."""
    cut = raw["telemetry"]["datetime"].min() + pd.Timedelta(days=30)
    tampered = raw["telemetry"].copy()
    future = tampered["datetime"] > cut
    tampered.loc[future, ["volt", "rotate", "pressure", "vibration"]] *= 5
    a, b = _feats(raw), _feats(raw, tampered)
    a, b = a[a["datetime"] <= cut], b[b["datetime"] <= cut]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_rolling_mean_matches_manual(raw):
    f = _feats(raw)
    row = f.iloc[500]
    tel = raw["telemetry"]
    window = tel[(tel["machineID"] == row["machineID"]) & (tel["datetime"] <= row["datetime"])
                 & (tel["datetime"] > row["datetime"] - pd.Timedelta(hours=24))]
    assert len(window) == 24
    assert np.isclose(row["volt_mean_24h"], window["volt"].mean())


def test_labels(raw):
    horizon = 24
    df = add_labels(_feats(raw), raw["failures"], horizon)
    fail = raw["failures"].iloc[0]
    mine = df[df["machineID"] == fail["machineID"]].set_index("datetime")
    before = mine.loc[fail["datetime"] - pd.Timedelta(hours=5)]
    assert before["label"] == 1 and before["hours_to_failure"] == 5
    assert mine.loc[fail["datetime"] - pd.Timedelta(hours=horizon + 1), "label"] == 0
    # at the failure hour itself the failure has already happened: it is not a "future" failure
    assert mine.loc[fail["datetime"], "hours_to_failure"] != 0
    # rows too close to the end of the data cannot be labelled and are removed
    assert df["datetime"].max() <= raw["telemetry"]["datetime"].max() - pd.Timedelta(hours=horizon)


def test_days_since_maintenance_only_looks_back(raw):
    f = _feats(raw)
    assert (f[[c for c in FEATURES if c.startswith("days_since_")]].min() >= 0).all()
