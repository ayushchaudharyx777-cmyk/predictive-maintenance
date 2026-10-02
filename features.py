"""Feature engineering and labelling.

Rule that everything here follows: a row at time t may only use information recorded at or before t.
The label is the only thing that looks forward.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["volt", "rotate", "pressure", "vibration"]
ERRORS = [f"error{i}" for i in range(1, 6)]
COMPS = [f"comp{i}" for i in range(1, 5)]
MODELS = [f"model{i}" for i in range(1, 5)]
WINDOWS = (3, 24)          # rolling windows, in hours
MIN_HISTORY = 48           # hours of telemetry needed before the first usable row (24h mean + 24h lag)

FEATURES = (
    SENSORS
    + [f"{c}_{stat}_{w}h" for c in SENSORS for w in WINDOWS for stat in ("mean", "std")]
    + [f"{c}_trend_24h" for c in SENSORS]
    + [f"{e}_count_24h" for e in ERRORS]
    + [f"days_since_{c}" for c in COMPS]
    + ["age"]
    + [f"model_{m}" for m in MODELS]
)

FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "maint": "PdM_maint.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
}


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).astype("datetime64[ns]")


def load_raw(data_dir) -> dict:
    data_dir = Path(data_dir)
    missing = [f for f in FILES.values() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {data_dir}: {missing}. See README (Data).")
    raw = {name: pd.read_csv(data_dir / f) for name, f in FILES.items()}
    for name in ("telemetry", "errors", "maint", "failures"):
        raw[name]["datetime"] = _dt(raw[name]["datetime"])
    return raw


def build_features(telemetry, errors, maint, machines) -> pd.DataFrame:
    """One row per machine per hour. Expects hourly telemetry without gaps (validate.py checks this)."""
    df = telemetry[["datetime", "machineID", *SENSORS]].copy()
    df["datetime"] = _dt(df["datetime"])
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    # 1. rolling statistics: the window ends at the current hour, so only past + present readings
    by = df.groupby("machineID", sort=False)
    new = {}
    for c in SENSORS:
        for w in WINDOWS:
            roll = by[c].rolling(w, min_periods=w)
            new[f"{c}_mean_{w}h"] = roll.mean().reset_index(level=0, drop=True)
            new[f"{c}_std_{w}h"] = roll.std().reset_index(level=0, drop=True)
    df = df.assign(**new)

    # 2. lag feature: today's 24h average minus yesterday's 24h average = is the sensor drifting?
    by = df.groupby("machineID", sort=False)
    for c in SENSORS:
        df[f"{c}_trend_24h"] = df[f"{c}_mean_24h"] - by[f"{c}_mean_24h"].shift(24)

    # 3. error history: how many errors of each type in the last 24 hours
    e = errors[["datetime", "machineID", "errorID"]].copy()
    e["datetime"] = _dt(e["datetime"])
    if len(e):
        counts = (
            e.assign(n=1)
            .pivot_table(index=["machineID", "datetime"], columns="errorID", values="n", aggfunc="sum", fill_value=0)
            .reindex(columns=ERRORS, fill_value=0)
            .reset_index()
        )
        df = df.merge(counts, on=["machineID", "datetime"], how="left")
        df[ERRORS] = df[ERRORS].fillna(0)
    else:
        df[ERRORS] = 0.0
    by = df.groupby("machineID", sort=False)
    for er in ERRORS:
        df[f"{er}_count_24h"] = by[er].rolling(24, min_periods=1).sum().reset_index(level=0, drop=True)
    df = df.drop(columns=ERRORS)

    # 4. maintenance history: days since each component was last replaced (backward lookup only)
    m = maint[["datetime", "machineID", "comp"]].copy()
    m["datetime"] = _dt(m["datetime"])
    left = df[["datetime", "machineID"]].reset_index().sort_values("datetime")
    for comp in COMPS:
        mc = m.loc[m["comp"] == comp, ["datetime", "machineID"]].rename(columns={"datetime": "last"}).sort_values("last")
        if mc.empty:
            df[f"days_since_{comp}"] = np.nan
            continue
        j = pd.merge_asof(left, mc, left_on="datetime", right_on="last", by="machineID", direction="backward")
        days = (j["datetime"] - j["last"]).dt.total_seconds() / 86400
        df[f"days_since_{comp}"] = pd.Series(days.to_numpy(), index=j["index"].to_numpy())

    # 5. static machine info
    df = df.merge(machines[["machineID", "model", "age"]], on="machineID", how="left")
    for mo in MODELS:
        df[f"model_{mo}"] = (df["model"] == mo).astype(int)
    df = df.drop(columns="model")

    # the first 47 hours of every machine do not have a full window yet
    df = df.dropna(subset=[f"{SENSORS[0]}_trend_24h"]).reset_index(drop=True)
    return df[["machineID", "datetime", *FEATURES]]


def add_labels(feats: pd.DataFrame, failures: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """label = 1 if the machine fails within the next `horizon_hours` (strictly after the row's time).

    failing_comps = component(s) of that next failure, e.g. "comp2" or "comp1+comp3" (two can fail together).
    """
    f = failures[["datetime", "machineID", "failure"]].copy()
    f["datetime"] = _dt(f["datetime"])
    f = (f.sort_values("failure").groupby(["machineID", "datetime"])["failure"].agg("+".join).reset_index()
         .rename(columns={"datetime": "next_failure", "failure": "failing_comps"}).sort_values("next_failure"))
    left = feats[["datetime", "machineID"]].reset_index().sort_values("datetime")
    j = pd.merge_asof(
        left, f, left_on="datetime", right_on="next_failure", by="machineID",
        direction="forward", allow_exact_matches=False,
    )
    hours = (j["next_failure"] - j["datetime"]).dt.total_seconds() / 3600
    out = feats.copy()
    out["hours_to_failure"] = pd.Series(hours.to_numpy(), index=j["index"].to_numpy())
    out["failing_comps"] = pd.Series(j["failing_comps"].to_numpy(), index=j["index"].to_numpy())
    out["label"] = (out["hours_to_failure"] <= horizon_hours).astype(int)
    # the last `horizon` hours of the data have an unknown future -> cannot be labelled honestly
    cutoff = out["datetime"].max() - pd.Timedelta(hours=horizon_hours)
    return out[out["datetime"] <= cutoff].reset_index(drop=True)
