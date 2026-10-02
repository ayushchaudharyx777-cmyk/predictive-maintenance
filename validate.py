"""Data checks that run before training. `python validate.py` exits 1 if anything is wrong."""
import sys

import pandas as pd

from config import data_dir
from features import COMPS, ERRORS, MODELS, SENSORS, load_raw

REQUIRED = {
    "telemetry": ["datetime", "machineID", *SENSORS],
    "errors": ["datetime", "machineID", "errorID"],
    "maint": ["datetime", "machineID", "comp"],
    "failures": ["datetime", "machineID", "failure"],
    "machines": ["machineID", "model", "age"],
}
ALLOWED = {("errors", "errorID"): ERRORS, ("maint", "comp"): COMPS, ("failures", "failure"): COMPS,
           ("machines", "model"): MODELS}


def validate(raw: dict) -> list[str]:
    problems = []
    for name, cols in REQUIRED.items():
        missing = [c for c in cols if c not in raw[name].columns]
        if missing:
            problems.append(f"{name}: missing columns {missing}")
    if problems:
        return problems

    tel = raw["telemetry"]
    if tel[SENSORS].isna().any().any():
        problems.append("telemetry: missing sensor values")
    if (tel[SENSORS] < 0).any().any():
        problems.append("telemetry: negative sensor values")
    if tel.duplicated(["machineID", "datetime"]).any():
        problems.append("telemetry: duplicate (machineID, datetime) rows")
    gaps = tel.sort_values(["machineID", "datetime"]).groupby("machineID")["datetime"].diff().dropna()
    n_gaps = int((gaps != pd.Timedelta(hours=1)).sum())
    if n_gaps:
        problems.append(f"telemetry: {n_gaps} gaps (rolling features assume one row per machine per hour)")

    known = set(raw["machines"]["machineID"])
    for name in ("telemetry", "errors", "maint", "failures"):
        unknown = set(raw[name]["machineID"]) - known
        if unknown:
            problems.append(f"{name}: machineIDs not in machines file: {sorted(unknown)[:5]}")
    for (name, col), allowed in ALLOWED.items():
        bad = set(raw[name][col].dropna()) - set(allowed)
        if bad:
            problems.append(f"{name}.{col}: unexpected values {sorted(bad)[:5]}")
    if raw["failures"].empty:
        problems.append("failures: no failure events, nothing to learn")
    return problems


if __name__ == "__main__":
    issues = validate(load_raw(data_dir()))
    if issues:
        print("VALIDATION FAILED:")
        for p in issues:
            print(" -", p)
        sys.exit(1)
    print("Validation passed.")
