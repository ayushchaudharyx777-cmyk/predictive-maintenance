"""Writes small SYNTHETIC files with the same columns as the Azure PdM dataset, for tests and CI.

Never report numbers from this data. Usage: python scripts/make_synthetic_data.py [out_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = {"volt": (170, 15), "rotate": (446, 52), "pressure": (100, 11), "vibration": (40, 5)}
# which sensor drifts before which component fails, and in which direction
SIGNATURE = {"comp1": ("volt", 1), "comp2": ("rotate", -1), "comp3": ("pressure", 1), "comp4": ("vibration", 1)}


def make(out_dir, n_machines: int = 20, days: int = 150, seed: int = 0) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "PdM_telemetry.csv").exists() and not (out_dir / "SYNTHETIC").exists():
        raise SystemExit(f"{out_dir} holds real data - refusing to overwrite it.")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2015-01-01 06:00:00")
    hours = pd.date_range(start, periods=days * 24, freq="h")
    comps = list(SIGNATURE)
    tel, err, maint, fail, mach = [], [], [], [], []
    for mid in range(1, n_machines + 1):
        mach.append((mid, f"model{rng.integers(1, 5)}", int(rng.integers(0, 21))))
        data = {s: rng.normal(mu, sd, len(hours)) for s, (mu, sd) in BASE.items()}
        for comp in comps:  # every component was replaced at some point before the data starts
            maint.append((start - pd.Timedelta(days=int(rng.integers(10, 200))), mid, comp))
        for day in rng.choice(np.arange(6, days - 1, 8), size=rng.integers(2, 5), replace=False):
            i, comp = int(day) * 24, rng.choice(comps)
            sensor, sign = SIGNATURE[comp]
            data[sensor][i - 36:i] += sign * BASE[sensor][1] * np.linspace(0.5, 2.5, 36)
            for _ in range(rng.integers(1, 4)):
                err.append((hours[i - int(rng.integers(1, 48))], mid, f"error{comps.index(comp) + 1}"))
            fail.append((hours[i], mid, comp))
            maint.append((hours[i], mid, comp))
        for _ in range(days // 15):  # noise: errors and scheduled maintenance unrelated to failures
            err.append((hours[int(rng.integers(0, len(hours)))], mid, f"error{rng.integers(1, 6)}"))
        for day in range(15, days, 30):
            maint.append((hours[day * 24], mid, rng.choice(comps)))
        tel.append(pd.DataFrame({"datetime": hours, "machineID": mid, **data}))

    def save(rows, cols, name):
        df = pd.DataFrame(rows, columns=cols).sort_values(cols[:2]).drop_duplicates()
        df.to_csv(out_dir / name, index=False, date_format="%Y-%m-%d %H:%M:%S")

    pd.concat(tel).to_csv(out_dir / "PdM_telemetry.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    save(err, ["datetime", "machineID", "errorID"], "PdM_errors.csv")
    save(maint, ["datetime", "machineID", "comp"], "PdM_maint.csv")
    save(fail, ["datetime", "machineID", "failure"], "PdM_failures.csv")
    pd.DataFrame(mach, columns=["machineID", "model", "age"]).to_csv(out_dir / "PdM_machines.csv", index=False)
    (out_dir / "SYNTHETIC").write_text("These files are synthetic. Do not report results from them.\n")
    print(f"wrote SYNTHETIC data to {out_dir} ({n_machines} machines, {days} days)")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "data")
