"""Drift monitoring: do today's machines still look like the data the model was trained on?

  python monitor.py --simulate                       demo: drifted sensors -> ALERT (exit code 1)
  python monitor.py --current logs/predictions.jsonl real API traffic vs the training reference
  python monitor.py --current batch.csv              any CSV of feature rows

PSI per feature: < 0.10 OK, 0.10-0.25 WARN, > 0.25 ALERT. Report: reports/drift_report.json
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from config import model_dir, report_dir
from features import COMPS, SENSORS

# continuous features only: PSI on rare error counts or one-hot flags is noise
MONITORED = SENSORS + [f"{s}_mean_24h" for s in SENSORS] + [f"{s}_std_24h" for s in SENSORS] + \
    [f"days_since_{c}" for c in COMPS]
MIN_ROWS = 30


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference, current = reference.dropna(), current.dropna()
    if reference.empty or current.empty:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def level(value: float) -> str:
    return "ALERT" if value > 0.25 else "WARN" if value > 0.10 else "OK"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, source: str = "") -> dict:
    rows = []
    for col in MONITORED:
        if col in current:
            v = psi(reference[col], current[col])
            rows.append({"feature": col, "psi": round(v, 4), "level": level(v),
                         "reference_mean": round(float(reference[col].mean()), 3),
                         "current_mean": round(float(current[col].mean()), 3)})
    rows.sort(key=lambda r: -r["psi"])
    worst = "ALERT" if any(r["level"] == "ALERT" for r in rows) else \
        "WARN" if any(r["level"] == "WARN" for r in rows) else "OK"
    return {"status": worst, "source": source, "rows_checked": len(current), "generated": str(pd.Timestamp.now().floor("s")),
            "features": rows}


def simulate(reference: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A plausible bad day: pressure sensors read 12% high and vibration is up 20% across the fleet."""
    cur = reference.sample(min(len(reference), 1000), random_state=seed).copy()
    for col in cur.columns:
        if col.startswith("pressure") and "_std_" not in col and "_trend_" not in col:
            cur[col] *= 1.12
        if col.startswith("vibration") and "_std_" not in col and "_trend_" not in col:
            cur[col] *= 1.20
    return cur


def load_current(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True) if path.endswith(".jsonl") else pd.read_csv(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", help="CSV or JSONL of feature rows to check")
    ap.add_argument("--simulate", action="store_true", help="check a synthetic drifted batch")
    ap.add_argument("--save-batch", help="with --simulate: also save the drifted batch to this CSV")
    args = ap.parse_args(argv)

    ref_path = model_dir() / "reference.csv.gz"
    if not ref_path.exists():
        print("No models/reference.csv.gz - run `python train.py` first.")
        return 2
    reference = pd.read_csv(ref_path)
    if args.simulate:
        current = simulate(reference)
        if args.save_batch:
            current.to_csv(args.save_batch, index=False)
    elif args.current:
        current = load_current(args.current)
    else:
        ap.error("give --current FILE or --simulate")
    if len(current) < MIN_ROWS:
        print(f"Only {len(current)} rows - need at least {MIN_ROWS} for a meaningful check. Nothing done.")
        return 0

    source = "simulated drift demo (python monitor.py --simulate)" if args.simulate else args.current
    report = drift_report(reference, current, source)
    report_dir().mkdir(parents=True, exist_ok=True)
    (report_dir() / "drift_report.json").write_text(json.dumps(report, indent=2))
    print(f"Drift status: {report['status']} ({report['rows_checked']} rows checked)")
    for r in report["features"][:6]:
        print(f"  {r['level']:5} {r['feature']:22} PSI {r['psi']:.3f}  mean {r['reference_mean']} -> {r['current_mean']}")
    return 1 if report["status"] == "ALERT" else 0


if __name__ == "__main__":
    sys.exit(main())
