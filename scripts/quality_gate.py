"""Quality gate: block a new model that is clearly worse. Used by DVC (stage `gate`), Airflow and CI.

Exit code 1 = fail. Rules: PR-AUC and failure recall above a floor, and no big drop vs the previous version.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_DIR = Path(os.environ.get("PDM_MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.environ.get("PDM_REPORT_DIR", ROOT / "reports"))
MIN_PR_AUC = float(os.environ.get("GATE_MIN_PR_AUC", "0.80"))
MIN_FAILURE_RECALL = float(os.environ.get("GATE_MIN_FAILURE_RECALL", "0.90"))
MAX_DROP = 0.02


def main() -> int:
    m = json.loads((REPORT_DIR / "metrics.json").read_text())
    registry = json.loads((MODEL_DIR / "registry.json").read_text())
    problems = []
    if m["pr_auc"] < MIN_PR_AUC:
        problems.append(f"PR-AUC {m['pr_auc']:.4f} < {MIN_PR_AUC}")
    if m["failure_recall"] < MIN_FAILURE_RECALL:
        problems.append(f"failure recall {m['failure_recall']:.2%} < {MIN_FAILURE_RECALL:.0%}")
    if len(registry) >= 2 and registry[-2]["pr_auc"] - m["pr_auc"] > MAX_DROP:
        problems.append(f"PR-AUC dropped {registry[-2]['pr_auc']:.4f} -> {m['pr_auc']:.4f} (> {MAX_DROP})")
    if problems:
        print("QUALITY GATE FAILED:", "; ".join(problems))
        return 1
    print(f"quality gate passed (PR-AUC {m['pr_auc']:.4f}, failure recall {m['failure_recall']:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
