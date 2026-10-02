"""Writes reports/RESULTS.md and keeps the Results block in README.md in sync with the latest real run."""
import os

from config import ROOT


def _money(x: float) -> str:
    return f"{x:,.0f}"


def summary_md(m: dict) -> str:
    """The short block that goes into the README."""
    r, e, c = m["test"]["rows"], m["test"]["events"], m["component"]
    model_row = next(p for p in m["policy"] if p["policy"].startswith("Model"))
    comp = (f"{c['accuracy']:.1%} of {c['n']} warning rows (most-overdue-component rule: {c['baseline_accuracy']:.1%})"
            if c else "not trained (too few failures)")
    policy = "\n".join(
        f"| {p['policy']} | {p['failures_caught']} | {p['failures_missed']} | {p['false_alarm_days']} | "
        f"{_money(p['cost'])} | {_money(p['saving_vs_run_to_failure'])} ({p['saving_pct']:.0f}%) |" for p in m["policy"])
    k = m["costs"]
    be = m["breakeven_inspection_cost"]
    breakeven = ("The model raised no false alarms in the test period, so there is no break-even inspection cost."
                 if be == float("inf") else
                 f"The model's alerts stay worthwhile until one false-alarm inspection costs more than {_money(be)}.")
    return f"""Test period = the last ~20% of the timeline, never seen during training or threshold selection.
Model version `{m['version']}`. Full tables: [`reports/RESULTS.md`](reports/RESULTS.md).

| Metric (test period) | Value |
|---|---|
| Selected model | {m['model']} |
| PR-AUC | {r['pr_auc']:.4f}, 95% CI [{m['pr_auc_ci'][0]:.4f}, {m['pr_auc_ci'][1]:.4f}] (random guessing = {r['prevalence']:.3f}) |
| Recall / Precision at the alert threshold ({m['threshold']:.2f}) | {r['recall']:.1%} / {r['precision']:.1%} |
| Failures caught | {e['failures_caught']} of {e['failures']}, median warning {e['median_lead_hours']:.0f}h ahead |
| False-alarm days | {e['false_alarm_days_per_week']:.1f} per week across {e['machines']} machines |
| Which component will fail (top-1) | {comp} |

**Cost of each policy over the test period** (ASSUMED: breakdown {_money(k['unplanned_failure'])}, planned repair
{_money(k['planned_repair'])}, inspection after a false alarm {_money(k['inspection'])}):

| Policy | Caught | Missed | False-alarm days | Cost | Saving vs run-to-failure |
|---|---|---|---|---|---|
{policy}

{breakeven}
Model saving: **{_money(model_row['saving_vs_run_to_failure'])}** ({model_row['saving_pct']:.0f}%).

![PR curve](reports/pr_curve.png)
"""


def results_md(m: dict) -> str:
    r = m["test"]["rows"]
    br = m["baseline_any_error"]["rows"]
    warn = "> **SYNTHETIC DATA - these numbers mean nothing. Re-run on the real dataset.**\n\n" if m["synthetic_data"] else ""
    cand = "\n".join(f"| {k} | {v:.4f} |" for k, v in m["valid_pr_auc"].items())
    imp = "\n".join(f"| {k} | {v:.4f} |" for k, v in m["importance"])
    summary = summary_md(m).replace("reports/pr_curve.png", "pr_curve.png").replace(
        "Full tables: [`reports/RESULTS.md`](reports/RESULTS.md).", "")
    return f"""# Results

{warn}Question: will this machine fail in the next **{m['horizon_hours']} hours**?
Split by time: train until {m['split']['train_end'][:10]}, validation until {m['split']['valid_end'][:10]}, test after.
Rows: {m['rows']['train']} / {m['rows']['valid']} / {m['rows']['test']}.
Alert threshold {m['threshold']:.2f}, chosen on validation by `{m['threshold_by']}`.

{summary}
## Row-level detail

| Metric | Model | Rule: alert on any error in last 24h |
|---|---|---|
| ROC-AUC | {r['roc_auc']:.4f} | - |
| Recall | {r['recall']:.1%} | {br['recall']:.1%} |
| Precision | {r['precision']:.1%} | {br['precision']:.1%} |
| False alarms (rows) | {r['false_alarms']} | {br['false_alarms']} |

## Candidates (validation PR-AUC)

| Model | PR-AUC |
|---|---|
{cand}

## Top features (permutation importance, drop in PR-AUC)

| Feature | Importance |
|---|---|
{imp}
"""


def update_readme(marker: str, text: str) -> bool:
    """Replace what sits between <!-- MARKER:START --> and <!-- MARKER:END --> in README.md.

    Skipped when folders are redirected by env vars (tests, CI) so synthetic numbers never reach the README.
    """
    readme = ROOT / "README.md"
    if os.environ.get("PDM_MODEL_DIR") or os.environ.get("PDM_DATA_DIR") or not readme.exists():
        return False
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    s = readme.read_text(encoding="utf-8")
    if start not in s or end not in s:
        return False
    new = s[: s.index(start) + len(start)] + "\n" + text.strip() + "\n" + s[s.index(end):]
    readme.write_text(new, encoding="utf-8")
    return True
