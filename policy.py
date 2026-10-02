"""Turn alerts into money. Every number here follows from the ASSUMED costs in params.yaml.

A caught failure becomes a planned repair, a missed one is an unplanned breakdown,
and every false-alarm day costs one inspection.
"""


def total_cost(events: dict, costs: dict) -> float:
    missed = events["failures"] - events["failures_caught"]
    return (events["failures_caught"] * costs["planned_repair"]
            + missed * costs["unplanned_failure"]
            + events["false_alarm_days"] * costs["inspection"])


def compare(policies: dict, costs: dict) -> list[dict]:
    """policies: name -> event metrics. Returns one row per policy, with the saving vs running to failure."""
    rows = []
    for name, ev in policies.items():
        run_to_failure = ev["failures"] * costs["unplanned_failure"]
        cost = total_cost(ev, costs)
        rows.append({
            "policy": name,
            "failures_caught": ev["failures_caught"],
            "failures_missed": ev["failures"] - ev["failures_caught"],
            "false_alarm_days": ev["false_alarm_days"],
            "cost": cost,
            "saving_vs_run_to_failure": run_to_failure - cost,
            "saving_pct": 100 * (run_to_failure - cost) / run_to_failure if run_to_failure else 0.0,
        })
    return rows


def no_alerts(events: dict) -> dict:
    """The do-nothing policy: every failure is an unplanned breakdown."""
    return dict(events, failures_caught=0, false_alarm_days=0)


def breakeven_inspection_cost(events: dict, costs: dict) -> float:
    """How expensive could one false-alarm inspection get before the alerts stop paying for themselves?"""
    gain = events["failures_caught"] * (costs["unplanned_failure"] - costs["planned_repair"])
    return gain / events["false_alarm_days"] if events["false_alarm_days"] else float("inf")
