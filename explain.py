"""Why is this machine flagged? Occlusion: set one group of features back to its healthy-fleet median and
see how much the risk score drops. The groups that drop it most are the reasons. No extra dependency,
and the same function serves the dashboard and the API."""
import pandas as pd

from features import COMPS, ERRORS, FEATURES, SENSORS

GROUPS = {s: [f for f in FEATURES if f == s or f.startswith(f"{s}_")] for s in SENSORS}
GROUPS.update({e: [f"{e}_count_24h"] for e in ERRORS})
GROUPS.update({c: [f"days_since_{c}"] for c in COMPS})
GROUPS["age"] = ["age"]


def healthy_baseline(df: pd.DataFrame) -> dict:
    """Median feature values of rows that were not followed by a failure."""
    return df.loc[df["label"] == 0, FEATURES].median().to_dict()


def _text(group: str, row: pd.Series, base: dict) -> str:
    if group in SENSORS:
        col = f"{group}_mean_24h"
        return f"{group} 24h average {row[col]:.0f} (normal {base[col]:.0f})"
    if group in ERRORS:
        return f"{group} logged {row[f'{group}_count_24h']:.0f}x in the last 24h"
    if group in COMPS:
        return f"{group} not replaced for {row[f'days_since_{group}']:.0f} days"
    return f"machine age {row['age']:.0f} years"


def reasons(model, X: pd.DataFrame, baseline: dict, top: int = 3, min_drop: float = 0.02) -> list[list[dict]]:
    """For each row: up to `top` reasons, strongest first. A reason must lower the risk by at least `min_drop`."""
    X = X[FEATURES].reset_index(drop=True)
    if X.empty:
        return []
    risk = model.predict_proba(X)[:, 1]
    drops = {}
    for group, cols in GROUPS.items():
        occluded = X.copy()
        for c in cols:
            occluded[c] = baseline[c]
        drops[group] = risk - model.predict_proba(occluded)[:, 1]
    drops = pd.DataFrame(drops)
    out = []
    for i in range(len(X)):
        best = drops.iloc[i].sort_values(ascending=False).head(top)
        out.append([{"factor": g, "text": _text(g, X.iloc[i], baseline), "risk_drop": round(float(d), 3)}
                    for g, d in best.items() if d >= min_drop])
    return out


def as_text(reason_list: list[dict]) -> str:
    return "; ".join(r["text"] for r in reason_list)
