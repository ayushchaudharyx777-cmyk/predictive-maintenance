"""Validate -> features -> time split -> train candidates -> pick model + threshold on validation -> test once.

Run: python train.py
"""
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import data_dir, load_params, model_dir, report_dir
from evaluate import event_metrics, pick_threshold, row_metrics
from features import ERRORS, FEATURES, SENSORS, add_labels, build_features, load_raw
from validate import validate


def time_split(df: pd.DataFrame, train_frac: float, valid_frac: float, horizon_hours: int):
    """Chronological split. Rows whose label window crosses a boundary are dropped (no overlap between sets)."""
    t0, t1 = df["datetime"].min(), df["datetime"].max()
    train_end = t0 + (t1 - t0) * train_frac
    valid_end = t0 + (t1 - t0) * (train_frac + valid_frac)
    gap = pd.Timedelta(hours=horizon_hours)
    train = df[df["datetime"] <= train_end - gap]
    valid = df[(df["datetime"] > train_end) & (df["datetime"] <= valid_end - gap)]
    test = df[df["datetime"] > valid_end]
    return train, valid, test, {"train_end": str(train_end), "valid_end": str(valid_end)}


def candidates(seed: int) -> dict:
    # class_weight="balanced": failures are rare, so each failure row counts more during training
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=300, learning_rate=0.1, random_state=seed,
        ),
    }


def main() -> None:
    p = load_params()
    horizon, step = p["horizon_hours"], p["step_hours"]
    ddir, mdir, rdir = data_dir(), model_dir(), report_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(ddir)
    problems = validate(raw)
    if problems:
        print("VALIDATION FAILED:\n - " + "\n - ".join(problems))
        sys.exit(1)

    print("Building features...")
    feats = build_features(raw["telemetry"], raw["errors"], raw["maint"], raw["machines"])
    df = add_labels(feats, raw["failures"], horizon)
    df = df[df["datetime"].dt.hour % step == 0].reset_index(drop=True)
    train, valid, test, split = time_split(df, p["train_frac"], p["valid_frac"], horizon)
    print(f"rows: train {len(train)}, valid {len(valid)}, test {len(test)} | positives: "
          f"{train['label'].mean():.2%} / {valid['label'].mean():.2%} / {test['label'].mean():.2%}")

    fitted, valid_pr = {}, {}
    for name, model in candidates(p["seed"]).items():
        model.fit(train[FEATURES], train["label"])
        fitted[name] = model
        valid_pr[name] = float(average_precision_score(valid["label"], model.predict_proba(valid[FEATURES])[:, 1]))
        print(f"  {name}: validation PR-AUC {valid_pr[name]:.4f}")
    best = max(valid_pr, key=valid_pr.get)
    model = fitted[best]
    threshold = pick_threshold(valid["label"], model.predict_proba(valid[FEATURES])[:, 1], p["fbeta"])

    # the test set is touched exactly once, after model and threshold are fixed
    risk = model.predict_proba(test[FEATURES])[:, 1]
    scored = test[["machineID", "datetime", "label", "hours_to_failure"]].assign(risk=risk)
    rows = row_metrics(test["label"], risk, threshold)
    events = event_metrics(scored, threshold)

    # naive baseline: raise an alert whenever the machine logged any error in the last 24h
    rule = (test[[f"{e}_count_24h" for e in ERRORS]].sum(axis=1) > 0).astype(float)
    rule_rows = row_metrics(test["label"], rule, 0.5)
    rule_events = event_metrics(scored.assign(risk=rule.to_numpy()), 0.5)

    sample = valid.sample(min(len(valid), 20000), random_state=p["seed"])
    imp = permutation_importance(model, sample[FEATURES], sample["label"], scoring="average_precision",
                                 n_repeats=3, random_state=p["seed"])
    importance = sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()), key=lambda t: -t[1])[:10]

    meta = {
        "version": pd.Timestamp.now().strftime("%Y%m%d-%H%M%S"),
        "model": best,
        "synthetic_data": (ddir / "SYNTHETIC").exists(),
        "horizon_hours": horizon,
        "step_hours": step,
        "threshold": threshold,
        "fbeta": p["fbeta"],
        "features": FEATURES,
        "split": split,
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "valid_pr_auc": valid_pr,
        "test": {"rows": rows, "events": events},
        "baseline_any_error": {"rows": rule_rows, "events": rule_events},
        "importance": importance,
    }
    joblib.dump(model, mdir / "model.joblib")
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2))
    out = scored.merge(test[["machineID", "datetime", *[f"{c}_mean_24h" for c in SENSORS]]], on=["machineID", "datetime"])
    out.to_csv(mdir / "test_scores.csv.gz", index=False, float_format="%.4f")

    plt.switch_backend("Agg")
    precision, recall, _ = precision_recall_curve(test["label"], risk)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"{best} (PR-AUC {rows['pr_auc']:.3f})")
    ax.axhline(rows["prevalence"], ls="--", c="grey", label=f"random ({rows['prevalence']:.3f})")
    ax.scatter([rows["recall"]], [rows["precision"]], c="red", zorder=3, label=f"alert threshold {threshold:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall on the test period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rdir / "pr_curve.png", dpi=120)
    plt.close(fig)

    (rdir / "RESULTS.md").write_text(results_md(meta))
    print(f"\n{best} | threshold {threshold:.3f} | test PR-AUC {rows['pr_auc']:.4f} | recall {rows['recall']:.1%} | "
          f"precision {rows['precision']:.1%}")
    print(f"failures caught {events['failures_caught']}/{events['failures']} | "
          f"false-alarm days per week {events['false_alarm_days_per_week']:.1f} across {events['machines']} machines")
    print("Done. Next: streamlit run app.py | uvicorn api:app --reload")


def results_md(m: dict) -> str:
    r, e = m["test"]["rows"], m["test"]["events"]
    br, be = m["baseline_any_error"]["rows"], m["baseline_any_error"]["events"]
    warn = "> **SYNTHETIC DATA - these numbers mean nothing. Re-run on the real dataset.**\n\n" if m["synthetic_data"] else ""
    cand = "\n".join(f"| {k} | {v:.4f} |" for k, v in m["valid_pr_auc"].items())
    imp = "\n".join(f"| {k} | {v:.4f} |" for k, v in m["importance"])
    return f"""# Results (version {m['version']})

{warn}Question: will this machine fail in the next **{m['horizon_hours']} hours**?
Split by time: train until {m['split']['train_end'][:10]}, validation until {m['split']['valid_end'][:10]}, test after.
Rows: {m['rows']['train']} / {m['rows']['valid']} / {m['rows']['test']}.

## Model selection (validation PR-AUC)

| Model | PR-AUC |
|---|---|
{cand}

Selected: **{m['model']}**. Alert threshold {m['threshold']:.3f} (best F{m['fbeta']:g} on validation).

## Test period

| Metric | Model | Baseline: alert on any error in last 24h |
|---|---|---|
| PR-AUC (random = {r['prevalence']:.3f}) | {r['pr_auc']:.4f} | - |
| ROC-AUC | {r['roc_auc']:.4f} | - |
| Recall (rows) | {r['recall']:.1%} | {br['recall']:.1%} |
| Precision (rows) | {r['precision']:.1%} | {br['precision']:.1%} |
| False alarms (rows) | {r['false_alarms']} | {br['false_alarms']} |
| Failures caught | {e['failures_caught']} of {e['failures']} | {be['failures_caught']} of {be['failures']} |
| False-alarm days per week ({e['machines']} machines) | {e['false_alarm_days_per_week']:.1f} | {be['false_alarm_days_per_week']:.1f} |

## Top features (permutation importance, drop in PR-AUC)

| Feature | Importance |
|---|---|
{imp}

![PR curve](pr_curve.png)
"""


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
