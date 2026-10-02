"""Validate -> features -> time split -> train candidates -> pick model + threshold on validation -> test once.

Run: python train.py
"""
import json
import math
import os
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

import policy
from config import ROOT, data_dir, load_params, model_dir, report_dir
from evaluate import bootstrap_pr_auc, cost_threshold, event_metrics, fbeta_threshold, row_metrics
from explain import as_text, healthy_baseline, reasons
from features import COMPS, ERRORS, FEATURES, SENSORS, add_labels, build_features, load_raw
from report import results_md, summary_md, update_readme
from validate import validate

SCORE_COLS = ["machineID", "datetime", "label", "hours_to_failure"]


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


def candidates(p: dict) -> dict:
    # class_weight="balanced": failures are rare, so each failure row counts more during training
    models = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
    }
    for g in p["gb_grid"]:
        name = f"Gradient Boosting (lr={g['learning_rate']}, leaves={g['max_leaf_nodes']})"
        models[name] = HistGradientBoostingClassifier(class_weight="balanced", max_iter=300,
                                                      random_state=p["seed"], **g)
    return models


def train_component_model(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    """Second model, trained only on warning rows: which component is about to fail?"""
    pos, test_pos = train[train["label"] == 1], test[test["label"] == 1]
    target = pos["failing_comps"].str.split("+").str[0]
    if target.nunique() < 2 or test_pos.empty:
        return None, None
    model = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(pos[FEATURES], target)
    truth = test_pos["failing_comps"].str.split("+")
    pred = model.predict(test_pos[FEATURES])
    # naive rule to beat: "the component that has gone longest without replacement"
    overdue = test_pos[[f"days_since_{c}" for c in COMPS]].fillna(-1).to_numpy().argmax(axis=1)
    info = {
        "n": len(test_pos),
        "accuracy": float(np.mean([p in t for p, t in zip(pred, truth)])),
        "baseline_accuracy": float(np.mean([COMPS[i] in t for i, t in zip(overdue, truth)])),
        "classes": list(model.classes_),
    }
    return model, info


def annotate(scored: pd.DataFrame, rows: pd.DataFrame, model, comp_model, baseline: dict, threshold: float):
    """For machines at WATCH level or above: likely component and plain-language reasons."""
    scored = scored.assign(component="", reasons="")
    hot = (scored["risk"] >= threshold / 2).to_numpy()
    if hot.any():
        X = rows.loc[hot, FEATURES]
        if comp_model is not None:
            scored.loc[hot, "component"] = comp_model.predict(X)
        scored.loc[hot, "reasons"] = [as_text(r) for r in reasons(model, X, baseline)]
    return scored


def log_mlflow(meta: dict, flat: dict, artifacts: list) -> None:
    """Experiment tracking is optional: skipped if mlflow is not installed, never allowed to break training."""
    if os.environ.get("PDM_NO_MLFLOW"):
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("predictive-maintenance")
        with mlflow.start_run(run_name=meta["version"]):
            mlflow.log_params({"model": meta["model"], "horizon_hours": meta["horizon_hours"],
                               "step_hours": meta["step_hours"], "threshold": meta["threshold"],
                               "threshold_by": meta["threshold_by"], **{f"cost_{k}": v for k, v in meta["costs"].items()}})
            mlflow.log_metrics({k: v for k, v in flat.items() if not math.isnan(v)})
            for a in artifacts:
                mlflow.log_artifact(str(a))
        print("MLflow run logged. View: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        print(f"(MLflow logging skipped: {exc})")


def main() -> None:
    p = load_params()
    horizon, step, costs = p["horizon_hours"], p["step_hours"], p["costs"]
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
    for name, model in candidates(p).items():
        model.fit(train[FEATURES], train["label"])
        fitted[name] = model
        valid_pr[name] = float(average_precision_score(valid["label"], model.predict_proba(valid[FEATURES])[:, 1]))
        print(f"  {name}: validation PR-AUC {valid_pr[name]:.4f}")
    best = max(valid_pr, key=valid_pr.get)
    model = fitted[best]

    valid_scored = valid[SCORE_COLS].assign(risk=model.predict_proba(valid[FEATURES])[:, 1])
    if p["threshold_by"] == "cost" and valid["label"].sum() > 0:
        threshold = cost_threshold(valid_scored, costs)
    else:
        threshold = fbeta_threshold(valid["label"], valid_scored["risk"], p["fbeta"])

    # the test set is touched exactly once, after model and threshold are fixed
    risk = model.predict_proba(test[FEATURES])[:, 1]
    scored = test[SCORE_COLS].assign(risk=risk)
    rows = row_metrics(test["label"], risk, threshold)
    events = event_metrics(scored, threshold)
    ci = bootstrap_pr_auc(scored, p["n_bootstrap"], p["seed"])

    # naive baseline: raise an alert whenever the machine logged any error in the last 24h
    rule = (test[[f"{e}_count_24h" for e in ERRORS]].sum(axis=1) > 0).astype(float)
    rule_rows = row_metrics(test["label"], rule, 0.5)
    rule_events = event_metrics(scored.assign(risk=rule.to_numpy()), 0.5)

    comp_model, comp_info = train_component_model(train, test, p["seed"])
    baseline = healthy_baseline(train)
    scored = annotate(scored, test, model, comp_model, baseline, threshold)

    sample = valid.sample(min(len(valid), 20000), random_state=p["seed"])
    imp = permutation_importance(model, sample[FEATURES], sample["label"], scoring="average_precision",
                                 n_repeats=3, random_state=p["seed"])
    importance = sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()), key=lambda t: -t[1])[:10]

    policies = policy.compare({
        "Run to failure (no alerts)": policy.no_alerts(events),
        "Rule: alert on any error in last 24h": rule_events,
        f"Model (threshold {threshold:.2f})": events,
    }, costs)

    meta = {
        "version": pd.Timestamp.now().strftime("%Y%m%d-%H%M%S"),
        "model": best,
        "synthetic_data": (ddir / "SYNTHETIC").exists(),
        "horizon_hours": horizon,
        "step_hours": step,
        "threshold": threshold,
        "threshold_by": p["threshold_by"],
        "features": FEATURES,
        "split": split,
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
        "valid_pr_auc": valid_pr,
        "test": {"rows": rows, "events": events},
        "pr_auc_ci": ci,
        "baseline_any_error": {"rows": rule_rows, "events": rule_events},
        "component": comp_info,
        "costs": costs,
        "policy": policies,
        "breakeven_inspection_cost": policy.breakeven_inspection_cost(events, costs),
        "importance": importance,
        "baseline": baseline,
    }
    flat = {
        "pr_auc": rows["pr_auc"], "roc_auc": rows["roc_auc"], "recall": rows["recall"],
        "precision": rows["precision"], "failure_recall": events["failure_recall"],
        "false_alarm_days_per_week": events["false_alarm_days_per_week"],
        "median_lead_hours": events["median_lead_hours"],
        "component_accuracy": comp_info["accuracy"] if comp_info else float("nan"),
        "saving_vs_run_to_failure": policies[-1]["saving_vs_run_to_failure"],
    }

    joblib.dump(model, mdir / "model.joblib")
    if comp_model is not None:
        joblib.dump(comp_model, mdir / "component_model.joblib")
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2))
    (rdir / "metrics.json").write_text(json.dumps(flat, indent=2))
    registry_path = mdir / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({"version": meta["version"], "model": best, "threshold": threshold, **flat})
    registry_path.write_text(json.dumps(registry[-20:], indent=2))

    # what the monitoring side needs: test-period scores for the dashboard, a training sample for drift checks
    out = scored.merge(test[["machineID", "datetime", *[f"{c}_mean_24h" for c in SENSORS]]], on=["machineID", "datetime"])
    out.to_csv(mdir / "test_scores.csv.gz", index=False, float_format="%.4f")
    train[FEATURES].sample(min(len(train), 5000), random_state=p["seed"]).to_csv(
        mdir / "reference.csv.gz", index=False, float_format="%.4f")

    plt.switch_backend("Agg")
    precision, recall, _ = precision_recall_curve(test["label"], risk)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"model (PR-AUC {rows['pr_auc']:.3f})")
    ax.axhline(rows["prevalence"], ls="--", c="grey", label=f"random ({rows['prevalence']:.3f})")
    ax.scatter([rows["recall"]], [rows["precision"]], c="red", zorder=3, label=f"alert threshold {threshold:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall on the test period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rdir / "pr_curve.png", dpi=120)
    plt.close(fig)

    (rdir / "RESULTS.md").write_text(results_md(meta), encoding="utf-8")
    if not meta["synthetic_data"] and update_readme("RESULTS", summary_md(meta)):
        print("README.md results block updated.")
    log_mlflow(meta, flat, [rdir / "RESULTS.md", rdir / "pr_curve.png", mdir / "meta.json"])

    print(f"\n{best} | threshold {threshold:.2f} ({p['threshold_by']}) | test PR-AUC {rows['pr_auc']:.4f} "
          f"[{ci[0]:.4f}, {ci[1]:.4f}] | recall {rows['recall']:.1%} | precision {rows['precision']:.1%}")
    print(f"failures caught {events['failures_caught']}/{events['failures']} (median warning "
          f"{events['median_lead_hours']:.0f}h) | false-alarm days per week {events['false_alarm_days_per_week']:.1f} "
          f"across {events['machines']} machines")
    if comp_info:
        print(f"component model: {comp_info['accuracy']:.1%} top-1 (most-overdue rule {comp_info['baseline_accuracy']:.1%})")
    print(f"saving vs run-to-failure: {policies[-1]['saving_vs_run_to_failure']:,.0f} "
          f"({policies[-1]['saving_pct']:.0f}%) under the assumed costs")
    print("Done. Next: streamlit run app.py | uvicorn api:app --reload | python monitor.py --simulate")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
