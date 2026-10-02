"""Is a near-perfect score real signal or a bug? Three checks, written to reports/SANITY.md (and the README).

1. Shuffled labels: train on randomly shuffled labels. If the pipeline leaked the answer some other way,
   the score would stay high. It must fall to the random level.
2. Ablation: train on one feature group at a time, to see where the signal actually comes from.
3. Horizon: how does the score change if we ask for a shorter or longer warning window?

Run: python sanity_check.py   (a few minutes)
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from config import data_dir, load_params, report_dir
from features import FEATURES, SENSORS, add_labels, build_features, load_raw
from report import update_readme
from train import time_split

HORIZONS = (12, 24, 48, 72)


def main() -> None:
    p = load_params()
    raw = load_raw(data_dir())
    feats = build_features(raw["telemetry"], raw["errors"], raw["maint"], raw["machines"])

    def split(horizon):
        df = add_labels(feats, raw["failures"], horizon)
        df = df[df["datetime"].dt.hour % p["step_hours"] == 0]
        train, _, test, _ = time_split(df, p["train_frac"], p["valid_frac"], horizon)
        return train, test

    def score(train, test, cols, labels=None):
        m = HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, random_state=p["seed"])
        m.fit(train[cols], train["label"] if labels is None else labels)
        return average_precision_score(test["label"], m.predict_proba(test[cols])[:, 1])

    train, test = split(p["horizon_hours"])
    groups = {
        "All features": FEATURES,
        "Error counts only": [f for f in FEATURES if "_count_" in f],
        "Maintenance history only": [f for f in FEATURES if f.startswith("days_since_")],
        "Sensors only (raw, rolling, trend)": [f for f in FEATURES if f.split("_")[0] in SENSORS],
        "Machine age + model only": [f for f in FEATURES if f == "age" or f.startswith("model_")],
    }
    rows = [(name, score(train, test, cols)) for name, cols in groups.items()]
    shuffled = np.random.default_rng(p["seed"]).permutation(train["label"].to_numpy())
    rows.append(("All features, **shuffled** training labels", score(train, test, FEATURES, shuffled)))
    prevalence = test["label"].mean()

    horizon_rows = []
    for h in HORIZONS:
        tr, te = split(h)
        horizon_rows.append((h, te["label"].mean(), score(tr, te, FEATURES)))

    block = "\n".join([
        f"Test PR-AUC, {p['horizon_hours']}h window. Random guessing = {prevalence:.4f}.", "",
        "| Trained on | Test PR-AUC |", "|---|---|", *[f"| {n} | {s:.4f} |" for n, s in rows], "",
        "Same pipeline with a shorter or longer warning window:", "",
        "| Warning window | Positive rows | Test PR-AUC |", "|---|---|---|",
        *[f"| {h}h | {prev:.2%} | {s:.4f} |" for h, prev, s in horizon_rows],
    ])
    report_dir().mkdir(parents=True, exist_ok=True)
    (report_dir() / "SANITY.md").write_text("# Sanity check\n\n" + block + "\n", encoding="utf-8")
    print(block)
    if not (data_dir() / "SYNTHETIC").exists() and update_readme("SANITY", block):
        print("\nREADME.md sanity block updated.")


if __name__ == "__main__":
    main()
