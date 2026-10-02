"""Is a near-perfect score real signal or a bug? Two checks, written to reports/SANITY.md.

1. Shuffled labels: train on randomly shuffled labels. If the pipeline leaked the answer some other way,
   the score would stay high. It must fall to the random level.
2. Ablation: train on one feature group at a time, to see where the signal actually comes from.

Run: python sanity_check.py
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from config import data_dir, load_params, report_dir
from features import FEATURES, add_labels, build_features, load_raw
from train import time_split


def main() -> None:
    p = load_params()
    raw = load_raw(data_dir())
    df = add_labels(build_features(raw["telemetry"], raw["errors"], raw["maint"], raw["machines"]),
                    raw["failures"], p["horizon_hours"])
    df = df[df["datetime"].dt.hour % p["step_hours"] == 0]
    train, _, test, _ = time_split(df, p["train_frac"], p["valid_frac"], p["horizon_hours"])

    def score(cols, labels):
        m = HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, random_state=p["seed"])
        m.fit(train[cols], labels)
        return average_precision_score(test["label"], m.predict_proba(test[cols])[:, 1])

    groups = {
        "All features": FEATURES,
        "Sensors only (raw, rolling, trend)": [f for f in FEATURES if f.split("_")[0] in
                                               ("volt", "rotate", "pressure", "vibration")],
        "Error counts only": [f for f in FEATURES if "_count_" in f],
        "Maintenance history only": [f for f in FEATURES if f.startswith("days_since_")],
        "Machine age + model only": [f for f in FEATURES if f == "age" or f.startswith("model_")],
    }
    rows = [(name, score(cols, train["label"])) for name, cols in groups.items()]
    shuffled = np.random.default_rng(p["seed"]).permutation(train["label"].to_numpy())
    rows.append(("All features, SHUFFLED training labels", score(FEATURES, shuffled)))
    prevalence = test["label"].mean()

    lines = ["# Sanity check", "", f"Test PR-AUC. Random guessing = {prevalence:.4f}.", "",
             "| Trained on | Test PR-AUC |", "|---|---|", *[f"| {n} | {s:.4f} |" for n, s in rows]]
    out = "\n".join(lines) + "\n"
    report_dir().mkdir(parents=True, exist_ok=True)
    (report_dir() / "SANITY.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
