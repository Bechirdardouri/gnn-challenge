"""Baseline tabular model for HeteroShot challenge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    val = pd.read_csv(ROOT / "data" / "val.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")

    feature_cols = [c for c in train.columns if c.startswith("f")]

    x_train = train[feature_cols]
    y_train = train["target"]
    x_val = val[feature_cols]
    y_val = val["target"]

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(x_train, y_train)

    pred = clf.predict(x_val)
    score = f1_score(y_val, pred, average="macro")
    print(f"Val Macro-F1: {score:.4f}")

    test_pred = clf.predict(test[feature_cols])
    sub = pd.DataFrame({"node_id": test["node_id"], "target": test_pred})
    output = ROOT / "submissions" / "baseline_tabular_submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
