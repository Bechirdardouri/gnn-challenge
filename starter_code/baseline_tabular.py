"""Baseline tabular model for HeteroShot challenge.

This script trains a Random Forest classifier on node features only,
ignoring the graph structure. It serves as a simple baseline.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def main():
    """Train Random Forest classifier and generate predictions."""
    train = pd.read_csv("../data/train.csv")
    val = pd.read_csv("../data/val.csv")
    test = pd.read_csv("../data/test.csv")

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
    sub.to_csv("../submissions/sample_submission.csv", index=False)
    print("Wrote submissions/sample_submission.csv")


if __name__ == "__main__":
    main()
