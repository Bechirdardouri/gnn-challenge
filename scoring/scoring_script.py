"""Scoring script for HeteroShot challenge submissions.

This script validates submission format and computes Macro-F1 score
against ground truth labels.
"""

import argparse
import csv


def _read_labels(path):
    """Read and validate labels from CSV file.
    
    Args:
        path: Path to CSV file with node_id and target columns
        
    Returns:
        dict: Mapping from node_id (str) to target label (int)
        
    Raises:
        ValueError: If CSV format is invalid
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        for col in ["node_id", "target"]:
            if col not in reader.fieldnames:
                raise ValueError(f"CSV missing column: {col}")
        labels = {}
        for row in reader:
            node_id = row["node_id"]
            if node_id in labels:
                raise ValueError("Duplicate node_id in submission.")
            try:
                labels[node_id] = int(row["target"])
            except ValueError as exc:
                raise ValueError("Predictions must be integer class labels.") from exc
        return labels


def _macro_f1(y_true, y_pred):
    """Calculate Macro-F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Macro-averaged F1 score
    """
    labels = sorted(set(y_true))
    scores = []
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(scores) / len(scores) if scores else 0.0


def main():
    """Main function to score a submission."""
    ap = argparse.ArgumentParser(description="Score a submission against ground truth")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()

    sub = _read_labels(args.submission)
    truth = _read_labels(args.truth)

    missing = [node_id for node_id in truth.keys() if node_id not in sub]
    if missing:
        raise ValueError(
            f"Missing predictions for node_ids (showing up to 10): {missing[:10]}"
        )

    y_true = [truth[node_id] for node_id in truth.keys()]
    y_pred = [sub[node_id] for node_id in truth.keys()]

    score = _macro_f1(y_true, y_pred)
    print(f"MACRO_F1={score:.6f}")


if __name__ == "__main__":
    main()
