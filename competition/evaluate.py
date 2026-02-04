"""Evaluation script for HeteroShot challenge.

This script computes the Macro-F1 score for a submission against ground truth labels.
"""

import argparse
import csv
import sys
from pathlib import Path


def read_labels(path: str) -> dict[str, int]:
    """Read labels from a CSV file.
    
    Args:
        path: Path to CSV file with node_id and target columns
        
    Returns:
        Dictionary mapping node_id to target label
        
    Raises:
        ValueError: If CSV format is invalid
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        
        required = {"node_id", "target"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        
        labels = {}
        for row in reader:
            node_id = row["node_id"]
            if node_id in labels:
                raise ValueError(f"Duplicate node_id: {node_id}")
            try:
                labels[node_id] = int(row["target"])
            except ValueError:
                raise ValueError(f"Invalid target for node {node_id}: must be integer")
        
        return labels


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Compute Macro-averaged F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Macro-F1 score
    """
    labels = sorted(set(y_true))
    scores = []
    
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        
        denom = (2 * tp) + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        scores.append(f1)
    
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(submission_path: str, truth_path: str) -> dict:
    """Evaluate a submission against ground truth.
    
    Args:
        submission_path: Path to submission CSV
        truth_path: Path to ground truth CSV
        
    Returns:
        Dictionary with evaluation results
    """
    submission = read_labels(submission_path)
    truth = read_labels(truth_path)
    
    # Check all truth IDs are present
    missing = [nid for nid in truth if nid not in submission]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} nodes: {missing[:10]}")
    
    # Align predictions with truth
    y_true = [truth[nid] for nid in truth]
    y_pred = [submission[nid] for nid in truth]
    
    # Compute metrics
    score = macro_f1(y_true, y_pred)
    
    # Per-class breakdown
    labels = sorted(set(y_true))
    class_scores = {}
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for yt in y_true if yt == label)
        }
    
    return {
        "macro_f1": score,
        "num_classes": len(labels),
        "num_samples": len(y_true),
        "class_scores": class_scores
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate HeteroShot submission")
    parser.add_argument("--submission", required=True, help="Path to submission CSV")
    parser.add_argument("--truth", required=True, help="Path to ground truth CSV")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-class breakdown")
    args = parser.parse_args()
    
    try:
        results = evaluate(args.submission, args.truth)
        
        print(f"MACRO_F1={results['macro_f1']:.6f}")
        print(f"Classes: {results['num_classes']}")
        print(f"Samples: {results['num_samples']}")
        
        if args.verbose:
            print("\nPer-class breakdown:")
            print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
            print("-" * 54)
            for label, scores in sorted(results["class_scores"].items()):
                print(f"{label:<8} {scores['precision']:<12.4f} {scores['recall']:<12.4f} "
                      f"{scores['f1']:<12.4f} {scores['support']:<10}")
    
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
