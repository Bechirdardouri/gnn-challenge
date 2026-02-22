"""Leaderboard update script for HeteroShot challenge.

This script scores all submissions in the submissions/ directory
and updates the authoritative leaderboard/leaderboard.csv file.
The leaderboard.md file is generated separately by competition/render_leaderboard.py.
"""

import csv
import json
from datetime import datetime
from pathlib import Path


def _read_labels(path):
    """Read labels from CSV file.
    
    Args:
        path: Path to CSV file
        
    Returns:
        dict: Mapping from node_id to target label
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
            labels[node_id] = int(row["target"])
        return labels


def _macro_f1(y_true, y_pred):
    labels = sorted(set(y_true))
    scores = []
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(scores) / len(scores) if scores else 0.0


def score_one(sub_path, truth_path="data/test_labels.csv"):
    """Score a single submission file.
    
    Args:
        sub_path: Path to submission CSV
        truth_path: Path to ground truth CSV
        
    Returns:
        float: Macro-F1 score
        
    Raises:
        ValueError: If submission is invalid
    """
    sub = _read_labels(sub_path)
    truth = _read_labels(truth_path)
    missing = [node_id for node_id in truth.keys() if node_id not in sub]
    if missing:
        raise ValueError(
            f"Missing predictions for node_ids (showing up to 10): {missing[:10]}"
        )
    y_true = [truth[node_id] for node_id in truth.keys()]
    y_pred = [sub[node_id] for node_id in truth.keys()]
    return _macro_f1(y_true, y_pred)


def load_existing_leaderboard(csv_path="leaderboard/leaderboard.csv"):
    """Load existing leaderboard entries from CSV.
    
    Returns:
        dict: Mapping from team name to entry dict
    """
    entries = {}
    csv_file = Path(csv_path)
    
    if csv_file.exists():
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                team = row.get("team", "")
                if team:
                    entries[team] = {
                        "team": team,
                        "model_type": row.get("model_type", ""),
                        "score": float(row.get("score", 0)),
                        "date": row.get("date", ""),
                        "notes": row.get("notes", "")
                    }
    
    return entries


def save_leaderboard_csv(entries, csv_path="leaderboard/leaderboard.csv"):
    """Save leaderboard entries to CSV file.
    
    Args:
        entries: List of entry dictionaries
        csv_path: Output file path
    """
    # Ensure directory exists
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by score descending
    sorted_entries = sorted(entries, key=lambda x: x.get("score", 0), reverse=True)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["team", "model_type", "score", "date", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sorted_entries:
            writer.writerow({
                "team": entry.get("team", ""),
                "model_type": entry.get("model_type", ""),
                "score": entry.get("score", 0),
                "date": entry.get("date", ""),
                "notes": entry.get("notes", "")
            })


def find_all_submissions():
    """Find all submission files (both legacy and inbox format).
    
    Returns:
        List of tuples: (team_name, csv_path, metadata_dict or None)
    """
    submissions = []
    
    # Legacy format: submissions/*.csv
    for csv_path in Path("submissions").glob("*.csv"):
        team = csv_path.stem
        submissions.append((team, str(csv_path), None))
    
    # New inbox format: submissions/inbox/<team>/<run_id>/predictions.csv
    for pred_path in Path("submissions/inbox").glob("*/*/predictions.csv"):
        parts = pred_path.parts
        # submissions/inbox/<team>/<run_id>/predictions.csv
        team = parts[-3]
        run_id = parts[-2]
        team_run = f"{team}/{run_id}"
        
        # Try to load metadata
        metadata_path = pred_path.parent / "metadata.json"
        metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        submissions.append((team_run, str(pred_path), metadata))
    
    return submissions


def main():
    """Score all submissions and update leaderboard.csv."""
    # Load existing leaderboard
    existing = load_existing_leaderboard()
    
    # Find all submissions
    submissions = find_all_submissions()
    
    # Score each submission
    today = datetime.now().strftime("%Y-%m-%d")
    new_entries = {}
    
    for team, csv_path, metadata in submissions:
        try:
            score = score_one(csv_path)
            
            # Get metadata info
            model_type = ""
            notes = ""
            if metadata:
                model_type = metadata.get("model_type", "")
                notes = metadata.get("notes", "")
            
            # Check if this is better than existing score for same team
            if team in existing:
                old_score = existing[team].get("score", 0)
                if score > old_score:
                    new_entries[team] = {
                        "team": team,
                        "model_type": model_type or existing[team].get("model_type", ""),
                        "score": score,
                        "date": today,
                        "notes": notes or existing[team].get("notes", "")
                    }
                else:
                    new_entries[team] = existing[team]
            else:
                new_entries[team] = {
                    "team": team,
                    "model_type": model_type,
                    "score": score,
                    "date": today,
                    "notes": notes
                }
            
            print(f"✅ {team}: {score:.6f}")
            
        except Exception as e:
            print(f"❌ {team}: {e}")
            # Keep existing entry if scoring fails
            if team in existing:
                new_entries[team] = existing[team]
    
    # Save updated leaderboard.csv
    save_leaderboard_csv(list(new_entries.values()))
    print(f"\n📊 Updated leaderboard/leaderboard.csv with {len(new_entries)} entries")


if __name__ == "__main__":
    main()
