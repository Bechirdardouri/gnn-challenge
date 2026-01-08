import csv
from pathlib import Path


def _read_labels(path):
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


def main():
    subs = sorted(Path("submissions").glob("*.csv"))
    rows = []
    for path in subs:
        team = path.stem
        try:
            s = score_one(str(path))
            rows.append((team, s))
        except Exception:
            rows.append((team, None))

    rows.sort(key=lambda item: (item[1] is None, -(item[1] or 0)))

    lines = []
    lines.append("# Leaderboard")
    lines.append("")
    lines.append("| Rank | Team | Macro-F1 |")
    lines.append("|------|------|----------|")

    rank = 1
    for team, score in rows:
        if score is None:
            lines.append(f"| - | {team} | invalid |")
        else:
            lines.append(f"| {rank} | {team} | {score:.6f} |")
            rank += 1

    with open("leaderboard.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Updated leaderboard.md")


if __name__ == "__main__":
    main()
