from __future__ import annotations

import argparse

try:
    from .submission_utils import read_label_csv, read_submission_csv, score_submission_df
except ImportError:  # pragma: no cover - direct script execution path
    from submission_utils import read_label_csv, read_submission_csv, score_submission_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a submission against private labels.")
    parser.add_argument("submission_csv", help="Path to submission CSV.")
    parser.add_argument("labels_csv", help="Path to labels CSV.")
    parser.add_argument(
        "--metric",
        default="auto",
        choices=["auto", "macro_f1", "roc_auc"],
        help="Scoring metric. Default auto-detects from submission format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = read_submission_csv(args.submission_csv)
    labels = read_label_csv(args.labels_csv)
    score = score_submission_df(preds, labels, metric=args.metric)
    print(f"SCORE={score:.8f}")


if __name__ == "__main__":
    main()
