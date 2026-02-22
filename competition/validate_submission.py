from __future__ import annotations

import argparse
import pandas as pd

try:
    from .submission_utils import read_submission_csv, validate_submission_df
except ImportError:  # pragma: no cover - direct script execution path
    from submission_utils import read_submission_csv, validate_submission_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate submission format and coverage.")
    parser.add_argument("submission_csv", help="Path to submission CSV.")
    parser.add_argument("test_nodes_csv", help="Path to expected test-node IDs CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = read_submission_csv(args.submission_csv)
    test_nodes = pd.read_csv(args.test_nodes_csv)
    validate_submission_df(preds, test_nodes)
    print("VALID SUBMISSION")


if __name__ == "__main__":
    main()
