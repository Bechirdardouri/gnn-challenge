import pandas as pd
import sys

try:
    from .submission_utils import read_submission_csv, validate_submission_df
except ImportError:  # pragma: no cover - direct script execution path
    from submission_utils import read_submission_csv, validate_submission_df

def main(pred_path, test_nodes_path):
    preds = read_submission_csv(pred_path)
    test_nodes = pd.read_csv(test_nodes_path)
    validate_submission_df(preds, test_nodes)

    print("VALID SUBMISSION")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
