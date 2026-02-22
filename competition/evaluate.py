import sys

try:
    from .submission_utils import read_label_csv, read_submission_csv, score_submission_df
except ImportError:  # pragma: no cover - direct script execution path
    from submission_utils import read_label_csv, read_submission_csv, score_submission_df

def main(pred_path, label_path):
    preds = read_submission_csv(pred_path)
    labels = read_label_csv(label_path)
    score = score_submission_df(preds, labels)
    print(f"SCORE={score:.8f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
