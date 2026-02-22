# Competition Core

Everything here is part of the trusted scoring/validation pipeline.

## Main scripts

- `validate_submission.py`
  - checks schema, duplicates, missing IDs, and coverage against test-node list
- `evaluate.py`
  - scores one submission against label file
  - `--metric auto`:
    - uses Macro-F1 for class-label predictions
    - can use ROC-AUC for probability-style binary predictions
- `render_leaderboard.py`
  - regenerates:
    - `leaderboard/leaderboard.md`
    - `docs/leaderboard.json`

## Shared helpers

- `submission_utils.py`
  - accepts multiple column aliases (`node_id|id|filename`, `target|prediction|y_pred`)
- `metrics.py`
  - `macro_f1`
  - `binary_auc`
- `leaderboard_utils.py`
  - read/write leaderboard CSV rows

## Typical local commands

```bash
# validate
python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv

# evaluate (requires private labels)
python competition/evaluate.py submissions/sota_ensemble_submission.csv data/private/test_labels.csv --metric auto

# regenerate leaderboard artifacts
python competition/render_leaderboard.py
```
