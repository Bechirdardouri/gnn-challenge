# Competition Module

Core validation, scoring, and leaderboard rendering utilities.

## Files

- `validate_submission.py`: validates submission schema and ID coverage.
- `evaluate.py`: evaluates a submission against private labels.
- `render_leaderboard.py`: regenerates markdown/json leaderboard artifacts.
- `submission_utils.py`: shared normalization + validation + scoring helpers.
- `metrics.py`: Macro-F1 and ROC-AUC metrics.

## Submission CSV formats

Accepted ID columns:
- `node_id` (preferred)
- `id`
- `filename`

Accepted prediction columns:
- `target` (preferred for this challenge)
- `prediction`
- `y_pred`

## Usage

Validate:

```bash
python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv
```

Evaluate:

```bash
python competition/evaluate.py submissions/sota_ensemble_submission.csv data/private/test_labels.csv --metric auto
```

Render leaderboard:

```bash
python competition/render_leaderboard.py
```
