# Competition Module

This directory contains the evaluation and validation infrastructure for the HeteroShot challenge.

## Files

- `validate_submission.py` - Validate submission format and completeness
- `evaluate.py` - Score submissions against ground truth labels
- `render_leaderboard.py` - Generate leaderboard.md from leaderboard.csv
- `metrics.py` - Evaluation metrics (customize for different competitions)

## Usage

### Validate a submission
```bash
python competition/validate_submission.py --submission submissions/team_name.csv
```

### Evaluate a submission
```bash
python competition/evaluate.py --submission submissions/team_name.csv --truth data/test_labels.csv
```

### Render leaderboard
```bash
python competition/render_leaderboard.py
```

## Customization

To change the evaluation metric, edit `metrics.py` and update the `DEFAULT_METRIC` function.
