# Submissions Inbox

This directory is for participant submissions via Pull Request.

## Submission Structure

Each submission must follow this structure:
```
submissions/inbox/<team_name>/<run_id>/
├── predictions.csv
└── metadata.json
```

### predictions.csv
Required columns:
- `node_id`: Node identifier (must match test.csv)
- `target`: Predicted class label (integer)

### metadata.json
Required fields:
```json
{
  "model_type": "human|llm-only|human+llm",
  "notes": "Optional description of approach"
}
```

## How to Submit

1. Fork this repository
2. Create your submission directory: `submissions/inbox/<your_team>/<run_id>/`
3. Add your `predictions.csv` and `metadata.json`
4. Open a Pull Request to the main repository

Your submission will be automatically validated and scored via GitHub Actions.
