# Submissions Folder

This folder contains:

- generated plaintext prediction CSVs (local artifacts)
- encrypted submission files (`.enc`) for PRs
- sample/reference submissions

## Official Challenge Submission Flow

1. Generate predictions:

```bash
python starter_code/sota_graph_ensemble.py
```

2. Validate:

```bash
python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv
```

3. Encrypt:

```bash
python encryption/encrypt.py \
  submissions/sota_ensemble_submission.csv \
  encryption/public_key.pem \
  submissions/<team_name>__<model>.enc
```

4. Open PR with only `submissions/*.enc`.

For full instructions, see `docs/PARTICIPANT_GUIDE.md`.
