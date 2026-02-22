# Participant Guide

This guide is for teams submitting predictions.

## 1. Train and Generate Predictions

Recommended starter:

```bash
python starter_code/sota_graph_ensemble.py
```

This writes:
- `submissions/sota_ensemble_submission.csv`

Expected CSV schema:

```csv
node_id,target
2090,0
163,4
1650,3
```

## 2. Validate Before Submitting

```bash
python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv
```

If validation fails:
- check for duplicate `node_id`
- ensure all `node_id` from `data/test.csv` are present
- ensure `target` is integer class label

## 3. Encrypt Your File

```bash
python encryption/encrypt.py \
  submissions/sota_ensemble_submission.csv \
  encryption/public_key.pem \
  submissions/<team_name>__<model>.enc
```

Naming tip:
- Use `team__model.enc` (example: `graph_wizards__human+llm.enc`)

## 4. Submit Via Pull Request

- Open a PR to `main`
- Include only encrypted file(s): `submissions/*.enc`
- Do not commit plaintext predictions or secrets

The scoring workflow comments your result and updates the leaderboard automatically.

## Quick Troubleshooting

- `Failed to read public key`
  - Organizer has not published a real key yet; `encryption/public_key.pem` is placeholder.
- `Prediction IDs do not match expected test node IDs`
  - Rebuild your CSV directly from `data/test.csv`.
- `Failed` in PR comment
  - Read the workflow run logs and bot comment details.

## Related Docs

- Overview: `README.md`
- Repo map: `docs/REPO_MAP.md`
- Security model: `docs/SECURITY.md`
