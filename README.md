# HeteroShot GNN Challenge (Secure + SOTA-Ready)

Privacy-preserving competition template with automated scoring and real-time leaderboard updates.

## Live leaderboard

**https://bechirdardouri.github.io/gnn-challenge/leaderboard.html**

## What is included

- Secure encrypted submission pipeline (`submissions/*.enc` + GitHub Actions decryption with secrets)
- Google Form / Sheets / Drive ingestion pipeline
- Unified validation/scoring utilities (multiclass Macro-F1, plus auto metric detection)
- SOTA-ready starter submission model: `starter_code/sota_graph_ensemble.py`

## Quickstart (best baseline)

1. Train and generate predictions:

```bash
python starter_code/sota_graph_ensemble.py
```

2. Validate:

```bash
python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv
```

3. Encrypt:

```bash
python encryption/encrypt.py submissions/sota_ensemble_submission.csv encryption/public_key.pem submissions/<team_name>__<model>.enc
```

Note: `encryption/public_key.pem` must contain the real organizer public key (not the placeholder template key).

4. Open PR with only `.enc` submission files.

## SOTA starter details

`starter_code/sota_graph_ensemble.py` builds graph-aware features:
- raw node features
- 1-hop aggregated features
- 2-hop aggregated features
- degree statistics

Then it trains a multi-seed ExtraTrees ensemble for robust Macro-F1.

Recommended defaults:
- seeds: `7,13,77`
- estimators per model: `1400`

## Security model

- Public repo stores only encrypted submissions.
- Private labels stay outside git and are materialized in CI from secrets.
- Private key is stored in GitHub Actions secrets only.
- Scoring executes trusted base-branch code.

See `docs/SECURITY.md`.

## Repository layout

```text
.
├── .github/workflows/
│   ├── score_submission.yml
│   ├── process_google_form_submissions.yml
│   └── publish_leaderboard.yml
├── competition/
├── docs/
├── encryption/
├── starter_code/
├── submissions/
└── scripts/
```

## Required GitHub secrets

Encrypted PR method:
- `PRIVATE_KEY_PEM` (required)
- `PRIVATE_KEY_PASSWORD` (optional)

Private labels source (choose one):
- `PRIVATE_TEST_LABELS_CSV`, or
- `PRIVATE_DATA_METHOD` + source-specific secrets (`GOOGLE_DRIVE_FILE_ID`, `GOOGLE_CREDENTIALS_JSON`, `PRIVATE_DATA_URL`, `PRIVATE_DATA_S3_URI`)

Google Form method:
- `GOOGLE_SHEETS_ID`
- `GOOGLE_CREDENTIALS_JSON`

## Utilities

Environment check:

```bash
python check_setup.py
```

Leaderboard rendering:

```bash
python competition/render_leaderboard.py
```
