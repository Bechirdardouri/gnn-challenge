# Organizer Guide

This guide covers setup for secure automated scoring.

## 1. Generate and Publish Keys

```bash
python encryption/generate_keys.py --force
```

- Commit only: `encryption/public_key.pem`
- Never commit private key
- Store private key content in GitHub secret: `PRIVATE_KEY_PEM`

## 2. Configure Private Labels

Preferred:
- Set `PRIVATE_TEST_LABELS_CSV` (full CSV content) in GitHub Secrets

Legacy-compatible option:
- Keep `data/test_labels.csv.enc` in the repo and set `TEST_LABELS_KEY`

Or use external source:
- `PRIVATE_DATA_METHOD` = `google_drive` | `url` | `s3`
- plus matching source secrets (`GOOGLE_DRIVE_FILE_ID`, `GOOGLE_CREDENTIALS_JSON`, `PRIVATE_DATA_URL`, `PRIVATE_DATA_S3_URI`)

## 3. Verify Workflows

- Encrypted PR scoring:
  - `.github/workflows/score_submission.yml`
- Merge-time fallback sync:
  - `.github/workflows/sync_leaderboard_from_encrypted_submissions.yml`
- Google Form ingestion:
  - `.github/workflows/process_google_form_submissions.yml`
- Leaderboard publish render:
  - `.github/workflows/publish_leaderboard.yml`

Private-label materialization is shared by all workflows via:
- `scripts/materialize_private_labels.py`

## 4. Optional Google Form Pipeline

- Setup details:
  - `docs/GOOGLE_APPS_SCRIPT_CODE.js`
- Required secrets:
  - `GOOGLE_SHEETS_ID`
  - `GOOGLE_CREDENTIALS_JSON`

## 5. Governance Checklist

- Protect `main` branch
- Restrict who can edit workflows and secrets
- Rotate keys per competition round
- Monitor workflow failures and PR comments

## 6. Local Smoke Tests

```bash
python check_setup.py
python competition/render_leaderboard.py
python scripts/materialize_private_labels.py --output data/private/test_labels.csv
```

If you have local private labels:

```bash
python competition/evaluate.py submissions/sota_ensemble_submission.csv data/private/test_labels.csv --metric auto
```

## Related Docs

- Overview: `README.md`
- Repo map: `docs/REPO_MAP.md`
- Security model: `docs/SECURITY.md`
