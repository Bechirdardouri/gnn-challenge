# Privacy And Security Model

This competition template supports two secure submission channels:

1. Encrypted Pull Requests (`submissions/*.enc`)
2. Google Form + Google Drive uploads

Both channels are evaluated by trusted GitHub Actions code from the base branch.

## Encrypted PR Flow

1. Participants create predictions CSV offline.
   - Preferred schema: `node_id,target`
2. Participants encrypt the CSV using `encryption/public_key.pem`.
3. Participants submit only `.enc` files in pull requests.
4. Workflow decrypts submissions using `PRIVATE_KEY_PEM` from GitHub Secrets.
5. Workflow validates and scores against hidden labels (`data/private/test_labels.csv` materialized at runtime).
6. Leaderboard files are updated automatically.

Key guarantees:
- Test labels are never committed to the repository.
- Private key is never committed to the repository.
- Raw plaintext predictions are never committed to the repository.
- The scoring job reads files from GitHub API and does not execute untrusted PR code.

## Google Form Flow

1. Teams upload CSV via Google Form file upload.
2. Submission metadata is stored in Google Sheet.
3. Apps Script trigger dispatches `repository_dispatch` to GitHub.
4. Workflow reads Sheet rows, downloads CSVs from Drive with Service Account credentials.
5. Workflow validates/scores and updates leaderboard.

## Required GitHub Secrets

General:
- `PRIVATE_TEST_LABELS_CSV` (preferred) or external private-data source secrets.
- `PRIVATE_TEST_LABELS_CSV_GZIP_B64` (gzip+base64 variant when raw CSV exceeds secret size limit).
- `TEST_LABELS_KEY` (legacy option with `data/test_labels.csv.enc`).
- `PRIVATE_DATA_METHOD` (`google_drive` | `url` | `s3`) if not using inline labels.
- `GOOGLE_DRIVE_FILE_ID`, `GOOGLE_CREDENTIALS_JSON`, `PRIVATE_DATA_URL`, `PRIVATE_DATA_S3_URI` as needed.

Encrypted PR:
- `PRIVATE_KEY_PEM` (required)
- `PRIVATE_KEY_PASSWORD` (optional)

Google Form:
- `GOOGLE_SHEETS_ID` (required)
- `GOOGLE_CREDENTIALS_JSON` (required)

## Operational Recommendations

- Rotate keys regularly (at least each competition round).
- Restrict who can edit workflows and secrets.
- Use branch protections on `main`.
- Keep workflows pinned to trusted actions versions.
- Keep private-label materialization centralized (`scripts/materialize_private_labels.py`) to avoid drift.

## Related Docs

- Project overview: `README.md`
- Participant flow: `docs/PARTICIPANT_GUIDE.md`
- Organizer setup: `docs/ORGANIZER_GUIDE.md`
