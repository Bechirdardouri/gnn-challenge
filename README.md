# GNN Competition Template (Privacy-Preserving)

This repository provides a secure competition template where participants submit
predictions only. Test labels remain private and leaderboard updates are fully automated.

Two submission channels are supported:

1. Method I: Pull Request + Encryption (private `.enc` submissions)
2. Method II: Google Form + Google Sheet + Google Drive

Both methods use the same validator/scorer in `competition/`.

## Why this is private

- Test labels are never committed publicly.
- Participants submit encrypted files or private Drive uploads.
- Decryption/scoring runs only in GitHub Actions with secrets.
- Raw plaintext predictions are never pushed to the public repository.

Details: `docs/SECURITY.md`

## Repository structure

```text
.
├── competition/
│   ├── validate_submission.py
│   ├── evaluate.py
│   ├── render_leaderboard.py
│   ├── leaderboard_utils.py
│   └── submission_utils.py
├── encryption/
│   ├── generate_keys.py
│   ├── encrypt.py
│   ├── decrypt.py
│   └── public_key.pem
├── scripts/
│   ├── process_pr_encrypted_submissions.py
│   ├── process_google_form_submissions.py
│   └── download_private_data.py
├── submissions/
│   ├── README.md
│   └── encrypt_submissions.py
├── leaderboard/
│   ├── leaderboard.csv
│   └── leaderboard.md
└── .github/workflows/
    ├── score_submission.yml
    ├── process_google_form_submissions.yml
    └── publish_leaderboard.yml
```

## Method I: Pull Request + Encryption

### Participant workflow

1. Train model offline.
2. Create prediction CSV (`id,y_pred` or `filename,prediction`).
3. Encrypt:

```bash
python encryption/encrypt.py your_predictions.csv encryption/public_key.pem submissions/your_team.enc
```

4. Open PR adding `submissions/your_team.enc`.
5. Workflow scores automatically and updates leaderboard.

### Organizer setup

1. Generate keys:

```bash
python encryption/generate_keys.py --force
```

2. Commit `encryption/public_key.pem`.
3. Add secret `PRIVATE_KEY_PEM` with private key content.
4. Add private labels using either:
- Secret `PRIVATE_TEST_LABELS_CSV`, or
- `PRIVATE_DATA_METHOD` + source secrets (`GOOGLE_DRIVE_FILE_ID`, `GOOGLE_CREDENTIALS_JSON`, `PRIVATE_DATA_URL`, `PRIVATE_DATA_S3_URI`).
5. Ensure workflow permissions allow `contents: write` and `pull-requests: write`.

### Security model for PR scoring

- Uses `pull_request_target`.
- Executes trusted code from base branch.
- Reads changed `.enc` files through GitHub API.
- Never executes code from contributor branch.

## Method II: Google Form + Sheet + Drive

### Required components

1. Google Form fields:
- Team Name (required)
- Model Type (required)
- Submission File (CSV upload, required)

2. Google Cloud setup:
- Enable Google Sheets API and Google Drive API
- Create Service Account + JSON key
- Share Sheet and Drive upload folder with Service Account email

3. GitHub secrets:
- `GOOGLE_SHEETS_ID`
- `GOOGLE_CREDENTIALS_JSON`
- `GOOGLE_DRIVE_FILE_ID` (optional, for private labels download method)
- `PRIVATE_DATA_METHOD` (optional: `google_drive`, `url`, `s3`)

### Automation path

Google Form -> Google Sheet -> Apps Script trigger -> `repository_dispatch` -> GitHub Action -> score -> leaderboard update

Apps Script template: `docs/GOOGLE_APPS_SCRIPT_CODE.js`

### Apps Script webhook setup (recommended)

1. Create GitHub Personal Access Token (classic) with scopes:
- `repo`
- `workflow`

2. In the Google Sheet:
- Open `Extensions -> Apps Script`
- Paste `docs/GOOGLE_APPS_SCRIPT_CODE.js`
- Set `CONFIG.owner` and `CONFIG.repo`
- Run `setupGitHubPAT("ghp_...")` once

3. Add trigger:
- Function: `onFormSubmit`
- Event source: `From form`
- Event type: `On form submit`

4. After validation:
- Remove/comment PAT bootstrap calls in code
- Keep token only in Script Properties

## Leaderboard outputs

- CSV: `leaderboard/leaderboard.csv`
- Markdown: `leaderboard/leaderboard.md`
- JSON for docs UI: `docs/leaderboard.json`
- Interactive page: `docs/leaderboard.html`

Enable GitHub Pages on `/docs` and open:

```text
https://<owner>.github.io/<repo>/leaderboard.html
```

## Local validation and scoring

Validate submission format:

```bash
python competition/validate_submission.py your_submission.csv data/public/test_nodes.csv
```

Score using private labels:

```bash
python competition/evaluate.py your_submission.csv data/private/test_labels.csv
```

## Notes

- `encryption/public_key.pem` in this template is a placeholder. Replace it before launch.
- Keep private keys and private labels out of git.
- Use branch protection rules for `main`.
