# HeteroShot GNN Challenge

Privacy-preserving node-classification challenge with automated scoring and a public leaderboard.

## Live Leaderboard

**https://bechirdardouri.github.io/gnn-challenge/leaderboard.html**

## Start Here

If you are a participant:
- Read `docs/PARTICIPANT_GUIDE.md`
- Run `python starter_code/sota_graph_ensemble.py`
- Validate + encrypt + submit `.enc` via PR

If you are an organizer:
- Read `docs/ORGANIZER_GUIDE.md`
- Configure GitHub secrets and workflows
- Keep private labels/private key out of git

If you want the structure overview:
- Read `docs/REPO_MAP.md`

If you want quick answers:
- Read `docs/FAQ.md`

## Fast Participant Workflow

```bash
# 1) Generate predictions
python starter_code/sota_graph_ensemble.py

# 2) Validate coverage/format
python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv

# 3) Encrypt for private submission
python encryption/encrypt.py \
  submissions/sota_ensemble_submission.csv \
  encryption/public_key.pem \
  submissions/<team_name>__<model>.enc
```

Then open a PR with only `submissions/*.enc`.

Note:
- `encryption/public_key.pem` must be the real organizer key (not placeholder text).

## What This Repo Includes

- Secure encrypted PR scoring: `.github/workflows/score_submission.yml`
- Optional Google Form pipeline: `.github/workflows/process_google_form_submissions.yml`
- Leaderboard rendering/publishing: `.github/workflows/publish_leaderboard.yml`
- Merge-time fallback sync (encrypted submissions): `.github/workflows/sync_leaderboard_from_encrypted_submissions.yml`
  - also supports legacy plaintext CSV backfill on `main` for existing public submissions
- Shared validation/scoring utilities: `competition/`
- Encryption tools: `encryption/`
- Starter baselines: `starter_code/`

## SOTA Starter Baseline

`starter_code/sota_graph_ensemble.py` uses:
- raw node features
- 1-hop and 2-hop graph-aggregated features
- degree statistics
- multi-seed ExtraTrees ensembling

Recommended defaults:
- `--seeds 7,13,77`
- `--n-estimators 1400`
- `--max-features sqrt`

## Useful Commands

```bash
# Show common project commands
make help

# Check environment
python check_setup.py

# Render leaderboard artifacts after leaderboard.csv changes
python competition/render_leaderboard.py

# Evaluate locally (when private labels are available)
python competition/evaluate.py submissions/sota_ensemble_submission.csv data/private/test_labels.csv --metric auto

# Materialize private labels from configured secrets/sources
python scripts/materialize_private_labels.py --output data/private/test_labels.csv
```

## If Leaderboard Shows "No Results"

1. Open GitHub Actions and run:
   - `Sync Leaderboard From Encrypted Submissions` (manual `workflow_dispatch` run)
2. Confirm one private-label source is configured:
   - `PRIVATE_TEST_LABELS_CSV` (preferred), or
   - `TEST_LABELS_KEY` with `data/test_labels.csv.enc`, or
   - `PRIVATE_DATA_METHOD` + matching source secrets
3. Confirm secure PR submissions are encrypted `.enc` files (plaintext CSVs in PR are not scored).

## Security Model (Short Version)

- Public repo stores encrypted submissions (`.enc`) only.
- Decryption happens only in GitHub Actions using `PRIVATE_KEY_PEM`.
- Private labels are provided at runtime via secrets.
- Legacy private-label decrypt is supported via `TEST_LABELS_KEY` + `data/test_labels.csv.enc`.
- Trusted base-branch code performs evaluation.
- Fallback merge-time sync ensures merged encrypted submissions are reflected on leaderboard.

See `docs/SECURITY.md` for the full model.
