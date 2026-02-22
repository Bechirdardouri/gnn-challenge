# Repo Map

Short guide to where things live.

## Core folders

- `competition/`
  - validation/scoring/rendering logic
- `encryption/`
  - key generation and encrypt/decrypt utilities
- `scripts/`
  - automation scripts used by CI workflows
- `starter_code/`
  - baseline and SOTA-ready model scripts
- `submissions/`
  - participant plaintext examples and encrypted submission targets
- `leaderboard/`
  - source leaderboard CSV + generated markdown
- `docs/`
  - security docs, app script template, pages assets

## GitHub workflows

- `score_submission.yml`
  - secure PR scoring for `.enc` submissions
- `process_google_form_submissions.yml`
  - pulls Google Form submissions and updates leaderboard
- `publish_leaderboard.yml`
  - renders markdown/json leaderboard artifacts on updates
- `sync_leaderboard_from_encrypted_submissions.yml`
  - fallback scorer that syncs leaderboard from merged `.enc` files

## Data files

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- `data/edges.csv`

Private labels are not stored in plaintext in git.

## First files to read

1. `README.md`
2. `docs/PARTICIPANT_GUIDE.md` or `docs/ORGANIZER_GUIDE.md`
3. `docs/SECURITY.md`

## Convenience Commands

- `Makefile` provides shortcuts (`make help`).
