# HeteroShot GNN Challenge

A privacy-first node-classification competition template with automated secure scoring and a live public leaderboard.

## Live Links

- Live leaderboard: [Open Live Leaderboard](https://bechirdardouri.github.io/gnn-challenge/leaderboard.html)
- Repository: [Bechirdardouri/gnn-challenge](https://github.com/Bechirdardouri/gnn-challenge)

## Competition At A Glance

- Task: node classification on graph-structured data
- Public data: `data/train.csv`, `data/val.csv`, `data/test.csv`, `data/edges.csv`
- Hidden data: private test labels (never committed in plaintext)
- Main metric: Macro-F1 for class-label submissions
- Privacy model: predictions are submitted encrypted (`.enc`) and scored in trusted GitHub Actions
- Output: leaderboard updates are committed automatically and published via GitHub Pages

## Challenge Context

This competition is intentionally designed as a low-label graph problem:

- `train.csv`: 100 labeled nodes
- `val.csv`: 300 labeled nodes
- `test.csv`: 7200 unlabeled nodes for submission
- `edges.csv`: 53,411 graph edges over 7,600 total nodes

What makes it interesting:

- You have very little supervised signal relative to test size
- Performance depends on combining feature learning and graph structure
- Evaluation is private and automated, so leaderboard gains must come from genuine generalization

## Dataset Quirks And Hidden Traps

These are the main pitfalls teams should account for:

- Label scarcity is real:
  - only 400 labeled nodes total (`train + val`) vs 7,200 test nodes
- Class prior mismatch risk:
  - `val` is perfectly balanced across 5 classes, while `train` is slightly skewed
- Graph degree is heavy-tailed:
  - median total degree is 8, 95th percentile is 46, max degree is 2606
  - hub nodes can dominate naive neighbor aggregation
- Graph signal is not purely homophilous:
  - on known-known labeled edges, same-class rate is around 0.24 (close to 0.20 random baseline for 5 classes)
  - aggressive same-label propagation can hurt
- Direct label reach into test is limited:
  - only about 25% of test nodes have a direct edge to a labeled node
  - relying on 1-hop supervision alone leaves most test nodes weakly informed

## Training And Data Recommendations

Recommended strategy for strong, stable submissions:

1. Treat this as hybrid tabular + graph learning
   - start from feature-strong models
   - add graph-derived features (1-hop/2-hop aggregates, degree/log-degree)
2. Use `val` for serious model selection
   - tune seeds, depth/regularization, and feature mixing on Macro-F1
   - avoid single-seed conclusions; use seed ensembles for stability
3. Control hub effects
   - use degree-normalized aggregation
   - include explicit degree features so model can adapt to hub vs non-hub nodes
4. Handle class imbalance cautiously
   - class-balanced losses/weights can help on small-train settings
   - validate calibration behavior on `val`, not only raw train fit
5. Reduce variance before final test inference
   - after selecting hyperparameters, retrain on `train + val`
   - ensemble multiple seeds/models where feasible
6. Keep submissions strict and reproducible
   - validate `node_id` coverage against `data/test.csv`
   - keep a deterministic pipeline from training to CSV to encryption

## Why This Template Is Strong

- Fair evaluation: hidden labels are not exposed to participants
- Transparent operations: scoring logic is versioned in repo code
- Reproducible workflow: validation, encryption, scoring, and rendering are scripted
- Fast feedback: leaderboard updates automatically after successful evaluation runs

## Start Here

### For Participants

1. Read `docs/PARTICIPANT_GUIDE.md`
2. Train and generate predictions
3. Validate CSV format and coverage
4. Encrypt submission with organizer public key
5. Open PR containing only `submissions/*.enc`

Quick path:

```bash
# 1) Generate starter SOTA predictions
python starter_code/sota_graph_ensemble.py --output submissions/sota_ensemble_submission.csv

# 2) Validate your file
python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv

# 3) Encrypt (replace team/model names)
python encryption/encrypt.py \
  submissions/sota_ensemble_submission.csv \
  encryption/public_key.pem \
  submissions/<team_name>__<model_name>.enc
```

### For Organizers

1. Read `docs/ORGANIZER_GUIDE.md`
2. Configure secrets for private labels and private key
3. Verify Actions workflows are enabled with write permissions
4. Run a smoke test submission to confirm end-to-end scoring

## Submission Channels

### Channel A: Encrypted Pull Requests (recommended)

1. Participant generates `node_id,target` CSV offline
2. Participant encrypts with `encryption/public_key.pem`
3. Participant submits `.enc` in PR
4. `Score Encrypted PR Submission` workflow decrypts and scores in trusted runner
5. Leaderboard artifacts update automatically

Workflow: `.github/workflows/score_submission.yml`

### Channel B: Google Form + Drive (optional)

1. Team uploads CSV through Google Form
2. Google Sheet + Apps Script trigger dispatches workflow
3. Workflow downloads submissions from Drive and scores them
4. Leaderboard updates automatically

Workflow: `.github/workflows/process_google_form_submissions.yml`

## How Leaderboard Updates Work

1. Submissions are scored against private labels
2. `leaderboard/leaderboard.csv` is updated
3. `competition/render_leaderboard.py` regenerates `leaderboard/leaderboard.md` and `docs/leaderboard.json`
4. GitHub Pages serves the live UI from `docs/`

Primary publish workflows:
- `.github/workflows/sync_leaderboard_from_encrypted_submissions.yml`
- `.github/workflows/publish_leaderboard.yml`

## Security And Privacy Model

- Public repository stores encrypted submissions, not plaintext predictions
- Private key is injected at runtime from secret `PRIVATE_KEY_PEM`
- Private labels are materialized at runtime by `scripts/materialize_private_labels.py`
- Trusted base-branch workflow code performs scoring
- PR scorer reads submission files via GitHub API and avoids running untrusted PR code

Private label sources supported:
- `PRIVATE_TEST_LABELS_CSV` (preferred)
- `PRIVATE_TEST_LABELS_CSV_GZIP_B64` (for large secrets)
- `TEST_LABELS_KEY` + `data/test_labels.csv.enc` (legacy mode)
- External source via `PRIVATE_DATA_METHOD` (`google_drive`, `url`, `s3`)

Deep dive: `docs/SECURITY.md`

## Repository Structure

- `competition/`: validation, metrics, evaluation, rendering
- `encryption/`: key utilities and encryption/decryption scripts
- `starter_code/`: baseline and stronger starter models
- `scripts/`: CI helpers for private data and scoring pipelines
- `submissions/`: participant submissions and examples
- `leaderboard/`: canonical leaderboard data files
- `docs/`: live UI assets and operational documentation

## Useful Commands

```bash
# Environment and dependency checks
python check_setup.py

# Show shortcut commands
make help

# Build starter submission
make train-sota

# Validate a submission
make validate-sota

# Render leaderboard markdown/json from leaderboard.csv
make render-leaderboard

# Local evaluation (requires local private labels)
python competition/evaluate.py submissions/sota_ensemble_submission.csv data/private/test_labels.csv --metric auto
```

## Leaderboard Troubleshooting

If the leaderboard UI shows no rows:

1. Open Actions and inspect latest run of `Sync Leaderboard From Encrypted Submissions`
2. Confirm at least one private-label source secret is configured
3. Re-run sync workflow manually (`workflow_dispatch`)
4. Confirm run steps `Materialize private labels` and `Process repository submissions` are green
5. Hard refresh leaderboard page after Pages deploy completes

## Professional Standards

- Do not commit private keys or plaintext private labels
- Keep submission PRs limited to encrypted files
- Keep workflow and scoring changes documented in PR descriptions
- Preserve reproducibility: prefer explicit scripts over ad-hoc manual edits

## Additional Docs

- Participant instructions: `docs/PARTICIPANT_GUIDE.md`
- Organizer setup: `docs/ORGANIZER_GUIDE.md`
- Security model: `docs/SECURITY.md`
- Repo map: `docs/REPO_MAP.md`
- FAQ: `docs/FAQ.md`
- Contribution guide: `CONTRIBUTING.md`
