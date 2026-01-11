
## Overview
HeteroShot is a mini competition focused on few-shot node classification with noisy labels
and missing features. Participants submit predictions as a CSV and receive a Macro-F1 score
automatically via GitHub Actions.

📊 **[View Live Leaderboard](leaderboard.md)**

## Task
Predict node class labels on a graph (node classification). The dataset is derived from
the Actor dataset with:
- few-shot training labels per class
- label noise in the training split
- feature dropout applied to all splits

## Dataset files
- `data/train.csv`: labeled nodes (few-shot + noisy)
- `data/val.csv`: labeled nodes (public tuning)
- `data/test.csv`: unlabeled nodes for scoring
- `data/edges.csv`: graph edges (src, dst)
- `data/test_labels.csv.enc`: encrypted hidden labels used for scoring

To generate the dataset locally:

```bash
python data/make_dataset.py
```

## Metric
Macro-F1 on hidden test labels.

## Rules
- No external data.
- Must run on CPU in under 5 minutes.
- You may use any GNN / sampling method.

## How to submit
1. Fork the repo
2. Add your file: `submissions/<team>.csv`
3. CSV must have columns: `node_id,target`
4. Open a Pull Request

A bot will comment your Macro-F1 score.

## Baselines
- `starter_code/baseline_tabular.py` (RandomForest)
- `starter_code/baseline_gnn.py` (GraphSAGE + NeighborSampler)

## Scoring and leaderboard workflow
1. `scoring/scoring_script.py` validates the submission and prints `MACRO_F1=...`.
2. `pull_request_target` workflow downloads only the submission CSV and scores it.
3. On merge to `main`, `scoring/update_leaderboard.py` recomputes all scores and
   overwrites `leaderboard.md`.

## Secrets and encrypted labels
The test labels are encrypted and committed as `data/test_labels.csv.enc`. To enable
scoring workflows, add a repository secret named `TEST_LABELS_KEY` with the passphrase
used to encrypt the file.

To generate and encrypt labels locally:

```bash
python data/make_dataset.py
export TEST_LABELS_KEY="your-strong-passphrase"
openssl enc -aes-256-cbc -salt -pbkdf2 \
  -in data/test_labels.csv \
  -out data/test_labels.csv.enc \
  -pass pass:$TEST_LABELS_KEY
rm -f data/test_labels.csv
```

## Repository layout
```
.
├── data/
├── scoring/
├── starter_code/
├── submissions/
└── .github/workflows/
```

