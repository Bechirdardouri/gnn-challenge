# HeteroShot: Few-Shot Node Classification Challenge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
HeteroShot is a mini competition focused on few-shot node classification with noisy labels
and missing features. Participants submit predictions as a CSV and receive a Macro-F1 score
automatically via GitHub Actions.

 **[View Live Leaderboard](leaderboard.md)**

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

##  Evaluation Metric
**Macro-F1** on hidden test labels - equal weight to all classes regardless of frequency.

##  Rules
-  No external data allowed
-  Must run on CPU in under 5 minutes
-  Any GNN architecture or sampling method permitted
-  Standard ML libraries allowed (PyTorch, scikit-learn, etc.)

##  Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd gnn-challenge

# Install dependencies
pip install -r starter_code/requirements.txt

# Generate dataset
python data/make_dataset.py

# Verify setup
python check_setup.py
```

### Run Baselines
```bash
# Tabular baseline (Random Forest)
cd starter_code
python baseline_tabular.py

# GNN baseline (GraphSAGE)
python baseline_gnn.py
```

##  How to Submit
1. **Fork this repository**
2. **Create your solution** and generate predictions
3. **Add your submission**: `submissions/<your_team_name>.csv`
   - Required columns: `node_id`, `target`
   - Must include predictions for all test nodes
4. **Open a Pull Request**
   - A bot will automatically comment your Macro-F1 score
   - Upon merge, the leaderboard updates automatically

##  Baselines

| Model | Description | Val F1 | Test F1 |
|-------|-------------|--------|---------|
| Random Forest | `baseline_tabular.py` - Features only | 0.214 | 0.181 |
| GraphSAGE | `baseline_gnn.py` - GNN with 2 layers | 0.221 | 0.181 |

## 🔧 Scoring Workflow

### Automated PR Scoring
1. `scoring/scoring_script.py` validates submission format and computes Macro-F1
2. GitHub Actions workflow downloads submission from PR and scores it
3. Bot comments the score on the PR

### Leaderboard Updates
- On merge to `main`, `scoring/update_leaderboard.py` recomputes all scores
- `leaderboard.md` is automatically updated and committed

##  Encrypted Test Labels
Test labels are encrypted (`data/test_labels.csv.enc`) to prevent cheating.

**For maintainers:** To enable scoring workflows, add a repository secret `TEST_LABELS_KEY`.

### Generating Encrypted Labels
```bash
python data/make_dataset.py
export TEST_LABELS_KEY="your-strong-passphrase"
openssl enc -aes-256-cbc -salt -pbkdf2 \
  -in data/test_labels.csv \
  -out data/test_labels.csv.enc \
  -pass pass:$TEST_LABELS_KEY
rm -f data/test_labels.csv
```

##  Repository Structure
```
.
├── data/                      # Dataset files
│   ├── make_dataset.py       # Dataset generation script
│   ├── edges.csv             # Graph structure
│   ├── train.csv             # Training nodes (few-shot + noisy)
│   ├── val.csv               # Validation nodes
│   ├── test.csv              # Test nodes (unlabeled)
│   └── test_labels.csv.enc   # Encrypted ground truth
├── starter_code/             # Baseline implementations
│   ├── baseline_tabular.py   # Random Forest baseline
│   ├── baseline_gnn.py       # GraphSAGE baseline
│   └── requirements.txt      # Python dependencies
├── scoring/                  # Scoring infrastructure
│   ├── scoring_script.py     # Compute Macro-F1
│   └── update_leaderboard.py # Generate leaderboard
├── submissions/              # Participant submissions
│   └── *.csv                 # Team predictions
├── .github/workflows/        # CI/CD automation
│   ├── score_pr.yml         # Score PRs automatically
│   └── update_leaderboard.yml # Update leaderboard on merge
└── leaderboard.md           # Current rankings
```

##  Tips for Success
- Handle class imbalance carefully (Macro-F1 treats all classes equally)
- Leverage graph structure - tabular methods ignore valuable information
- Robust to label noise - training labels are 12% corrupted
- Deal with missing features - 30% feature dropout applied
- Few-shot learning - only 20 labeled nodes per class for training

##  License
MIT License - see [LICENSE](LICENSE) file for details .

##  Contributing
Issues and pull requests are welcome! For major changes, please open an issue first.

---

**Happy modeling!**

