# Contributing to HeteroShot Challenge

Thank you for your interest in contributing! This document provides guidelines for participating in the challenge.

## 🎯 Submission Guidelines

### Format Requirements
Your submission CSV must:
- Have exactly 2 columns: `node_id` and `target`
- Include predictions for all test nodes (check `data/test.csv`)
- Use integer class labels (0, 1, 2, ...)
- Have no missing values

Example format:
```csv
node_id,target
2090,0
163,1
1650,4
4045,2
```

### File Naming
- Place your submission in `submissions/<team_name>.csv`
- Use a unique team name (e.g., `submissions/awesome_gnn.csv`)
- Only alphanumeric characters and underscores in filename

## 📋 Submission Process

1. **Fork the Repository**
   ```bash
   # Fork via GitHub UI, then clone
   git clone https://github.com/<your-username>/gnn-challenge.git
   cd gnn-challenge
   ```

2. **Create Your Solution**
   ```bash
   # Install dependencies
   pip install -r starter_code/requirements.txt
   
   # Develop your model
   # Generate predictions and save to submissions/<team_name>.csv
   ```

3. **Validate Locally** (Optional)
   ```bash
   # Check format
   python -c "import pandas as pd; df = pd.read_csv('submissions/<team_name>.csv'); assert list(df.columns) == ['node_id', 'target']; print('✓ Format valid')"
   ```

4. **Submit via Pull Request**
   ```bash
   git add submissions/<team_name>.csv
   git commit -m "Add <team_name> submission"
   git push origin main
   # Open PR on GitHub
   ```

5. **Wait for Automated Scoring**
   - A bot will comment your Macro-F1 score within minutes
   - Fix any errors and update your PR if needed

## ✅ Validation Rules

Your submission will be checked for:
- Correct CSV format with required columns
- Predictions for all test nodes
- Integer class labels only
- No duplicate node IDs

## 🚫 What Not to Do

- ❌ Do not use external datasets
- ❌ Do not access test labels (they're encrypted)
- ❌ Do not submit solutions that take >5 minutes on CPU
- ❌ Do not copy other participants' code without attribution

## 🐛 Reporting Issues

Found a bug or have a question?
1. Check existing issues first
2. Open a new issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Your environment details

## 🤝 Code of Conduct

- Be respectful and constructive
- Help others learn and improve
- Share ideas and insights
- Give credit where due

## 📬 Questions?

Open an issue with the `question` label, and we'll help you out!

---

Good luck! 🎉
