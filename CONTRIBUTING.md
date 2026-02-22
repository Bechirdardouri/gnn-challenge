# Contributing And Submitting

This challenge uses **encrypted submissions** for privacy-preserving scoring.

## Submission format

Your plaintext CSV should contain:

```csv
node_id,target
2090,0
163,1
1650,3
```

Accepted alternatives:
- ID column: `node_id`, `id`, or `filename`
- prediction column: `target`, `prediction`, or `y_pred`

## Recommended baseline (SOTA-ready starter)

```bash
python starter_code/sota_graph_ensemble.py
```

This writes `submissions/sota_ensemble_submission.csv`.

## Encrypt before PR

```bash
python encryption/encrypt.py submissions/sota_ensemble_submission.csv encryption/public_key.pem submissions/<team_name>.enc
```

Only submit `.enc` files in PRs.

## Pull request checklist

- [ ] I only added encrypted submission files (`submissions/*.enc`)
- [ ] I did not modify scoring scripts or workflows
- [ ] My submission covers all test node IDs
- [ ] No private labels or secrets are committed

## Validation helpers

```bash
python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv
python check_setup.py
```
