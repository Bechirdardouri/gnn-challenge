# Submission Guide

## 1. Generate predictions

Recommended:

```bash
python starter_code/sota_graph_ensemble.py
```

This creates `submissions/sota_ensemble_submission.csv`.

## 2. Validate locally

```bash
python competition/validate_submission.py submissions/sota_ensemble_submission.csv data/test.csv
```

## 3. Encrypt the CSV

```bash
python encryption/encrypt.py submissions/sota_ensemble_submission.csv encryption/public_key.pem submissions/<team_name>__<model>.enc
```

## 4. Submit via PR

Open a pull request that includes only `submissions/*.enc`.

### Accepted plaintext CSV schemas

Preferred:
- `node_id,target`

Also accepted:
- `id,prediction`
- `filename,y_pred`
