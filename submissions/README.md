# Submission Guide

## Secure PR Submission (Encrypted)

1. Create your prediction CSV (accepted headers):
- `id,y_pred`
- `filename,prediction`

2. Encrypt your CSV with organizer public key:

```bash
python encryption/encrypt.py your_predictions.csv encryption/public_key.pem submissions/your_team.enc
```

Alternative helper:

```bash
cd submissions
python encrypt_submissions.py --input your_predictions.csv
cd ..
```

3. Open a Pull Request that adds only `submissions/*.enc`.

The automation decrypts and scores your file privately. Plaintext predictions are never published.

## Optional naming convention

Use `team__model.enc` to auto-populate model on the leaderboard.

Example:

```text
submissions/graph_wizards__human+llm.enc
```
