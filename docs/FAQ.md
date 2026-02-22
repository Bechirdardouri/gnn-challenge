# FAQ

## Why do I need to submit `.enc` files?

To keep private test labels safe and prevent other teams from seeing raw predictions in public PRs.

## My encryption command fails with public key error.

`encryption/public_key.pem` is probably still a placeholder. Ask organizers for the real key.

## Can I submit plaintext CSV in a PR?

No. Live scoring is designed for encrypted submissions only.

## Which metric is used?

For this challenge: Macro-F1 on class labels.  
The evaluation utility also supports ROC-AUC for probability-style binary tasks when needed.

## Where do I see my score?

Live leaderboard:

https://bechirdardouri.github.io/gnn-challenge/leaderboard.html

You should also get a PR comment from the scoring workflow.

If leaderboard does not update after merge:
- check workflow runs:
  - `Score Encrypted PR Submission`
  - `Sync Leaderboard From Encrypted Submissions`
  - `Publish Leaderboard`
- confirm required secrets are configured (especially `PRIVATE_KEY_PEM` and private labels source)
- if using legacy encrypted labels file (`data/test_labels.csv.enc`), ensure `TEST_LABELS_KEY` is set

## I am an organizer. Where do I start?

Read `docs/ORGANIZER_GUIDE.md`.
