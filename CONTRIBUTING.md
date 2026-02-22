# Contributing

Thanks for helping improve the challenge.

## Contribution Types

- Participant submission PRs (encrypted `.enc` files)
- Infrastructure improvements (workflows, scripts, docs)
- Bug fixes in validation/scoring/leaderboard tooling

## For Submission PRs

Use encrypted submissions only:

- Allowed: `submissions/*.enc`
- Not allowed: plaintext prediction files in PR

Template:
- `.github/PULL_REQUEST_TEMPLATE.md`

Guide:
- `docs/PARTICIPANT_GUIDE.md`

## For Code/Docs PRs

Before opening a PR:

```bash
python check_setup.py
python -m compileall competition scripts encryption starter_code
```

If your change touches leaderboard formatting:

```bash
python competition/render_leaderboard.py
```

## Style Expectations

- Keep explanations clear and concrete.
- Prefer small, reviewable commits.
- Update docs when behavior changes.
- Avoid introducing parallel submission paths that conflict with secure workflows.

## Security Rules

- Never commit private keys.
- Never commit plaintext private labels.
- Do not weaken `pull_request_target` safety assumptions in workflows.

See `docs/SECURITY.md` for details.
