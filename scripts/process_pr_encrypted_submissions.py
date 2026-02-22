from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition.leaderboard_utils import read_leaderboard, write_leaderboard
from competition.submission_utils import (
    normalize_submission_columns,
    read_label_csv,
    score_submission_df,
    validate_submission_df,
)
from encryption.crypto import decrypt_payload, load_private_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process encrypted .enc files from a pull request and update leaderboard."
    )
    parser.add_argument("--repo", required=True, help="GitHub repository in owner/name format.")
    parser.add_argument("--pr-number", required=True, type=int, help="Pull request number.")
    parser.add_argument("--github-token", required=True, help="GitHub token.")
    parser.add_argument("--private-key-path", required=True, help="Path to private key PEM file.")
    parser.add_argument(
        "--private-key-password",
        default=None,
        help="Optional password for encrypted private key.",
    )
    parser.add_argument(
        "--private-labels-path",
        default=str(ROOT / "data" / "private" / "test_labels.csv"),
        help="Path to hidden labels CSV.",
    )
    parser.add_argument(
        "--test-nodes-path",
        default=str(ROOT / "data" / "test.csv"),
        help="Path to public test nodes CSV.",
    )
    parser.add_argument(
        "--leaderboard-path",
        default=str(ROOT / "leaderboard" / "leaderboard.csv"),
        help="Leaderboard CSV path.",
    )
    parser.add_argument(
        "--comment-path",
        default=str(ROOT / "artifacts" / "pr_comment.md"),
        help="Where to write markdown summary for PR comment.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code when any submission fails validation/scoring.",
    )
    return parser.parse_args()


def gh_get_json(url: str, token: str) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "gnn-competition-bot",
        },
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def list_pr_files(repo: str, pr_number: int, token: str) -> list[dict]:
    files: list[dict] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
            f"?per_page=100&page={page}"
        )
        page_data = gh_get_json(url, token)
        if not page_data:
            break
        files.extend(page_data)
        page += 1
    return files


def get_pr_head_sha(repo: str, pr_number: int, token: str) -> str:
    pr = gh_get_json(f"https://api.github.com/repos/{repo}/pulls/{pr_number}", token)
    return str(pr["head"]["sha"])


def download_file_content(repo: str, path: str, ref: str, token: str) -> bytes:
    quoted_path = urllib.parse.quote(path)
    url = f"https://api.github.com/repos/{repo}/contents/{quoted_path}?ref={urllib.parse.quote(ref)}"
    data = gh_get_json(url, token)
    encoded = (data.get("content") or "").replace("\n", "")
    if not encoded:
        raise ValueError(f"Unable to read file content for {path}")
    return base64.b64decode(encoded)


def parse_team_and_model(file_path: str) -> tuple[str, str]:
    stem = Path(file_path).name
    if stem.endswith(".enc"):
        stem = stem[: -len(".enc")]
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    if "__" in stem:
        team, model = stem.split("__", 1)
    else:
        team, model = stem, ""
    return team.strip(), model.strip()


def format_ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def leaderboard_has_submission(rows: list[dict[str, str]], submission_id: str) -> bool:
    return any(r.get("submission_id") == submission_id for r in rows)


def render_comment(
    pr_number: int,
    processed: list[dict[str, str]],
    skipped: list[dict[str, str]],
    failed: list[dict[str, str]],
) -> str:
    lines: list[str] = []
    lines.append("## Secure Submission Evaluation\n")
    lines.append(f"- PR: `#{pr_number}`\n")
    lines.append(f"- Processed: `{len(processed)}`\n")
    lines.append(f"- Skipped (already scored): `{len(skipped)}`\n")
    lines.append(f"- Failed: `{len(failed)}`\n\n")

    if processed:
        lines.append("| Team | Model | Score | Submission ID | File |\n")
        lines.append("|---|---|---:|---|---|\n")
        for row in processed:
            lines.append(
                "| {team} | {model} | {score} | `{submission_id}` | `{file_path}` |\n".format(
                    **row
                )
            )
        lines.append("\n")

    if skipped:
        lines.append("### Skipped\n")
        for row in skipped:
            lines.append(f"- `{row['file_path']}` (`{row['submission_id']}` already scored)\n")
        lines.append("\n")

    if failed:
        lines.append("### Failed\n")
        for row in failed:
            lines.append(f"- `{row['file_path']}`: {row['error']}\n")
        lines.append("\n")

    lines.append(
        "Predictions remain private: only encrypted `.enc` files are stored in the public repository."
    )
    lines.append("\n")
    return "".join(lines)


def main() -> int:
    args = parse_args()
    Path(args.comment_path).parent.mkdir(parents=True, exist_ok=True)

    private_key = load_private_key(args.private_key_path, args.private_key_password)
    labels = read_label_csv(args.private_labels_path)
    test_nodes = pd.read_csv(args.test_nodes_path)
    leaderboard_rows = read_leaderboard(args.leaderboard_path)

    head_sha = get_pr_head_sha(args.repo, args.pr_number, args.github_token)
    files = list_pr_files(args.repo, args.pr_number, args.github_token)
    enc_files = [
        f
        for f in files
        if str(f.get("filename", "")).startswith("submissions/")
        and str(f.get("filename", "")).endswith(".enc")
        and f.get("status") in {"added", "modified", "renamed"}
    ]

    processed: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    changed = False

    for entry in enc_files:
        file_path = str(entry["filename"])
        submission_id = str(entry.get("sha", "")).strip()
        if not submission_id:
            submission_id = f"sha_missing:{file_path}"

        if leaderboard_has_submission(leaderboard_rows, submission_id):
            skipped.append({"file_path": file_path, "submission_id": submission_id})
            continue

        try:
            raw_enc_file = download_file_content(args.repo, file_path, head_sha, args.github_token)
            payload = json.loads(raw_enc_file.decode("utf-8"))
            plaintext_csv = decrypt_payload(payload, private_key)
            preds_raw = pd.read_csv(io.BytesIO(plaintext_csv))
            preds = normalize_submission_columns(preds_raw)
            validate_submission_df(preds, test_nodes)
            score = score_submission_df(preds, labels)

            team, model = parse_team_and_model(file_path)
            row = {
                "timestamp_utc": format_ts_now(),
                "team": team or "unknown_team",
                "model": model,
                "score": f"{score:.8f}",
                "source": "encrypted_pr",
                "submission_id": submission_id,
                "pr_number": str(args.pr_number),
                "notes": f"{file_path}",
            }
            leaderboard_rows.append(row)
            processed.append({**row, "file_path": file_path})
            changed = True
        except Exception as exc:  # noqa: BLE001 - return exact message in PR comment
            failed.append({"file_path": file_path, "error": str(exc)})

    if changed:
        write_leaderboard(args.leaderboard_path, leaderboard_rows)

    comment = render_comment(args.pr_number, processed, skipped, failed)
    Path(args.comment_path).write_text(comment, encoding="utf-8")

    if not enc_files:
        print("No encrypted submissions were found in this PR.")
    else:
        print(comment)

    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
