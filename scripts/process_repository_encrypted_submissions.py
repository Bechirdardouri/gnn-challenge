from __future__ import annotations

import argparse
import io
import json
import sys
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
    sha256_bytes,
    validate_submission_df,
)
from encryption.crypto import decrypt_payload, load_private_key, read_encrypted_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process encrypted submissions in repository and update leaderboard."
    )
    parser.add_argument(
        "--submissions-dir",
        default=str(ROOT / "submissions"),
        help="Directory to scan for .enc submissions.",
    )
    parser.add_argument(
        "--include-plaintext-csv",
        action="store_true",
        help="Also score plaintext CSV files in submissions/ (legacy/backfill mode).",
    )
    parser.add_argument(
        "--private-key-path",
        default="",
        help="Path to private key PEM. Optional when only processing plaintext CSV submissions.",
    )
    parser.add_argument(
        "--private-key-password",
        default=None,
        help="Optional private key password.",
    )
    parser.add_argument(
        "--private-labels-path",
        default=str(ROOT / "data" / "private" / "test_labels.csv"),
        help="Path to private labels CSV.",
    )
    parser.add_argument(
        "--test-nodes-path",
        default=str(ROOT / "data" / "test.csv"),
        help="Path to expected test IDs CSV.",
    )
    parser.add_argument(
        "--leaderboard-path",
        default=str(ROOT / "leaderboard" / "leaderboard.csv"),
        help="Path to leaderboard CSV.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if at least one file fails.",
    )
    return parser.parse_args()


def format_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_team_and_model(path: Path) -> tuple[str, str]:
    stem = path.name
    if stem.endswith(".enc"):
        stem = stem[: -len(".enc")]
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]
    if "__" in stem:
        team, model = stem.split("__", 1)
    else:
        team, model = stem, ""
    return team.strip() or "unknown_team", model.strip()


def safe_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def is_plaintext_submission_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    if path.name.lower() == "sample_submission.csv":
        return False
    if ".enc." in path.name.lower():
        return False
    return True


def main() -> int:
    args = parse_args()
    submissions_dir = Path(args.submissions_dir)
    if not submissions_dir.exists():
        raise FileNotFoundError(f"Submissions directory not found: {submissions_dir}")

    private_key = None
    if args.private_key_path:
        private_key_path = Path(args.private_key_path)
        if private_key_path.exists():
            private_key = load_private_key(private_key_path, args.private_key_password)
        else:
            print(f"Private key path not found, skipping .enc scoring: {private_key_path}")
    labels = read_label_csv(args.private_labels_path)
    test_nodes = pd.read_csv(args.test_nodes_path)
    leaderboard_rows = read_leaderboard(args.leaderboard_path)
    seen_submission_ids = {row.get("submission_id", "") for row in leaderboard_rows}

    enc_files = sorted(submissions_dir.glob("*.enc")) + sorted(submissions_dir.glob("**/*.enc"))
    enc_files = sorted(set(enc_files))

    csv_files: list[Path] = []
    if args.include_plaintext_csv:
        all_csv = sorted(submissions_dir.glob("*.csv")) + sorted(submissions_dir.glob("**/*.csv"))
        csv_files = sorted({p for p in all_csv if is_plaintext_submission_csv(p)})

    processed = 0
    skipped = 0
    failed = 0
    changed = False

    for path in enc_files:
        if private_key is None:
            skipped += 1
            print(f"Skipped {path}: no private key configured.")
            continue
        try:
            enc_bytes = path.read_bytes()
            submission_id = f"enc_sha256:{sha256_bytes(enc_bytes)}"
            if submission_id in seen_submission_ids:
                skipped += 1
                continue

            payload = read_encrypted_file(path)
            plaintext = decrypt_payload(payload, private_key)
            preds_raw = pd.read_csv(io.BytesIO(plaintext))
            preds = normalize_submission_columns(preds_raw)
            validate_submission_df(preds, test_nodes)
            score = score_submission_df(preds, labels, metric="auto")

            team, model = parse_team_and_model(path)
            leaderboard_rows.append(
                {
                    "timestamp_utc": format_utc_now(),
                    "team": team,
                    "model": model,
                    "score": f"{score:.8f}",
                    "source": "encrypted_repo_scan",
                    "submission_id": submission_id,
                    "pr_number": "",
                    "notes": safe_repo_relative(path),
                }
            )
            seen_submission_ids.add(submission_id)
            processed += 1
            changed = True
            print(f"Processed {path}: score={score:.8f}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"Failed {path}: {exc}")

    for path in csv_files:
        try:
            raw_csv = path.read_bytes()
            submission_id = f"csv_sha256:{sha256_bytes(raw_csv)}"
            if submission_id in seen_submission_ids:
                skipped += 1
                continue

            preds_raw = pd.read_csv(io.BytesIO(raw_csv))
            preds = normalize_submission_columns(preds_raw)
            validate_submission_df(preds, test_nodes)
            score = score_submission_df(preds, labels, metric="auto")

            team, model = parse_team_and_model(path)
            leaderboard_rows.append(
                {
                    "timestamp_utc": format_utc_now(),
                    "team": team,
                    "model": model,
                    "score": f"{score:.8f}",
                    "source": "plaintext_repo_scan",
                    "submission_id": submission_id,
                    "pr_number": "",
                    "notes": safe_repo_relative(path),
                }
            )
            seen_submission_ids.add(submission_id)
            processed += 1
            changed = True
            print(f"Processed {path}: score={score:.8f}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"Failed {path}: {exc}")

    if changed:
        write_leaderboard(args.leaderboard_path, leaderboard_rows)

    print(
        "Repository encrypted scan summary: "
        f"processed={processed}, skipped={skipped}, failed={failed}"
    )
    if args.strict and failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
