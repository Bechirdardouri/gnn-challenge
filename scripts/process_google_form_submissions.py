from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
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
    validate_submission_df,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Google Form submissions from Google Sheets and update leaderboard."
    )
    parser.add_argument("--google-sheets-id", required=True, help="Google Sheet ID.")
    parser.add_argument(
        "--google-credentials-json",
        required=True,
        help="Service account JSON content or file path.",
    )
    parser.add_argument(
        "--sheet-range",
        default="Form Responses 1!A:Z",
        help="Sheets range containing form responses.",
    )
    parser.add_argument(
        "--team-column",
        default="Team Name",
        help="Header name for team field.",
    )
    parser.add_argument(
        "--model-column",
        default="Model Type",
        help="Header name for model field.",
    )
    parser.add_argument(
        "--file-column",
        default="Submission File",
        help="Header name for uploaded file field.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="Timestamp",
        help="Header name for form timestamp field.",
    )
    parser.add_argument(
        "--leaderboard-path",
        default=str(ROOT / "leaderboard" / "leaderboard.csv"),
        help="Leaderboard CSV path.",
    )
    parser.add_argument(
        "--private-labels-path",
        default=str(ROOT / "data" / "private" / "test_labels.csv"),
        help="Path to hidden labels CSV.",
    )
    parser.add_argument(
        "--test-nodes-path",
        default=str(ROOT / "data" / "public" / "test_nodes.csv"),
        help="Path to test nodes CSV.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code when any submission fails.",
    )
    return parser.parse_args()


def load_credentials_info(raw: str) -> dict:
    try:
        path = Path(raw)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        pass
    return json.loads(raw)


def build_google_clients(credentials_raw: str):
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google API dependencies missing. Install requirements.txt before running this script."
        ) from exc

    info = load_credentials_info(credentials_raw)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    credentials = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    sheets = build("sheets", "v4", credentials=credentials, cache_discovery=False)
    drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return sheets, drive


def sheet_rows(sheets_service, spreadsheet_id: str, cell_range: str) -> list[list[str]]:
    response = (
        sheets_service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=cell_range)
        .execute()
    )
    return response.get("values", [])


def value_at(row: list[str], idx: int) -> str:
    if idx < 0:
        return ""
    if idx >= len(row):
        return ""
    return str(row[idx]).strip()


def extract_drive_file_ids(raw: str) -> list[str]:
    if not raw:
        return []
    candidates = set()
    for part in re.split(r"[,\s]+", raw):
        token = part.strip()
        if not token:
            continue
        match = re.search(r"/d/([A-Za-z0-9_-]{20,})", token)
        if match:
            candidates.add(match.group(1))
            continue
        match = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", token)
        if match:
            candidates.add(match.group(1))
            continue
        if re.fullmatch(r"[A-Za-z0-9_-]{20,}", token):
            candidates.add(token)
    return sorted(candidates)


def download_drive_file_bytes(drive_service, file_id: str) -> bytes:
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise RuntimeError(
            "Google API dependencies missing. Install requirements.txt before running this script."
        ) from exc

    request = drive_service.files().get_media(fileId=file_id)
    output = io.BytesIO()
    downloader = MediaIoBaseDownload(output, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return output.getvalue()


def parse_timestamp(raw: str) -> str:
    if not raw:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = raw.strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_render_script() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "competition" / "render_leaderboard.py")],
        check=True,
    )


def main() -> int:
    args = parse_args()
    labels = read_label_csv(args.private_labels_path)
    test_nodes = pd.read_csv(args.test_nodes_path)
    leaderboard = read_leaderboard(args.leaderboard_path)
    scored_submission_ids = {r.get("submission_id", "") for r in leaderboard}

    sheets, drive = build_google_clients(args.google_credentials_json)
    rows = sheet_rows(sheets, args.google_sheets_id, args.sheet_range)
    if not rows:
        print("No rows found in the Google Sheet.")
        return 0

    header = rows[0]
    idx_team = header.index(args.team_column) if args.team_column in header else -1
    idx_model = header.index(args.model_column) if args.model_column in header else -1
    idx_file = header.index(args.file_column) if args.file_column in header else -1
    idx_ts = header.index(args.timestamp_column) if args.timestamp_column in header else -1

    if idx_team == -1 or idx_file == -1:
        raise ValueError(
            f"Required columns missing in sheet header. Found: {header}. "
            f"Expected at least '{args.team_column}' and '{args.file_column}'."
        )

    processed = 0
    skipped = 0
    failed = 0
    changed = False

    for row_number, row in enumerate(rows[1:], start=2):
        team = value_at(row, idx_team) or "unknown_team"
        model = value_at(row, idx_model)
        timestamp = parse_timestamp(value_at(row, idx_ts))
        raw_file_field = value_at(row, idx_file)
        file_ids = extract_drive_file_ids(raw_file_field)
        if not file_ids:
            failed += 1
            print(f"Row {row_number}: no Drive file ID found in '{raw_file_field}'.")
            continue

        file_id = file_ids[0]
        submission_id = f"google_drive:{file_id}"
        if submission_id in scored_submission_ids:
            skipped += 1
            continue

        try:
            csv_bytes = download_drive_file_bytes(drive, file_id)
            preds_raw = pd.read_csv(io.BytesIO(csv_bytes))
            preds = normalize_submission_columns(preds_raw)
            validate_submission_df(preds, test_nodes)
            score = score_submission_df(preds, labels)

            leaderboard.append(
                {
                    "timestamp_utc": timestamp,
                    "team": team,
                    "model": model,
                    "score": f"{score:.8f}",
                    "source": "google_form",
                    "submission_id": submission_id,
                    "pr_number": "",
                    "notes": f"sheet_row={row_number}",
                }
            )
            scored_submission_ids.add(submission_id)
            processed += 1
            changed = True
        except Exception as exc:  # noqa: BLE001 - preserve detailed failure in CI logs
            failed += 1
            print(f"Row {row_number} failed: {exc}")

    if changed:
        write_leaderboard(args.leaderboard_path, leaderboard)
        run_render_script()

    print(
        "Google Form processing summary: "
        f"processed={processed}, skipped={skipped}, failed={failed}"
    )
    if args.strict and failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
