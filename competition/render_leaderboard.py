from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

try:
    from .leaderboard_utils import read_leaderboard
except ImportError:  # pragma: no cover - direct script execution path
    from leaderboard_utils import read_leaderboard

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "leaderboard" / "leaderboard.csv"
MD_PATH = ROOT / "leaderboard" / "leaderboard.md"
JSON_PATH = ROOT / "docs" / "leaderboard.json"


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def score_key(row: dict[str, str]) -> float:
        try:
            return float(row.get("score", "-inf"))
        except Exception:
            return float("-inf")

    def ts_key(row: dict[str, str]) -> datetime:
        try:
            return datetime.fromisoformat(row.get("timestamp_utc", "").replace("Z", "+00:00"))
        except Exception:
            return datetime.fromtimestamp(0)

    return sorted(rows, key=lambda r: (score_key(r), ts_key(r)), reverse=True)


def write_markdown(rows: list[dict[str, str]]) -> None:
    lines: list[str] = []
    lines.append("# Leaderboard\n")
    lines.append("This leaderboard is **auto-updated** from secure evaluation pipelines.\n\n")
    lines.append("| Rank | Team | Model | Score | Source | Date (UTC) | Notes |\n")
    lines.append("|---:|---|---|---:|---|---|---|\n")
    for i, row in enumerate(rows, start=1):
        team = row.get("team", "")
        model = row.get("model", "")
        score = row.get("score", "")
        source = row.get("source", "")
        ts = row.get("timestamp_utc", "")
        notes = row.get("notes", "")
        model_disp = f"`{model}`" if model else ""
        lines.append(f"| {i} | {team} | {model_disp} | {score} | {source} | {ts} | {notes} |\n")
    MD_PATH.write_text("".join(lines), encoding="utf-8")


def write_json(rows: list[dict[str, str]]) -> None:
    payload = []
    for i, row in enumerate(rows, start=1):
        payload.append({"rank": i, **row})
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    rows = read_leaderboard(CSV_PATH)
    sorted_rows = sort_rows(rows)
    write_markdown(sorted_rows)
    write_json(sorted_rows)


if __name__ == "__main__":
    main()
