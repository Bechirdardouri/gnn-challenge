from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


LEADERBOARD_COLUMNS = [
    "timestamp_utc",
    "team",
    "model",
    "score",
    "source",
    "submission_id",
    "pr_number",
    "notes",
]


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    out = {col: (row.get(col) or "").strip() for col in LEADERBOARD_COLUMNS}
    if not out["source"]:
        out["source"] = "manual"
    return out


def read_leaderboard(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [normalize_row(r) for r in reader if (r.get("team") or "").strip()]


def write_leaderboard(path: str | Path, rows: Iterable[dict[str, str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = [normalize_row(r) for r in rows]
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized_rows)
