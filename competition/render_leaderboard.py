"""Render leaderboard.md from leaderboard.csv.

This script reads the authoritative leaderboard.csv file and generates
a static Markdown leaderboard for GitHub display.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path


def load_leaderboard(csv_path: str = "leaderboard/leaderboard.csv") -> list[dict]:
    """Load leaderboard data from CSV.
    
    Args:
        csv_path: Path to leaderboard.csv
        
    Returns:
        List of leaderboard entries sorted by score (descending)
    """
    entries = []
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                entry = {
                    "team": row.get("team", "Unknown"),
                    "model_type": row.get("model_type", ""),
                    "score": float(row.get("score", 0)),
                    "date": row.get("date", ""),
                    "notes": row.get("notes", "")
                }
                entries.append(entry)
            except (ValueError, KeyError):
                continue
    
    # Sort by score descending
    entries.sort(key=lambda x: x["score"], reverse=True)
    
    return entries


def render_markdown(entries: list[dict], output_path: str = "leaderboard.md") -> None:
    """Render leaderboard entries to Markdown file.
    
    Args:
        entries: List of leaderboard entries
        output_path: Path to output Markdown file
    """
    lines = []
    
    # Header
    lines.append("# 🏆 HeteroShot Leaderboard")
    lines.append("")
    lines.append(f"**Last Updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("")
    lines.append("📊 **[View Interactive Leaderboard](https://YOUR-ORG.github.io/YOUR-REPO/leaderboard.html)**")
    lines.append("")
    
    # Table header
    lines.append("| Rank | Team | Model Type | Macro-F1 | Date |")
    lines.append("|:----:|------|:----------:|:--------:|:----:|")
    
    # Table rows
    for rank, entry in enumerate(entries, start=1):
        rank_display = get_rank_emoji(rank)
        model_badge = get_model_badge(entry["model_type"])
        score_str = f"{entry['score']:.6f}"
        date_str = entry["date"] or "N/A"
        
        lines.append(f"| {rank_display} | {entry['team']} | {model_badge} | {score_str} | {date_str} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Statistics
    lines.append("### 📈 Statistics")
    lines.append("")
    
    if entries:
        lines.append(f"- **Total Submissions:** {len(entries)}")
        lines.append(f"- **Best Score:** {entries[0]['score']:.6f} ({entries[0]['team']})")
        
        # Count by model type
        type_counts = {}
        for entry in entries:
            mt = entry["model_type"] or "unspecified"
            type_counts[mt] = type_counts.get(mt, 0) + 1
        
        lines.append("- **By Model Type:**")
        for mt, count in sorted(type_counts.items()):
            lines.append(f"  - {mt}: {count}")
    
    lines.append("")
    lines.append("### 📊 Baseline Scores")
    lines.append("- **Random Forest (Tabular):** 0.181")
    lines.append("- **GraphSAGE (GNN):** 0.181")
    lines.append("")
    lines.append("### 🎯 Challenge Details")
    lines.append("- **Metric:** Macro-F1 (equal weight per class)")
    lines.append("- **Task:** Few-shot node classification with noisy labels")
    lines.append("- **Difficulty:** Label noise (12%) + Feature dropout (30%)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Want to compete? Check out the [README](README.md) for submission instructions!")
    lines.append("")
    lines.append("*This file is auto-generated. Do not edit manually.*")
    lines.append("*Source of truth: [leaderboard/leaderboard.csv](leaderboard/leaderboard.csv)*")
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def get_rank_emoji(rank: int) -> str:
    """Get rank display with medal emoji for top 3."""
    if rank == 1:
        return "🥇 1"
    elif rank == 2:
        return "🥈 2"
    elif rank == 3:
        return "🥉 3"
    else:
        return str(rank)


def get_model_badge(model_type: str) -> str:
    """Get display badge for model type."""
    badges = {
        "human": "👤 Human",
        "llm-only": "🤖 LLM",
        "human+llm": "🤝 Human+LLM",
        "baseline": "📊 Baseline"
    }
    return badges.get(model_type, model_type or "—")


def main():
    """Main entry point."""
    csv_path = "leaderboard/leaderboard.csv"
    output_path = "leaderboard.md"
    
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        return
    
    entries = load_leaderboard(csv_path)
    render_markdown(entries, output_path)
    
    print(f"✅ Generated {output_path} with {len(entries)} entries")


if __name__ == "__main__":
    main()
