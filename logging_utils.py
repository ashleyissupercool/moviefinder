from __future__ import annotations
 
import csv
import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "query_log.csv"

FIELDNAMES = [
    "ts",
    "query",
    "top_title",
    "top_confidence",
    "genre",
]

def _ensure_parent_dir(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

def _needs_header(log_path: Path) -> bool:
    return (not log_path.exists()) or (log_path.stat().st_size == 0)

def _safe_str(x: object, max_len: int = 2000) -> str:
    try:
        s = str(x)
    except Exception:
        s = repr(x)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def log_query(
    query: str,
    results,  # pandas DataFrame returned by search_movies
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one row to the CSV log for the given search query and results."""
    import datetime

    _ensure_parent_dir(log_path)
    write_header = _needs_header(log_path)

    top = results.iloc[0] if len(results) > 0 else None
    # support both old (Title/Genre) and new (title/overview) column names
    top_title = (
        _safe_str(top.get("Title") or top.get("title", ""))
        if top is not None else ""
    )
    top_genre = (
        _safe_str(top.get("Genre") or top.get("genre", ""))
        if top is not None else ""
    )
    row = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "query": _safe_str(query),
        "top_title": top_title,
        "top_confidence": f"{top['confidence']:.2f}" if top is not None else "",
        "genre": top_genre,
    }

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_logs(log_path: Path = DEFAULT_LOG_PATH):
    """Return all logged queries as a list of dicts, or [] if no log exists."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return []
    with open(log_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
