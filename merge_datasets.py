"""
merge_datasets.py — Merge multiple movie CSVs into one clean dataset.

Usage:
    python merge_datasets.py file1.csv file2.csv file3.csv ...

Each input CSV must have at minimum a title column and an overview/description column.
The script will ask you to map column names if they differ from the defaults.

Output: data/TMDB_movies.csv  (overwrites existing file)
Then run: python reindex.py   (to rebuild the vector index)
"""

import sys
import pandas as pd
from pathlib import Path

OUTPUT = Path("data/TMDB_movies.csv")

# Common column name variations found in popular datasets
TITLE_ALIASES    = ["title", "movie_title", "Title", "Movie Title", "name", "original_title"]
OVERVIEW_ALIASES = ["overview", "description", "plot", "synopsis", "Description",
                    "Plot", "Overview", "summary", "long_description"]


def find_col(df: pd.DataFrame, aliases: list[str], label: str) -> str:
    """Return the first matching column name, or prompt the user to pick one."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    print(f"\nCould not auto-detect the {label} column.")
    print("Available columns:", list(df.columns))
    choice = input(f"Enter the column name to use as {label}: ").strip()
    if choice not in df.columns:
        raise ValueError(f"Column '{choice}' not found in dataset.")
    return choice


def load_csv(path: str) -> pd.DataFrame:
    print(f"\nLoading: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    title_col    = find_col(df, TITLE_ALIASES, "title")
    overview_col = find_col(df, OVERVIEW_ALIASES, "overview")

    out = pd.DataFrame({
        "title":    df[title_col].astype(str).str.strip(),
        "overview": df[overview_col].astype(str).str.strip(),
    })
    # Drop rows with no useful content
    out = out[out["title"].str.len() > 0]
    out = out[out["overview"].str.len() > 10]
    out = out[~out["overview"].isin(["nan", "None", "N/A", ""])]
    print(f"  → {len(out)} usable rows after filtering.")
    return out


def main():
    files = sys.argv[1:]
    if not files:
        files = sorted(str(p) for p in Path("data").glob("*.csv"))
        if not files:
            print("No CSV files found in data/ folder.")
            sys.exit(1)
        print(f"Auto-detected {len(files)} CSV files: {[Path(f).name for f in files]}")

    frames = [load_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)
    # Keep the entry with the longest overview for duplicate titles
    combined["_len"] = combined["overview"].str.len()
    combined = (
        combined.sort_values("_len", ascending=False)
        .drop_duplicates(subset=["title"], keep="first")
        .drop(columns=["_len"])
        .reset_index(drop=True)
    )
    after = len(combined)

    print(f"\n✅ Merged {before} rows → {after} unique movies after deduplication.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)
    print(f"✅ Saved to {OUTPUT}")
    print("\nNext step: run  python reindex.py  to rebuild the vector index.")


if __name__ == "__main__":
    main()
