"""Review cleaning and preprocessing logic.

Reads raw CSVs from data/raw/, applies cleaning steps (encoding fix,
deduplication, null handling, normalization, type coercion), and
writes cleaned outputs to data/processed/.

Run from project root:
    python -m src.preprocessing.clean_reviews
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_FILES = {
    "app_store": RAW_DIR / "app_store_reviews.csv",
    "google_play": RAW_DIR / "google_play_reviews.csv",
}

TEXT_COLS = ["author", "title", "text", "app_name"]
WS_RE = re.compile(r"\s+")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    s = unicodedata.normalize("NFC", str(value))
    s = s.replace("​", "").replace("﻿", "")
    s = WS_RE.sub(" ", s).strip()
    return s


def _load_raw(path: Path) -> pd.DataFrame:
    # utf-8-sig handles the BOM at the start of both files.
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False, na_values=[""])


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"input_rows": len(df)}

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].map(_normalize_text)

    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    stats["dropped_empty_text"] = before - len(df)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    before = len(df)
    df = df[df["rating"].between(1, 5)].copy()
    df["rating"] = df["rating"].astype("int8")
    stats["dropped_invalid_rating"] = before - len(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed")
    before = len(df)
    df = df[df["date"].notna()].copy()
    stats["dropped_invalid_date"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    stats["dropped_duplicate_id"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["platform", "author", "text"], keep="first")
    stats["dropped_duplicate_content"] = before - len(df)

    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    stats["output_rows"] = len(df)
    return df, stats


def _print_report(name: str, stats: dict) -> None:
    print(f"\n[{name}]")
    for k, v in stats.items():
        print(f"  {k:.<28} {v:>7}")


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined_frames = []

    for name, path in RAW_FILES.items():
        if not path.exists():
            print(f"WARN: {path} not found, skipping", file=sys.stderr)
            continue
        raw = _load_raw(path)
        cleaned, stats = clean(raw)
        out_path = PROCESSED_DIR / f"{name}_reviews_clean.csv"
        cleaned.to_csv(out_path, index=False, encoding="utf-8")
        _print_report(name, stats)
        print(f"  wrote -> {out_path}")
        combined_frames.append(cleaned)

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(subset=["review_id"], keep="first")
        cross_dedup = before - len(combined)
        combined = combined.sort_values("date", ascending=False).reset_index(drop=True)
        combined_path = PROCESSED_DIR / "all_reviews_clean.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8")
        print(f"\n[combined]")
        print(f"  cross_platform_dup_ids....... {cross_dedup:>7}")
        print(f"  total_rows................... {len(combined):>7}")
        print(f"  wrote -> {combined_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
