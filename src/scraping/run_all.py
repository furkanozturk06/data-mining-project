"""
Main scraper runner.

Usage:
    python -m src.scraping.run_all                  # both platforms, all apps
    python -m src.scraping.run_all --platform gp    # Google Play only
    python -m src.scraping.run_all --platform as    # App Store only
    python -m src.scraping.run_all --app "Akbank"   # single app (both platforms)
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.scraping.app_config import APPS
from src.scraping import google_play, app_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GP_OUTPUT = OUTPUT_DIR / "google_play_reviews.csv"
AS_OUTPUT = OUTPUT_DIR / "app_store_reviews.csv"

DELAY_BETWEEN_APPS = 4  # seconds between apps to avoid rate limiting


def load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def save(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    existing = load_existing(path)
    combined = pd.concat([existing, df], ignore_index=True)
    combined.drop_duplicates(subset=["review_id", "platform"], inplace=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(combined)} rows → {path}")


def run(platform: str, app_filter: Optional[str]) -> None:
    apps = APPS
    if app_filter:
        apps = [a for a in apps if app_filter.lower() in a["app_name"].lower()]
        if not apps:
            logger.error(f"No app found matching '{app_filter}'")
            return

    gp_frames, as_frames = [], []

    for i, app in enumerate(apps, 1):
        name = app["app_name"]
        logger.info(f"── [{i}/{len(apps)}] {name} ──────────────────────────")

        if platform in ("gp", "both") and app.get("google_play_id"):
            try:
                df = google_play.scrape_app(name, app["google_play_id"])
                if not df.empty:
                    gp_frames.append(df)
                    save(df, GP_OUTPUT)
            except Exception as exc:
                logger.error(f"[GP] {name} failed: {exc}")

        if platform in ("as", "both") and app.get("app_store_id"):
            try:
                df = app_store.scrape_app(name, app["app_store_id"])
                if not df.empty:
                    as_frames.append(df)
                    save(df, AS_OUTPUT)
            except Exception as exc:
                logger.error(f"[AS] {name} failed: {exc}")

        if i < len(apps):
            time.sleep(DELAY_BETWEEN_APPS)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    if gp_frames:
        total = pd.concat(gp_frames, ignore_index=True)
        logger.info(f"Google Play total rows this run: {len(total)}")
        _print_balance(total, "Google Play")

    if as_frames:
        total = pd.concat(as_frames, ignore_index=True)
        logger.info(f"App Store total rows this run: {len(total)}")
        _print_balance(total, "App Store")


def _print_balance(df: pd.DataFrame, label: str) -> None:
    dist = df.groupby("rating").size().reindex(range(1, 6), fill_value=0)
    logger.info(f"{label} star distribution:\n{dist.to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        choices=["gp", "as", "both"],
        default="both",
        help="gp=Google Play, as=App Store, both=her ikisi",
    )
    parser.add_argument("--app", default=None, help="Tek bir uygulama adı (kısmi eşleşme)")
    args = parser.parse_args()

    run(platform=args.platform, app_filter=args.app)
