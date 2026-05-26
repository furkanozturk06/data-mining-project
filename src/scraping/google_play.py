"""Google Play Store review scraper — balanced per star rating."""

import time
import logging
import pandas as pd
from google_play_scraper import reviews, Sort

logger = logging.getLogger(__name__)

REVIEWS_PER_STAR = 200
FETCH_MULTIPLIER = 3   # fetch more than needed to compensate for duplicates
DELAY_BETWEEN_STARS = 2  # seconds


def scrape_app(app_name: str, package_id: str, reviews_per_star: int = REVIEWS_PER_STAR) -> pd.DataFrame:
    """Scrape balanced reviews for one app from Google Play."""
    rows = []

    for star in range(1, 6):
        collected = []
        continuation_token = None
        attempts = 0

        while len(collected) < reviews_per_star and attempts < 10:
            try:
                result, continuation_token = reviews(
                    package_id,
                    lang="tr",
                    country="tr",
                    sort=Sort.NEWEST,
                    count=min(reviews_per_star * FETCH_MULTIPLIER, 500),
                    filter_score_with=star,
                    continuation_token=continuation_token,
                )
            except Exception as exc:
                logger.warning(f"[GP] {app_name} ★{star} attempt {attempts}: {exc}")
                time.sleep(5)
                attempts += 1
                continue

            if not result:
                break

            for r in result:
                text = (r.get("content") or "").strip()
                if not text:
                    continue
                collected.append({
                    "review_id": r.get("reviewId", ""),
                    "platform": "google_play",
                    "app_name": app_name,
                    "author": r.get("userName", ""),
                    "rating": r.get("score", star),
                    "title": "",   # Google Play has no title field
                    "text": text,
                    "date": r.get("at", ""),
                })

            if len(collected) >= reviews_per_star or not continuation_token:
                break

            attempts += 1
            time.sleep(1)

        rows.extend(collected[:reviews_per_star])
        logger.info(f"[GP] {app_name} ★{star}: {len(collected[:reviews_per_star])} reviews")
        time.sleep(DELAY_BETWEEN_STARS)

    return pd.DataFrame(rows)
