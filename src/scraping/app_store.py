"""App Store review scraper — uses iTunes RSS API directly."""

import time
import logging
import hashlib
import requests
import pandas as pd

logger = logging.getLogger(__name__)

COUNTRY = "tr"
MAX_PAGES = 10          # Apple RSS limit: 10 pages × 50 reviews = 500 total
PAGE_SIZE = 50
REVIEWS_PER_STAR = 200
REQUEST_DELAY = 0.5     # seconds between page requests


def _rss_url(app_id: str, page: int) -> str:
    return (
        f"https://itunes.apple.com/{COUNTRY}/rss/customerreviews"
        f"/id={app_id}/page={page}/sortBy=mostRecent/json"
    )


def _make_review_id(app_id: str, author: str, date: str) -> str:
    raw = f"{app_id}|{author}|{date}"
    return hashlib.md5(raw.encode()).hexdigest()


def _fetch_page(app_id: str, page: int) -> list[dict]:
    url = _rss_url(app_id, page)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("feed", {}).get("entry", [])
    except Exception as exc:
        logger.warning(f"[AS] app_id={app_id} page={page}: {exc}")
        return []


def scrape_app(app_name: str, app_store_id: str, reviews_per_star: int = REVIEWS_PER_STAR) -> pd.DataFrame:
    """Fetch all available reviews then return a balanced subset per star."""
    all_rows = []

    for page in range(1, MAX_PAGES + 1):
        entries = _fetch_page(app_store_id, page)
        if not entries:
            break

        for e in entries:
            try:
                text = e["content"]["label"].strip()
                if not text:
                    continue
                author = e["author"]["name"]["label"]
                date_str = e["updated"]["label"]
                all_rows.append({
                    "review_id": _make_review_id(app_store_id, author, date_str),
                    "platform": "app_store",
                    "app_name": app_name,
                    "author": author,
                    "rating": int(e["im:rating"]["label"]),
                    "title": e["title"]["label"].strip(),
                    "text": text,
                    "date": date_str,
                })
            except (KeyError, ValueError):
                continue

        time.sleep(REQUEST_DELAY)

    if not all_rows:
        logger.warning(f"[AS] {app_name}: 0 reviews fetched")
        return pd.DataFrame()

    df_all = pd.DataFrame(all_rows)

    # Balance: up to reviews_per_star rows per star level
    balanced = []
    for star in range(1, 6):
        subset = df_all[df_all["rating"] == star]
        sampled = subset.head(reviews_per_star)
        logger.info(f"[AS] {app_name} ★{star}: {len(sampled)} reviews (pool: {len(subset)})")
        balanced.append(sampled)

    return pd.concat(balanced, ignore_index=True)
