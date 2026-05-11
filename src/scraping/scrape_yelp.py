from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import re
import time

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException


BUSINESS_URLS = [
    "https://www.yelp.com/biz/the-halal-guys-new-york-3",
    "https://www.yelp.com/biz/katzs-delicatessen-new-york",
    "https://www.yelp.com/biz/shake-shack-new-york-7",
    "https://www.yelp.com/biz/russ-and-daughters-cafe-new-york",
    "https://www.yelp.com/biz/levain-bakery-new-york-3",
]

REVIEWS_PER_BUSINESS = 100
REVIEWS_PER_PAGE = 10
PAGE_LOAD_WAIT = 5
SCROLL_PAUSE = 1.2
INTER_BUSINESS_DELAY = 6.0
CHROME_MAJOR_VERSION = 147


def build_driver() -> uc.Chrome:
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    return uc.Chrome(
        options=options,
        headless=False,
        use_subprocess=True,
        version_main=CHROME_MAJOR_VERSION,
    )


def build_page_url(business_url: str, start: int) -> str:
    if start <= 0:
        return business_url
    separator = "&" if "?" in business_url else "?"
    return f"{business_url}{separator}start={start}"


def scroll_for_lazy_load(driver: uc.Chrome) -> None:
    """Scroll through the page so lazy-loaded review blocks render."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(6):
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(SCROLL_PAUSE)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def parse_rating(container) -> int | None:
    """Extract integer star rating from the review container's aria-label."""
    rating_el = container.select_one("div[aria-label*='star rating']")
    if not rating_el:
        return None
    label = rating_el.get("aria-label", "")
    match = re.search(r"(\d+(?:[.,]\d+)?)", label)
    if not match:
        return None
    try:
        return int(float(match.group(1).replace(",", ".")))
    except ValueError:
        return None


def parse_date(container) -> str | None:
    """Find a span whose text matches the Yelp review date format (e.g. 'Mar 19, 2026')."""
    pattern = re.compile(
        r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}$"
    )
    for span in container.select("span"):
        text = span.get_text(strip=True)
        if text and pattern.match(text):
            return text
    return None


def parse_reviews(html: str, business_url: str) -> list[dict[str, Any]]:
    """Extract reviews from a Yelp business page."""
    soup = BeautifulSoup(html, "lxml")
    reviews: list[dict[str, Any]] = []

    for p in soup.select("p[class*='comment__']"):
        # Walk up to the enclosing <li> review container.
        container = p
        for _ in range(8):
            container = container.parent
            if container is None or container.name == "li":
                break
        if container is None or container.name != "li":
            continue

        # Real review containers always carry a star rating widget.
        if not container.select_one("div[aria-label*='star rating']"):
            continue

        review_text = p.get_text(" ", strip=True)
        if not review_text:
            continue

        reviews.append(
            {
                "platform": "yelp",
                "review_text": review_text,
                "rating": parse_rating(container),
                "review_date": parse_date(container),
                "source_url": business_url,
            }
        )

    return reviews


def scrape_business(driver: uc.Chrome, business_url: str, target_count: int) -> list[dict[str, Any]]:
    """Paginate through a business's reviews and collect up to target_count unique entries."""
    print(f"Fetching business: {business_url}")
    collected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    pages_needed = (target_count + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE
    for page_idx in range(pages_needed + 2):
        start = page_idx * REVIEWS_PER_PAGE
        page_url = build_page_url(business_url, start)

        try:
            driver.get(page_url)
        except WebDriverException as exc:
            print(f"  Page {page_idx + 1} navigation failed: {exc}")
            continue

        time.sleep(PAGE_LOAD_WAIT)
        scroll_for_lazy_load(driver)
        time.sleep(2)

        page_reviews = parse_reviews(driver.page_source, business_url)
        new_count = 0
        for review in page_reviews:
            text = review["review_text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            collected.append(review)
            new_count += 1

        print(f"  Page {page_idx + 1} (start={start}): +{new_count} new (total {len(collected)})")

        if len(collected) >= target_count:
            break
        if new_count == 0:
            print("  No new reviews on this page, stopping early.")
            break

    return collected[:target_count]


def save_to_csv(data: list[dict[str, Any]]) -> Path:
    columns = ["platform", "review_text", "rating", "review_date", "source_url"]
    df = pd.DataFrame(data, columns=columns)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_text"]).reset_index(drop=True)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"yelp_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = output_dir / filename
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    driver = build_driver()
    all_reviews: list[dict[str, Any]] = []

    try:
        for idx, business_url in enumerate(BUSINESS_URLS):
            if idx > 0:
                time.sleep(INTER_BUSINESS_DELAY)
            try:
                business_reviews = scrape_business(driver, business_url, REVIEWS_PER_BUSINESS)
            except Exception as exc:
                print(f"Skipped '{business_url}' due to error: {exc}")
                continue
            all_reviews.extend(business_reviews)

        print(f"\nTotal reviews collected: {len(all_reviews)}")

        if not all_reviews:
            print("No reviews were collected.")
            return

        output_path = save_to_csv(all_reviews)
        print(f"Saved reviews to: {output_path}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
