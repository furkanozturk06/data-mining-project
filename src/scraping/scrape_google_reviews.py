from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
import re
import time

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


PLACE_QUERIES = [
    "Times Square, New York",
    "Joe's Pizza Greenwich Village, New York",
    "Eiffel Tower, Paris",
    "Apple Fifth Avenue, New York",
    "Empire State Building, New York",
    "Statue of Liberty",
]

REVIEWS_PER_PLACE = 100
SCROLL_PAUSE = 2.0
MAX_SCROLL_ROUNDS = 120
INTER_PLACE_DELAY = 6.0
CHROME_MAJOR_VERSION = 147


def build_driver() -> uc.Chrome:
    """Start undetected Chrome. Plain selenium is blocked by Google's limited-view filter."""
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    return uc.Chrome(
        options=options,
        headless=False,
        use_subprocess=True,
        version_main=CHROME_MAJOR_VERSION,
    )


def build_search_url(query: str) -> str:
    return f"https://www.google.com/maps/search/{quote_plus(query)}/?hl=en"


def force_reviews_in_url(place_url: str) -> str:
    """Rebuild the place URL with the minimal data block that triggers the reviews panel."""
    name_match = re.search(r"/place/([^/]+)/", place_url)
    coord_match = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+),(\d+z)", place_url)
    feature_match = re.search(r"!1s([^!?]+)", place_url)
    lat_match = re.search(r"!3d(-?\d+\.\d+)", place_url)
    lng_match = re.search(r"!4d(-?\d+\.\d+)", place_url)

    if not all([name_match, coord_match, feature_match, lat_match, lng_match]):
        return place_url

    return (
        f"https://www.google.com/maps/place/{name_match.group(1)}/"
        f"@{coord_match.group(1)},{coord_match.group(2)},{coord_match.group(3)}/"
        f"data=!4m7!3m6"
        f"!1s{feature_match.group(1)}"
        f"!8m2!3d{lat_match.group(1)}!4d{lng_match.group(1)}"
        f"!9m1!1b1?hl=en"
    )


def navigate_to_place(driver: uc.Chrome, query: str) -> str | None:
    """Open Google Maps search and return the redirected place URL."""
    driver.get(build_search_url(query))

    # Wait up to 12 seconds for an automatic redirect to /place/.
    for _ in range(12):
        time.sleep(1)
        if "/place/" in driver.current_url:
            return driver.current_url

    # Search returned a result list. Click the first place card.
    for selector in ("a.hfpxzc", "div[role='feed'] a[href*='/place/']"):
        try:
            first = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            first.click()
            time.sleep(4)
            if "/place/" in driver.current_url:
                return driver.current_url
        except TimeoutException:
            continue

    return None


def find_scroll_panel(driver: uc.Chrome):
    """Locate the scrollable reviews panel."""
    selectors = [
        "div[role='feed']",
        "div.m6QErb.DxyBCb.kA9KIf.dS8AEf",
        "div.m6QErb.DxyBCb",
    ]
    for selector in selectors:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            if element.size.get("height", 0) > 200:
                return element
    return None


def count_unique_reviews(driver: uc.Chrome) -> int:
    """Count unique review cards by data-review-id (DOM may render duplicates)."""
    return driver.execute_script(
        "const s = new Set();"
        "document.querySelectorAll('div[data-review-id]').forEach("
        "el => s.add(el.getAttribute('data-review-id')));"
        "return s.size;"
    )


def scroll_until_loaded(driver: uc.Chrome, target_count: int) -> int:
    """Scroll the reviews panel until enough unique cards load or progress stops."""
    panel = find_scroll_panel(driver)
    if panel is None:
        return 0

    last_count = 0
    stable_rounds = 0

    for _ in range(MAX_SCROLL_ROUNDS):
        try:
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight;", panel
            )
        except WebDriverException:
            break

        time.sleep(SCROLL_PAUSE)

        current = count_unique_reviews(driver)
        if current >= target_count:
            return current

        if current == last_count:
            stable_rounds += 1
            if stable_rounds >= 6:
                return current
        else:
            stable_rounds = 0
            last_count = current

    return last_count


def expand_more_buttons(driver: uc.Chrome) -> None:
    """Click 'More' on truncated reviews so the full text shows up in the DOM."""
    buttons = driver.find_elements(By.CSS_SELECTOR, "button.w8nwRe")
    for btn in buttons:
        try:
            driver.execute_script("arguments[0].click();", btn)
        except WebDriverException:
            continue
    if buttons:
        time.sleep(1)


def parse_rating_from_card(card) -> int | None:
    """Extract integer star rating from a review card's aria-label."""
    rating_el = card.select_one("span[role='img'][aria-label*='star']")
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


def parse_reviews(html: str, place_url: str) -> list[dict[str, Any]]:
    """Parse review cards from the reviews panel and dedupe by data-review-id."""
    soup = BeautifulSoup(html, "lxml")
    reviews: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for card in soup.select("div[data-review-id]"):
        review_id = card.get("data-review-id", "")
        if not review_id or review_id in seen_ids:
            continue

        text_el = card.select_one("span.wiI7pd")
        review_text = text_el.get_text(" ", strip=True) if text_el else ""
        if not review_text:
            continue

        rating = parse_rating_from_card(card)

        date_el = card.select_one("span.rsqaWe")
        review_date = date_el.get_text(strip=True) if date_el else None

        seen_ids.add(review_id)
        reviews.append(
            {
                "platform": "google_reviews",
                "review_text": review_text,
                "rating": rating,
                "review_date": review_date,
                "source_url": place_url,
            }
        )

    return reviews


def scrape_place(
    driver: uc.Chrome,
    query: str,
    target_count: int,
) -> list[dict[str, Any]]:
    """Navigate to a place and collect reviews up to the target count."""
    print(f"Fetching place: {query}")

    place_url = navigate_to_place(driver, query)
    if place_url is None:
        print(f"  Could not resolve place URL for: {query}")
        return []

    reviews_url = force_reviews_in_url(place_url)
    driver.get(reviews_url)
    time.sleep(6)

    loaded = scroll_until_loaded(driver, target_count)
    expand_more_buttons(driver)
    time.sleep(1)

    reviews = parse_reviews(driver.page_source, reviews_url)
    print(f"  Loaded {loaded} cards, parsed {len(reviews)} reviews for: {query}")
    return reviews[:target_count]


def save_to_csv(data: list[dict[str, Any]]) -> Path:
    """Save reviews under data/raw/ as a timestamped CSV."""
    columns = ["platform", "review_text", "rating", "review_date", "source_url"]
    df = pd.DataFrame(data, columns=columns)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_text", "review_date"]).reset_index(drop=True)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"google_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = output_dir / filename
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    driver = build_driver()
    all_reviews: list[dict[str, Any]] = []

    try:
        for idx, query in enumerate(PLACE_QUERIES):
            if idx > 0:
                time.sleep(INTER_PLACE_DELAY)
            try:
                place_reviews = scrape_place(driver, query, REVIEWS_PER_PLACE)
            except Exception as exc:
                print(f"Skipped '{query}' due to error: {exc}")
                continue
            all_reviews.extend(place_reviews)

        print(f"Total reviews collected: {len(all_reviews)}")

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
