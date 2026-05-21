from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


PLACE_QUERIES = [
    "McDonald's Times Square New York",
    "Starbucks Fifth Avenue New York",
    "Apple Store Fifth Avenue New York",
    "Whole Foods Columbus Circle New York",
    "Best Buy Herald Square New York",
    "Target Tribeca New York",
    "Macy's Herald Square New York",
    "The Plaza Hotel New York",
    "Marriott Marquis Times Square New York",
    "CVS Pharmacy Midtown New York",
    "Trader Joe's Upper West Side New York",
    "Home Depot Brooklyn New York",
    "Costco Brooklyn New York",
    "IKEA Brooklyn New York",
    "Nike Store Fifth Avenue New York",
]

REVIEWS_PER_PLACE = 80
SCROLL_PAUSE = 2.0
MAX_SCROLL_ROUNDS = 120
INTER_PLACE_DELAY = 6.0
CHROME_MAJOR_VERSION = 148


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


def build_search_url(query: str) -> str:
    return f"https://www.google.com/maps/search/{quote_plus(query)}/?hl=en"


def force_reviews_in_url(place_url: str) -> str:
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
    driver.get(build_search_url(query))
    for _ in range(12):
        time.sleep(1)
        if "/place/" in driver.current_url:
            return driver.current_url

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


def extract_place_info(driver: uc.Chrome) -> tuple[str, str, str]:
    """Extract place name, overall rating, and total review count from Google Maps."""
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")

    # Place name
    company_name = "N/A"
    for sel in ["h1.DUwDvf", "h1[class*='fontHeadlineLarge']", "h1"]:
        el = soup.select_one(sel)
        if el:
            company_name = el.get_text(strip=True)
            break

    # Overall rating
    overall_rating = "N/A"
    for sel in ["div.F7nice span", "span.ceNzKf", "div[aria-label*='stars']"]:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(strip=True)
            m = re.search(r"(\d+[.,]\d+|\d+)", text)
            if m:
                overall_rating = m.group(1).replace(",", ".")
                break

    # Total reviews
    total_reviews = "N/A"
    for sel in ["span[aria-label*='review']", "div.F7nice span:nth-child(2)", "button[jsaction*='review']"]:
        el = soup.select_one(sel)
        if el:
            text = el.get("aria-label", "") or el.get_text(strip=True)
            m = re.search(r"([\d,]+)\s*review", text, re.IGNORECASE)
            if m:
                total_reviews = m.group(1).replace(",", "")
                break

    return company_name, overall_rating, total_reviews


def find_scroll_panel(driver: uc.Chrome):
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
    return driver.execute_script(
        "const s = new Set();"
        "document.querySelectorAll('div[data-review-id]').forEach("
        "el => s.add(el.getAttribute('data-review-id')));"
        "return s.size;"
    )


def scroll_until_loaded(driver: uc.Chrome, target_count: int) -> int:
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
    buttons = driver.find_elements(By.CSS_SELECTOR, "button.w8nwRe")
    for btn in buttons:
        try:
            driver.execute_script("arguments[0].click();", btn)
        except WebDriverException:
            continue
    if buttons:
        time.sleep(1)


def parse_reviews(
    html: str,
    company_name: str,
    overall_rating: str,
    total_reviews: str,
) -> list[dict[str, Any]]:
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

        # Rating
        review_rating = "N/A"
        rating_el = card.select_one("span[role='img'][aria-label*='star']")
        if rating_el:
            m = re.search(r"(\d+)", rating_el.get("aria-label", ""))
            if m:
                review_rating = m.group(1)

        # Date
        date_el = card.select_one("span.rsqaWe")
        review_date = date_el.get_text(strip=True) if date_el else "N/A"

        # Reviewer name
        reviewer_name = "N/A"
        for sel in ["div.d4r55", "span.G3zWCf", "a[href*='contrib']"]:
            name_el = card.select_one(sel)
            if name_el:
                name = name_el.get_text(strip=True)
                if name:
                    reviewer_name = name
                    break

        seen_ids.add(review_id)
        reviews.append({
            "company_name": company_name,
            "overall_rating": overall_rating,
            "total_reviews": total_reviews,
            "review_title": "N/A",
            "review_text": review_text,
            "review_rating": review_rating,
            "review_date": review_date,
            "reviewer_name": reviewer_name,
        })

    return reviews


def scrape_place(driver: uc.Chrome, query: str, target_count: int) -> list[dict[str, Any]]:
    print(f"  Fetching: {query}")

    place_url = navigate_to_place(driver, query)
    if place_url is None:
        print(f"    Could not resolve place URL.")
        return []

    reviews_url = force_reviews_in_url(place_url)
    driver.get(reviews_url)
    time.sleep(6)

    company_name, overall_rating, total_reviews = extract_place_info(driver)
    print(f"    {company_name} | {overall_rating}★ | {total_reviews} reviews")

    loaded = scroll_until_loaded(driver, target_count)
    expand_more_buttons(driver)
    time.sleep(1)

    reviews = parse_reviews(driver.page_source, company_name, overall_rating, total_reviews)
    print(f"    Loaded {loaded} cards, parsed {len(reviews)}")
    return reviews[:target_count]


def save_to_csv(data: list[dict[str, Any]]) -> Path:
    columns = [
        "company_name", "overall_rating", "total_reviews",
        "review_title", "review_text", "review_rating",
        "review_date", "reviewer_name",
    ]
    df = pd.DataFrame(data, columns=columns)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_text"]).reset_index(drop=True)

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
                reviews = scrape_place(driver, query, REVIEWS_PER_PLACE)
            except Exception as exc:
                print(f"  Skipped '{query}': {exc}")
                continue
            all_reviews.extend(reviews)
            print(f"  Grand total: {len(all_reviews)}")

        print(f"\nTotal reviews: {len(all_reviews)}")
        if not all_reviews:
            return

        output_path = save_to_csv(all_reviews)
        print(f"Saved to: {output_path}")

        df = pd.read_csv(output_path)
        print("\nRating distribution:")
        print(df["review_rating"].value_counts().sort_index())
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
