<<<<<<< Updated upstream
"""Yelp scraping module."""
=======
from __future__ import annotations

import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException


BUSINESS_URLS = [
    # NYC Restoranlar
    "https://www.yelp.com/biz/the-halal-guys-new-york-3",
    "https://www.yelp.com/biz/katzs-delicatessen-new-york",
    "https://www.yelp.com/biz/shake-shack-new-york-7",
    "https://www.yelp.com/biz/russ-and-daughters-cafe-new-york",
    "https://www.yelp.com/biz/levain-bakery-new-york-3",
    # LA Restoranlar
    "https://www.yelp.com/biz/guelaguetza-los-angeles-4",
    "https://www.yelp.com/biz/republique-los-angeles",
    "https://www.yelp.com/biz/in-n-out-burger-los-angeles-12",
    # Chicago Restoranlar
    "https://www.yelp.com/biz/lou-malnatis-pizzeria-chicago-3",
    "https://www.yelp.com/biz/au-cheval-chicago",
    # Oteller
    "https://www.yelp.com/biz/the-plaza-hotel-new-york",
    "https://www.yelp.com/biz/waldorf-astoria-new-york-new-york",
    # Perakende / Hizmet
    "https://www.yelp.com/biz/apple-store-fifth-avenue-new-york",
    "https://www.yelp.com/biz/best-buy-new-york-3",
    "https://www.yelp.com/biz/ulta-beauty-new-york-6",
]

REVIEWS_PER_BUSINESS = 100
REVIEWS_PER_PAGE = 10
PAGE_LOAD_WAIT = 5
SCROLL_PAUSE = 1.2
INTER_BUSINESS_DELAY = (5, 10)
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


def build_page_url(business_url: str, start: int) -> str:
    if start <= 0:
        return business_url
    sep = "&" if "?" in business_url else "?"
    return f"{business_url}{sep}start={start}"


def scroll_for_lazy_load(driver: uc.Chrome) -> None:
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(6):
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(SCROLL_PAUSE)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def parse_business_info(soup: BeautifulSoup) -> tuple[str, str, str]:
    """Extract business name, overall rating, and total review count."""
    # Business name
    name_el = soup.select_one("h1")
    company_name = name_el.get_text(strip=True) if name_el else "N/A"

    # Overall rating — first star-rating widget outside of a review list
    overall_rating = "N/A"
    for el in soup.select("div[aria-label*='star rating']"):
        label = el.get("aria-label", "")
        m = re.search(r"(\d+(?:[.,]\d+)?)", label)
        if m:
            overall_rating = m.group(1).replace(",", ".")
            break

    # Total reviews count
    total_reviews = "N/A"
    for tag in soup.select("a, span, p"):
        text = tag.get_text(strip=True)
        m = re.match(r"^([\d,]+)\s+review", text, re.IGNORECASE)
        if m:
            total_reviews = m.group(1).replace(",", "")
            break

    return company_name, overall_rating, total_reviews


def parse_rating(container) -> str:
    el = container.select_one("div[aria-label*='star rating']")
    if not el:
        return "N/A"
    label = el.get("aria-label", "")
    m = re.search(r"(\d+(?:[.,]\d+)?)", label)
    if not m:
        return "N/A"
    try:
        return str(int(float(m.group(1).replace(",", "."))))
    except ValueError:
        return "N/A"


def parse_date(container) -> str:
    pattern = re.compile(
        r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}$"
    )
    for span in container.select("span"):
        text = span.get_text(strip=True)
        if text and pattern.match(text):
            return text
    return "N/A"


def parse_reviewer_name(container) -> str:
    for selector in [
        "a[href*='/user_details']",
        "span[class*='user-display-name']",
        "div[class*='user-display-name']",
        "a[href*='/userid']",
    ]:
        el = container.select_one(selector)
        if el:
            name = el.get_text(strip=True)
            if name:
                return name
    return "N/A"


def parse_reviews(
    html: str,
    company_name: str,
    overall_rating: str,
    total_reviews: str,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    reviews: list[dict[str, Any]] = []

    for p in soup.select("p[class*='comment__']"):
        container = p
        for _ in range(8):
            container = container.parent
            if container is None or container.name == "li":
                break
        if container is None or container.name != "li":
            continue
        if not container.select_one("div[aria-label*='star rating']"):
            continue

        review_text = p.get_text(" ", strip=True)
        if not review_text:
            continue

        reviews.append({
            "company_name": company_name,
            "overall_rating": overall_rating,
            "total_reviews": total_reviews,
            "review_title": "N/A",
            "review_text": review_text,
            "review_rating": parse_rating(container),
            "review_date": parse_date(container),
            "reviewer_name": parse_reviewer_name(container),
        })

    return reviews


def scrape_business(
    driver: uc.Chrome,
    business_url: str,
    target_count: int,
) -> list[dict[str, Any]]:
    print(f"  Fetching: {business_url}")
    collected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    company_name = overall_rating = total_reviews = "N/A"

    pages_needed = (target_count + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE
    for page_idx in range(pages_needed + 2):
        start = page_idx * REVIEWS_PER_PAGE
        page_url = build_page_url(business_url, start)

        try:
            driver.get(page_url)
        except WebDriverException as exc:
            print(f"    Page {page_idx + 1} failed: {exc}")
            continue

        time.sleep(PAGE_LOAD_WAIT)
        scroll_for_lazy_load(driver)
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "lxml")

        if page_idx == 0:
            company_name, overall_rating, total_reviews = parse_business_info(soup)
            print(f"    {company_name} | {overall_rating}★ | {total_reviews} reviews")

        page_reviews = parse_reviews(
            driver.page_source, company_name, overall_rating, total_reviews
        )

        new_count = 0
        for r in page_reviews:
            if r["review_text"] not in seen_texts:
                seen_texts.add(r["review_text"])
                collected.append(r)
                new_count += 1

        print(f"    Page {page_idx + 1}: +{new_count} new (total {len(collected)})")

        if len(collected) >= target_count:
            break
        if new_count == 0:
            print("    No new reviews, stopping.")
            break

    return collected[:target_count]


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
                delay = random.uniform(*INTER_BUSINESS_DELAY)
                time.sleep(delay)
            try:
                reviews = scrape_business(driver, business_url, REVIEWS_PER_BUSINESS)
            except Exception as exc:
                print(f"  Skipped '{business_url}': {exc}")
                continue
            all_reviews.extend(reviews)
            print(f"  Business total: {len(reviews)} | Grand total: {len(all_reviews)}")

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
>>>>>>> Stashed changes
