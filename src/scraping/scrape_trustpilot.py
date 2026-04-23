from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import re
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def fetch_page_with_selenium(url: str) -> str:
    """Open target URL in Chrome and return page source."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")

    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)

        # Short static wait for dynamic review blocks to render.
        time.sleep(3)

        html = driver.page_source
        if not html or len(html.strip()) < 100:
            raise RuntimeError("Failed to load page source with Selenium.")

        return html
    except WebDriverException as exc:
        raise RuntimeError(f"Selenium error: {exc}") from exc
    finally:
        if driver is not None:
            driver.quit()


def build_page_url(base_url: str, page_number: int) -> str:
    """Create page URL for Trustpilot pagination."""
    if page_number <= 1:
        return base_url

    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}page={page_number}"


def parse_reviews(html: str) -> list[dict[str, Any]]:
    """Parse review blocks and extract core fields."""
    soup = BeautifulSoup(html, "lxml")
    reviews: list[dict[str, Any]] = []

    # Trustpilot commonly exposes review data in JSON-LD scripts.
    json_ld_scripts = soup.select('script[type="application/ld+json"]')
    for script in json_ld_scripts:
        raw_text = script.get_text(strip=True)
        if not raw_text:
            continue

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            continue

        graph = payload.get("@graph", []) if isinstance(payload, dict) else []
        for node in graph:
            if not isinstance(node, dict) or node.get("@type") != "Review":
                continue

            review_text = str(node.get("reviewBody", "")).strip()

            rating = None
            review_rating = node.get("reviewRating")
            if isinstance(review_rating, dict):
                raw_rating = review_rating.get("ratingValue")
                if raw_rating is not None:
                    try:
                        rating = int(float(str(raw_rating)))
                    except ValueError:
                        rating = None

            review_date = node.get("datePublished")

            if not review_text:
                continue

            reviews.append(
                {
                    "platform": "trustpilot",
                    "review_text": review_text,
                    "rating": rating,
                    "review_date": review_date,
                    "source_url": "",
                }
            )

    if reviews:
        return reviews

    # Fallback parser in case JSON-LD format changes.
    review_blocks = soup.select("article[data-service-review-id]")
    if not review_blocks:
        review_blocks = soup.select("article")

    for block in review_blocks:
        text_element = block.select_one("p[data-service-review-text-typography]")
        if not text_element:
            text_element = block.select_one("p")

        review_text = text_element.get_text(" ", strip=True) if text_element else ""

        rating = None
        rating_img = block.select_one('img[alt*="Rated"]')
        if rating_img and rating_img.get("alt"):
            alt_text = rating_img.get("alt", "")
            match = re.search(r"(\d)", alt_text)
            if match:
                rating = int(match.group(1))

        review_date = None
        date_element = block.select_one("time")
        if date_element:
            review_date = date_element.get("datetime") or date_element.get_text(strip=True)

        if not review_text or rating is None:
            continue

        reviews.append(
            {
                "platform": "trustpilot",
                "review_text": review_text,
                "rating": rating,
                "review_date": review_date,
                "source_url": "",
            }
        )

    return reviews


def save_to_csv(data: list[dict[str, Any]]) -> Path:
    """Convert data to DataFrame and save it under data/raw/."""
    columns = ["platform", "review_text", "rating", "review_date", "source_url"]
    df = pd.DataFrame(data, columns=columns)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_text", "review_date"]).reset_index(drop=True)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"trustpilot_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = output_dir / filename
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    base_url = "https://www.trustpilot.com/review/www.tripadvisor.com"
    max_pages = 30

    try:
        all_reviews: list[dict[str, Any]] = []

        for page_number in range(1, max_pages + 1):
            page_url = build_page_url(base_url, page_number)
            print(f"Fetching page {page_number}: {page_url}")

            try:
                html = fetch_page_with_selenium(page_url)
                page_reviews = parse_reviews(html)
            except Exception as page_exc:
                print(f"Page {page_number} skipped due to error: {page_exc}")
                continue

            for review in page_reviews:
                review["source_url"] = page_url

            print(f"Page {page_number} review count: {len(page_reviews)}")
            all_reviews.extend(page_reviews)

        print(f"Total parsed review count: {len(all_reviews)}")

        if not all_reviews:
            print("No reviews were parsed from the page.")
            return

        output_file = save_to_csv(all_reviews)
        print(f"Saved reviews to: {output_file}")
    except Exception as exc:
        print(f"Scraping failed: {exc}")


if __name__ == "__main__":
    main()