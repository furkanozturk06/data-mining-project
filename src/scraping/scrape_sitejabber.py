from __future__ import annotations

import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


TARGET_COMPANIES = [
    "amazon.com",
    "ebay.com",
    "walmart.com",
    "etsy.com",
    "shein.com",
    "temu.com",
    "bestbuy.com",
    "target.com",
    "wayfair.com",
    "chewy.com",
    "zappos.com",
    "newegg.com",
    "overstock.com",
    "homedepot.com",
    "macys.com",
]

BASE_URL = "https://www.sitejabber.com/reviews/{slug}?page={page}"
MAX_PAGES_PER_COMPANY = 5
MIN_DELAY = 2.0
MAX_DELAY = 5.0
PAGE_LOAD_WAIT = 3
ELEMENT_WAIT_TIMEOUT = 10

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]


def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def scroll_page(driver: webdriver.Chrome) -> None:
    total_height = driver.execute_script("return document.body.scrollHeight")
    current = 0
    while current < total_height:
        driver.execute_script(f"window.scrollTo(0, {current});")
        time.sleep(0.3)
        current += 400
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)


def parse_company_info(soup: BeautifulSoup) -> tuple[str, str, str]:
    company_name = "N/A"
    for sel in ["h1[class*='business-name']", "h1[class*='company']", "h1"]:
        el = soup.select_one(sel)
        if el:
            company_name = el.get_text(strip=True)
            break

    overall_rating = "N/A"
    for sel in ["[class*='score__number']", "[class*='rating__number']", "span[itemprop='ratingValue']"]:
        el = soup.select_one(sel)
        if el:
            try:
                overall_rating = str(float(el.get_text(strip=True).replace(",", ".")))
            except ValueError:
                pass
            break

    total_reviews = "N/A"
    for sel in ["[class*='reviews-count']", "[class*='total-reviews']", "[itemprop='reviewCount']", "[class*='score__reviews']"]:
        el = soup.select_one(sel)
        if el:
            digits = "".join(filter(str.isdigit, el.get_text()))
            if digits:
                total_reviews = digits
            break

    return company_name, overall_rating, total_reviews


def parse_reviews(
    soup: BeautifulSoup,
    company_name: str,
    overall_rating: str,
    total_reviews: str,
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []

    containers = (
        soup.select("[class*='review-item']")
        or soup.select("[class*='ReviewItem']")
        or soup.select("article[class*='review']")
        or soup.select("[data-testid*='review']")
        or soup.select("[class*='review__']")
        or soup.select(".review")
    )

    for container in containers:
        # Title
        title_el = (
            container.select_one("[class*='review__title']")
            or container.select_one("[class*='review-title']")
            or container.select_one("h3")
        )
        review_title = title_el.get_text(strip=True) if title_el else "N/A"

        # Text
        text_el = (
            container.select_one("[class*='review__body']")
            or container.select_one("[class*='review-text']")
            or container.select_one("[class*='review__content']")
            or container.select_one("[class*='content__review']")
            or container.select_one("p")
        )
        review_text = text_el.get_text(strip=True) if text_el else ""
        if not review_text:
            continue

        # Rating
        review_rating = "N/A"
        rating_meta = container.select_one("meta[itemprop='ratingValue']")
        if rating_meta:
            try:
                review_rating = str(int(float(rating_meta.get("content", ""))))
            except (ValueError, TypeError):
                pass
        else:
            for sel in ["[class*='rating__value']", "[class*='star-rating']", "[aria-label*='star']"]:
                el = container.select_one(sel)
                if el:
                    raw = el.get("aria-label", "") or el.get_text(strip=True)
                    m = re.search(r"(\d+(?:[.,]\d+)?)", raw)
                    if m:
                        try:
                            review_rating = str(int(float(m.group(1).replace(",", "."))))
                        except ValueError:
                            pass
                    break

        # Date
        review_date = "N/A"
        date_el = (
            container.select_one("time")
            or container.select_one("[class*='review__date']")
            or container.select_one("[itemprop='datePublished']")
        )
        if date_el:
            review_date = date_el.get("datetime") or date_el.get_text(strip=True) or "N/A"

        # Reviewer name
        reviewer_name = "N/A"
        for sel in ["[class*='reviewer__name']", "[class*='reviewer-name']", "[itemprop='author']", "[class*='user__name']"]:
            el = container.select_one(sel)
            if el:
                name = el.get_text(strip=True)
                if name:
                    reviewer_name = name
                    break

        reviews.append({
            "company_name": company_name,
            "overall_rating": overall_rating,
            "total_reviews": total_reviews,
            "review_title": review_title,
            "review_text": review_text,
            "review_rating": review_rating,
            "review_date": review_date,
            "reviewer_name": reviewer_name,
        })

    return reviews


def scrape_company(driver: webdriver.Chrome, slug: str) -> list[dict[str, Any]]:
    all_reviews: list[dict[str, Any]] = []
    company_name = overall_rating = total_reviews = "N/A"

    for page_num in range(1, MAX_PAGES_PER_COMPANY + 1):
        url = BASE_URL.format(slug=slug, page=page_num)
        print(f"    [p{page_num}] {url}")

        try:
            driver.get(url)
            try:
                WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR,
                        "[class*='review-item'], [class*='ReviewItem'], article[class*='review'], .review"))
                )
            except Exception:
                pass
            scroll_page(driver)
            time.sleep(PAGE_LOAD_WAIT)
        except Exception as exc:
            print(f"    Page load error: {exc}")
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            continue

        soup = BeautifulSoup(driver.page_source, "html.parser")

        if page_num == 1:
            company_name, overall_rating, total_reviews = parse_company_info(soup)
            print(f"    {company_name} | {overall_rating}★ | {total_reviews} reviews")

        page_reviews = parse_reviews(soup, company_name, overall_rating, total_reviews)
        print(f"    +{len(page_reviews)} reviews")
        all_reviews.extend(page_reviews)

        if len(page_reviews) == 0:
            print("    No reviews found, stopping.")
            break

        if page_num < MAX_PAGES_PER_COMPANY:
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    return all_reviews


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

    filename = f"sitejabber_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = output_dir / filename
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    driver = build_driver()
    all_reviews: list[dict[str, Any]] = []

    try:
        for idx, slug in enumerate(TARGET_COMPANIES):
            print(f"\n=== [{idx + 1}/{len(TARGET_COMPANIES)}] {slug} ===")
            if idx > 0:
                delay = random.uniform(4.0, 8.0)
                print(f"  Waiting {delay:.1f}s...")
                time.sleep(delay)
            try:
                reviews = scrape_company(driver, slug)
            except Exception as exc:
                print(f"  Skipped: {exc}")
                continue
            all_reviews.extend(reviews)
            print(f"  Company: {len(reviews)} | Total: {len(all_reviews)}")

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
