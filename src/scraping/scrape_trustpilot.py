from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


COMPANIES = [
    # E-ticaret
    "www.amazon.com",
    "www.ebay.com",
    "www.walmart.com",
    "www.etsy.com",
    # Fintech / Ödeme
    "www.paypal.com",
    "www.klarna.com",
    # Seyahat / Konaklama
    "www.booking.com",
    "www.airbnb.com",
    "www.expedia.com",
    "www.tripadvisor.com",
    # Teknoloji / Sosyal
    "www.facebook.com",
    "www.netflix.com",
    # Yüksek puanlı (pozitif sınıf için)
    "www.chewy.com",
    "www.zappos.com",
    "www.shopify.com",
]

STAR_RATINGS = [1, 2, 3, 4, 5]
PAGES_PER_STAR = 4        # her yıldız × şirket başına sayfa sayısı (10 yorum/sayfa → 40 yorum)
PAGE_LOAD_WAIT = 6        # saniye
INTER_PAGE_DELAY = (2, 5)  # (min, max) saniye
INTER_COMPANY_DELAY = (5, 10)

BASE_URL = "https://www.trustpilot.com/review/{company}"


def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def build_page_url(company: str, stars: int, page: int) -> str:
    base = BASE_URL.format(company=company)
    return f"{base}?stars={stars}&page={page}"


def extract_business_info(driver: webdriver.Chrome) -> tuple[str, str, str]:
    """Extract company name, overall rating and total review count from JSON-LD."""
    js = """
    (() => {
        const scripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
        for (const s of scripts) {
            try {
                const data = JSON.parse(s.textContent);
                const graph = data['@graph'] || [data];
                for (const node of (Array.isArray(graph) ? graph : [graph])) {
                    if (node['@type'] === 'LocalBusiness' && node.aggregateRating) {
                        const ar = node.aggregateRating;
                        return JSON.stringify({
                            name: node.name || 'N/A',
                            rating: String(ar.ratingValue || 'N/A'),
                            count: String(ar.reviewCount || 'N/A')
                        });
                    }
                }
            } catch(e) {}
        }
        return null;
    })()
    """
    result = driver.execute_script(js)
    if not result:
        return "N/A", "N/A", "N/A"
    try:
        payload = json.loads(result)
        return payload.get("name", "N/A"), payload.get("rating", "N/A"), payload.get("count", "N/A")
    except json.JSONDecodeError:
        return "N/A", "N/A", "N/A"


def parse_reviews(
    html: str,
    company_name: str,
    overall_rating: str,
    total_reviews: str,
    page_url: str,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    reviews: list[dict[str, Any]] = []

    cards = soup.select("article[data-service-review-card-paper='true']")
    for card in cards:
        # Review text
        text_el = card.select_one("p[data-service-review-text-typography]")
        review_text = text_el.get_text(" ", strip=True) if text_el else ""
        if not review_text:
            continue  # skip carousel/promo cards without text

        # Review title
        title_el = card.select_one("h2[data-service-review-title-typography]")
        review_title = title_el.get_text(" ", strip=True) if title_el else "N/A"

        # Star rating from <img alt="Rated X out of 5 stars">
        review_rating = "N/A"
        img_el = card.select_one("img[alt^='Rated']")
        if img_el:
            alt = img_el.get("alt", "")
            m = re.search(r"(\d+)", alt)
            if m:
                review_rating = m.group(1)

        # Date
        review_date = "N/A"
        time_el = card.select_one("time")
        if time_el:
            review_date = time_el.get("datetime") or time_el.get_text(strip=True) or "N/A"

        # Reviewer name
        name_el = card.select_one("span[data-consumer-name-typography]")
        reviewer_name = name_el.get_text(strip=True) if name_el else "N/A"

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


def scrape_company(
    driver: webdriver.Chrome,
    company: str,
) -> list[dict[str, Any]]:
    all_reviews: list[dict[str, Any]] = []

    # Load unfiltered first page to get business metadata from JSON-LD
    base_url = BASE_URL.format(company=company)
    try:
        driver.get(base_url)
        WebDriverWait(driver, PAGE_LOAD_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "script[type='application/ld+json']"))
        )
    except Exception:
        pass
    company_name, overall_rating, total_reviews_str = extract_business_info(driver)
    if company_name == "N/A":
        company_name = company  # fallback to domain slug
    print(f"  Company: {company_name} | Rating: {overall_rating} | Total: {total_reviews_str}")

    for stars in STAR_RATINGS:
        star_reviews: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        for page in range(1, PAGES_PER_STAR + 1):
            url = build_page_url(company, stars, page)
            print(f"    [{stars}★ p{page}] {url}")

            try:
                driver.get(url)
            except WebDriverException as exc:
                print(f"      Navigation failed: {exc}")
                continue

            # Wait for review cards or "no reviews" state
            try:
                WebDriverWait(driver, PAGE_LOAD_WAIT).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "article[data-service-review-card-paper='true']")
                    )
                )
            except TimeoutException:
                print("      No review cards found, skipping remaining pages for this star.")
                break

            # (business info already fetched above)

            page_reviews = parse_reviews(
                driver.page_source,
                company_name,
                overall_rating,
                total_reviews_str,
                url,
            )

            new_count = 0
            for r in page_reviews:
                if r["review_text"] not in seen_texts:
                    seen_texts.add(r["review_text"])
                    star_reviews.append(r)
                    new_count += 1

            print(f"      +{new_count} new (star total: {len(star_reviews)})")

            if new_count == 0:
                print("      No new reviews, stopping star tier.")
                break

            time.sleep(random.uniform(*INTER_PAGE_DELAY))

        all_reviews.extend(star_reviews)

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

    filename = f"trustpilot_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = output_dir / filename
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> None:
    driver = build_driver()
    all_reviews: list[dict[str, Any]] = []

    try:
        for idx, company in enumerate(COMPANIES):
            print(f"\n=== [{idx + 1}/{len(COMPANIES)}] {company} ===")
            if idx > 0:
                delay = random.uniform(*INTER_COMPANY_DELAY)
                print(f"  Waiting {delay:.1f}s before next company...")
                time.sleep(delay)

            try:
                company_reviews = scrape_company(driver, company)
            except Exception as exc:
                print(f"  Skipped due to error: {exc}")
                continue

            all_reviews.extend(company_reviews)
            print(f"  Company total: {len(company_reviews)} | Grand total: {len(all_reviews)}")

        print(f"\nTotal reviews collected: {len(all_reviews)}")

        if not all_reviews:
            print("No reviews collected.")
            return

        output_path = save_to_csv(all_reviews)
        print(f"Saved to: {output_path}")

        # Print rating distribution
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
