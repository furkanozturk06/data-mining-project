# scraper.py — Sitejabber E-Ticaret Yorum Kazıyıcısı
#
# Selenium kullanarak gerçek tarayıcı gibi davranır.
# Sitejabber yorumları JavaScript ile yüklendiği için
# Selenium olmadan içerik alınamaz.
# Tüm veriler pandas DataFrame'e aktarılarak CSV'ye kaydedilir.

import time
import random
import os

from bs4 import BeautifulSoup
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import (
    BASE_URL,
    TARGET_COMPANIES,
    MAX_PAGES_PER_COMPANY,
    OUTPUT_FILE,
    MIN_DELAY,
    MAX_DELAY,
    PAGE_LOAD_WAIT,
    ELEMENT_WAIT_TIMEOUT,
    USER_AGENTS,
)


def build_driver() -> webdriver.Chrome:
    """
    Bot tespitini zorlaştıracak ayarlarla Chrome WebDriver başlatır.
    """
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

    driver = webdriver.Chrome(options=options)

    # navigator.webdriver özelliğini gizle
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def random_delay() -> None:
    """
    İstekler arasında bot tespitinden kaçınmak için rastgele bekler.
    """
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    print(f"  Bekleniyor: {delay:.1f} saniye...")
    time.sleep(delay)


def scroll_page(driver: webdriver.Chrome) -> None:
    """
    Sayfayı yavaşça aşağı kaydırır. Lazy-load ile yüklenen
    yorumların DOM'a gelmesini sağlar.
    """
    total_height = driver.execute_script("return document.body.scrollHeight")
    current = 0
    step = 400
    while current < total_height:
        driver.execute_script(f"window.scrollTo(0, {current});")
        time.sleep(0.3)
        current += step
    # Sayfanın en altına git
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)


def parse_company_info(soup: BeautifulSoup) -> dict:
    """
    Sayfa HTML'inden şirketin genel puanını ve toplam yorum sayısını çeker.
    """
    overall_rating = None
    for selector in [
        "[class*='score__number']",
        "[class*='rating__number']",
        "span[itemprop='ratingValue']",
        "[class*='business-score']",
    ]:
        tag = soup.select_one(selector)
        if tag:
            try:
                overall_rating = float(tag.get_text(strip=True).replace(",", "."))
            except ValueError:
                pass
            break

    total_reviews = None
    for selector in [
        "[class*='reviews-count']",
        "[class*='total-reviews']",
        "[itemprop='reviewCount']",
        "[class*='score__reviews']",
    ]:
        tag = soup.select_one(selector)
        if tag:
            digits = "".join(filter(str.isdigit, tag.get_text()))
            if digits:
                total_reviews = int(digits)
            break

    return {"overall_rating": overall_rating, "total_reviews": total_reviews}


def parse_reviews(soup: BeautifulSoup, company_slug: str, company_info: dict) -> list:
    """
    Sayfa HTML'inden bireysel yorumları ayrıştırır.
    """
    reviews = []

    # Yorum kartı konteynerlerini bul
    containers = (
        soup.select("[class*='review-item']")
        or soup.select("[class*='ReviewItem']")
        or soup.select("article[class*='review']")
        or soup.select("[data-testid*='review']")
        or soup.select("[class*='review__']")
        or soup.select(".review")
    )

    if not containers:
        print(f"    '{company_slug}' icin yorum karti bulunamadi.")
        return reviews

    for container in containers:
        # Yorum başlığı
        title_tag = (
            container.select_one("[class*='review__title']")
            or container.select_one("[class*='review-title']")
            or container.select_one("h3")
            or container.select_one("h2")
        )
        review_title = title_tag.get_text(strip=True) if title_tag else None

        # Yorum metni
        text_tag = (
            container.select_one("[class*='review__body']")
            or container.select_one("[class*='review-text']")
            or container.select_one("[class*='review__content']")
            or container.select_one("[class*='content__review']")
            or container.select_one("p")
        )
        review_text = text_tag.get_text(strip=True) if text_tag else None

        # Yorum puanı
        review_rating = None
        rating_meta = container.select_one("meta[itemprop='ratingValue']")
        if rating_meta:
            try:
                review_rating = float(rating_meta.get("content", ""))
            except (ValueError, TypeError):
                pass
        else:
            for sel in [
                "[class*='rating__value']",
                "[class*='star-rating']",
                "[class*='review__rating']",
                "[aria-label*='star']",
            ]:
                tag = container.select_one(sel)
                if tag:
                    raw = tag.get("aria-label", "") or tag.get_text(strip=True)
                    digits = "".join(c for c in raw if c.isdigit() or c in ".,")
                    try:
                        review_rating = float(digits.replace(",", "."))
                    except ValueError:
                        pass
                    break

        # Yorum tarihi
        date_tag = (
            container.select_one("time")
            or container.select_one("[class*='review__date']")
            or container.select_one("[class*='review-date']")
            or container.select_one("[itemprop='datePublished']")
        )
        if date_tag:
            review_date = date_tag.get("datetime") or date_tag.get_text(strip=True)
        else:
            review_date = None

        # Yorumcu adı
        reviewer_tag = (
            container.select_one("[class*='reviewer__name']")
            or container.select_one("[class*='reviewer-name']")
            or container.select_one("[itemprop='author']")
            or container.select_one("[class*='user__name']")
            or container.select_one("[class*='user-name']")
        )
        reviewer_name = reviewer_tag.get_text(strip=True) if reviewer_tag else None

        # Tüm alanlar boşsa bu kartı atla
        if not any([review_title, review_text, reviewer_name]):
            continue

        reviews.append(
            {
                "company_name": company_slug,
                "overall_rating": company_info.get("overall_rating"),
                "total_reviews": company_info.get("total_reviews"),
                "review_title": review_title,
                "review_text": review_text,
                "review_rating": review_rating,
                "review_date": review_date,
                "reviewer_name": reviewer_name,
            }
        )

    return reviews


def get_reviews(driver: webdriver.Chrome, company_slug: str, max_pages: int = MAX_PAGES_PER_COMPANY) -> list:
    """
    Verilen şirket için tüm sayfalardaki yorumları Selenium ile çeker.

    Parametreler
    company_slug : Sitejabber URL'sindeki şirket kısa adı (örn. 'amazon.com')
    max_pages    : Kazınacak maksimum sayfa sayısı
    driver       : Önceden başlatılmış Chrome WebDriver nesnesi
    """
    all_reviews = []
    company_info = {}

    for page_num in range(1, max_pages + 1):
        url = BASE_URL.format(slug=company_slug, page=page_num)
        print(f"\n  Sayfa {page_num}/{max_pages} => {url}")

        try:
            driver.get(url)

            # Yorum kartlarının DOM'a yüklenmesini bekle
            try:
                WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR,
                         "[class*='review-item'], [class*='ReviewItem'], article[class*='review'], .review")
                    )
                )
            except Exception:
                # Zaman aşımı durumunda mevcut içerikle devam et
                print(f"    Element bekleme zaman asimi, devam ediliyor...")

            # Lazy-load yorumların yüklenmesi için sayfayı kaydır
            scroll_page(driver)

            # JS yüklemesinin tamamlanması için ekstra bekle
            time.sleep(PAGE_LOAD_WAIT)

        except Exception as nav_err:
            print(f"    Sayfa yukleme hatasi: {nav_err}")
            random_delay()
            continue

        # Yüklenen HTML'i BeautifulSoup ile ayrıştır
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Şirket genel bilgisini yalnızca ilk sayfada çek
        if page_num == 1:
            company_info = parse_company_info(soup)
            print(
                f"    Genel Puan: {company_info['overall_rating']} | "
                f"Toplam Yorum: {company_info['total_reviews']}"
            )

        page_reviews = parse_reviews(soup, company_slug, company_info)
        print(f"    Bu sayfadan {len(page_reviews)} yorum alindi.")
        all_reviews.extend(page_reviews)

        # Yorum bulunamadıysa bu şirket için döngüyü bitir
        if len(page_reviews) == 0:
            print("    Yorum kalmadi, sonraki sirkete geciliyor.")
            break

        if page_num < max_pages:
            random_delay()

    return all_reviews


def scrape_all() -> None:
    """
    TARGET_COMPANIES listesindeki tüm şirketler için get_reviews() çağırır,
    sonuçları birleştirip OUTPUT_FILE yoluna CSV olarak kaydeder.
    """
    print("=" * 60)
    print("Sitejabber Yorum Kaziyici Baslatiliyor (Selenium)")
    print(f"Hedef sirketler: {', '.join(TARGET_COMPANIES)}")
    print(f"Sirket basina max sayfa: {MAX_PAGES_PER_COMPANY}")
    print("=" * 60)

    # Tek bir driver tüm şirketler için paylaşılır (performans)
    driver = build_driver()
    print("Chrome WebDriver baslatildi.")

    all_data = []

    try:
        for idx, company in enumerate(TARGET_COMPANIES, start=1):
            print(f"\n[{idx}/{len(TARGET_COMPANIES)}] Sirket: {company}")
            print("*" * 40)

            company_reviews = get_reviews(driver, company, MAX_PAGES_PER_COMPANY)
            all_data.extend(company_reviews)

            print(f"  '{company}' tamamlandi => {len(company_reviews)} yorum.")

            # Son şirket değilse şirketler arası ek bekleme uygula
            if idx < len(TARGET_COMPANIES):
                extra_delay = random.uniform(4.0, 8.0)
                print(f"\n  Sirketler arasi {extra_delay:.1f} saniye bekleniyor...")
                time.sleep(extra_delay)

    finally:
        # Hata olsa da driver'ı kapat
        driver.quit()
        print("\nChrome WebDriver kapatildi.")

    print("\n" + "=" * 60)

    if not all_data:
        print("Hic veri toplanamadi.")
        print("Olasi nedenler:")
        print("  Sitejabber CSS yapisi degismis olabilir.")
        print("  Bot korumasi tum istekleri engellemis olabilir.")
        return

    # Çıktı klasörünü oluştur (yoksa)
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(all_data)

    # Sütun sırasını belirle
    column_order = [
        "company_name",
        "overall_rating",
        "total_reviews",
        "review_title",
        "review_text",
        "review_rating",
        "review_date",
        "reviewer_name",
    ]
    df = df[[col for col in column_order if col in df.columns]]

    # UTF-8 BOM ile kaydet (Excel Türkçe karakter uyumlu)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Kazima tamamlandi!")
    print(f"Toplam yorum : {len(df):,}")
    print(f"Kayit yeri   : {os.path.abspath(OUTPUT_FILE)}")
    print("=" * 60)


if __name__ == "__main__":
    scrape_all()
