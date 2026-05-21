from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "reviews_cleaned.csv"

# Sitejabber dahil edilmiyor (rating yok)
RAW_FILES = {
    "trustpilot": "trustpilot_balanced_20260520_231900.csv",
    "yelp":       "yelp_reviews_20260521_012500.csv",
}

SENTIMENT_MAP = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)           # HTML tag temizle
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URL kaldır
    text = re.sub(r"[^\w\s'\".,!?-]", " ", text)   # özel karakterleri kaldır
    text = re.sub(r"\s+", " ", text)                # fazla boşlukları tek boşluğa indir
    return text.strip()


def load_and_tag(path: Path, platform: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["platform"] = platform
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for platform, filename in RAW_FILES.items():
        path = RAW_DIR / filename
        if not path.exists():
            print(f"[UYARI] Dosya bulunamadı, atlanıyor: {path}")
            continue
        df = load_and_tag(path, platform)
        frames.append(df)
        print(f"  Yüklendi [{platform}]: {len(df)} satır")

    if not frames:
        print("Hiç dosya yüklenemedi.")
        return

    df = pd.concat(frames, ignore_index=True)
    print(f"\nBirleştirildi: {len(df)} satır")

    # ── 1. Gerekli sütunları seç ──────────────────────────────────────────────
    keep = ["platform", "company_name", "review_title", "review_text",
            "review_rating", "review_date", "reviewer_name"]
    df = df[[c for c in keep if c in df.columns]]

    # ── 2. Boş review_text & rating satırlarını düşür ─────────────────────────
    df = df[df["review_text"].notna() & (df["review_text"].str.strip() != "")]
    df = df[df["review_rating"].notna() & (df["review_rating"].str.strip() != "") & (df["review_rating"] != "N/A")]
    print(f"Boş text/rating düşürüldü: {len(df)} satır kaldı")

    # ── 3. Rating'i tam sayıya çevir ──────────────────────────────────────────
    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    df = df[df["review_rating"].notna() & df["review_rating"].between(1, 5)]
    df["review_rating"] = df["review_rating"].astype(int)
    print(f"Geçersiz rating düşürüldü: {len(df)} satır kaldı")

    # ── 4. Metin temizliği ────────────────────────────────────────────────────
    df["review_text"] = df["review_text"].apply(clean_text)

    # Çok kısa yorumları at (10 karakterden az)
    df = df[df["review_text"].str.len() >= 10]
    print(f"Kısa metin düşürüldü: {len(df)} satır kaldı")

    # ── 5. Duplicate temizliği ────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["review_text"]).reset_index(drop=True)
    print(f"Duplikasyon temizliği: {before - len(df)} satır kaldırıldı → {len(df)} satır kaldı")

    # ── 6. Sentiment etiketi ──────────────────────────────────────────────────
    df["sentiment"] = df["review_rating"].map(SENTIMENT_MAP)

    # ── 7. Sütun sırası & kayıt ───────────────────────────────────────────────
    final_cols = ["platform", "company_name", "review_title", "review_text",
                  "review_rating", "sentiment", "review_date", "reviewer_name"]
    df = df[[c for c in final_cols if c in df.columns]]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nKaydedildi: {OUTPUT_FILE}")

    # ── 8. Özet rapor ────────────────────────────────────────────────────────
    print("\n── Platform Dağılımı ──")
    print(df["platform"].value_counts().to_string())

    print("\n── Rating Dağılımı ──")
    print(df["review_rating"].value_counts().sort_index().to_string())

    print("\n── Sentiment Dağılımı ──")
    print(df["sentiment"].value_counts().to_string())

    print(f"\nToplam temiz yorum: {len(df)}")


if __name__ == "__main__":
    main()
