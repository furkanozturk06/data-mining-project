"""Batch sentiment labeling for conflicting reviews using Groq API.

Usage:
    export GROQ_API_KEY="..."
    python src/preprocessing/groq_label_reviews.py
    python src/preprocessing/groq_label_reviews.py --limit 100   # test modu
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from groq import Groq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import GROQ_API_KEY

INPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "all_reviews_clean.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "all_reviews_labeled.csv")

BATCH_SIZE = 20
CHECKPOINT_EVERY = 5
# 6000 token/dk ücretsiz tier: her çağrı ~740 token → ~8 çağrı/dk → ~7.5sn bekleme
SLEEP_BETWEEN_BATCHES = 8.0

POSITIVE_KW = [
    "harika", "mükemmel", "süper", "güzel", "teşekkür", "tesekkur",
    "sevdim", "beğendim", "begendim", "muhteşem", "muthesem", "başarılı",
    "basarili", "iyi", "memnun", "tavsiye", "öneririm", "oneririm",
    "kusursuz", "fevkalade", "şahane", "sahane",
]
NEGATIVE_KW = [
    "berbat", "rezalet", "kötü", "kotü", "kotu", "sorun", "problem",
    "çalışmıyor", "calismiyor", "donuyor", "açılmıyor", "acilmiyor",
    "hata", "crash", "çöküyor", "cokuyor", "bozuk", "mahvettin",
    "mahvettiniz", "berbat", "rezil", "saçma", "sacma", "işe yaramaz",
    "ise yaramaz", "para iadesi", "siliyorum",
]


def star_label(rating: float) -> str:
    if rating <= 2:
        return "negatif"
    if rating == 3:
        return "nötr"
    return "pozitif"


def is_conflicting(text: str, rating: float) -> bool:
    t = str(text).lower()
    has_pos = any(kw in t for kw in POSITIVE_KW)
    has_neg = any(kw in t for kw in NEGATIVE_KW)
    if rating <= 2 and has_pos and not has_neg:
        return True
    if rating >= 4 and has_neg and not has_pos:
        return True
    return False


def parse_labels(response_text: str, n: int) -> list[str]:
    """Parse Groq response into a list of n labels."""
    valid = {"pozitif", "nötr", "notr", "negatif"}
    lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]

    labels = []
    for line in lines:
        # Strip leading numbering like "1. " or "1) "
        for sep in [". ", ") ", "- "]:
            if sep in line:
                line = line.split(sep, 1)[-1]
        word = line.strip().lower().rstrip(".")
        if word in valid:
            labels.append("nötr" if word == "notr" else word)

    # Pad or trim to exactly n
    while len(labels) < n:
        labels.append(None)
    return labels[:n]


def label_batch(client: Groq, rows: list[dict]) -> list[str]:
    numbered = "\n".join(
        f"{i+1}. [yıldız:{int(r['rating'])}] {str(r['text'])[:150]}"
        for i, r in enumerate(rows)
    )
    prompt = (
        "Aşağıdaki yorumları etiketle. Her satır için SADECE şu üç kelimeden birini yaz: "
        "pozitif, nötr, negatif\n"
        "Her satır için tek kelime, sırayla:\n\n"
        f"{numbered}"
    )
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    return parse_labels(raw, len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Kaç çelişkili yorum etiketlensin (test modu)")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("HATA: GROQ_API_KEY ayarlanmamış. Önce 'export GROQ_API_KEY=...' çalıştırın.")
        sys.exit(1)

    print(f"Veri yükleniyor: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)

    # Eğer daha önce etiketlenmiş dosya varsa ondan başla
    if os.path.exists(OUTPUT_CSV):
        print(f"Mevcut checkpoint bulundu: {OUTPUT_CSV}")
        labeled_df = pd.read_csv(OUTPUT_CSV)
        if "groq_labeled" not in labeled_df.columns:
            labeled_df["groq_labeled"] = False
        # Sadece Groq tarafından işlenmiş olanları atla
        already_labeled = set(labeled_df[labeled_df["groq_labeled"] == True]["review_id"].tolist())
    else:
        labeled_df = df.copy()
        labeled_df["sentiment_label"] = labeled_df["rating"].apply(star_label)
        labeled_df["groq_labeled"] = False
        already_labeled = set()

    # Henüz Groq ile etiketlenmemiş tüm yorumlar
    remaining_df = df[~df["review_id"].isin(already_labeled)].copy()

    if args.limit:
        remaining_df = remaining_df.head(args.limit)

    total = len(remaining_df)
    print(f"Toplam {len(df):,} yorum, {len(already_labeled):,} zaten etiketlendi.")
    print(f"Etiketlenecek: {total:,} yorum ({BATCH_SIZE}'li batch'ler)")

    if total == 0:
        print("Etiketlenecek yorum yok.")
        labeled_df.to_csv(OUTPUT_CSV, index=False)
        return

    client = Groq(api_key=GROQ_API_KEY)
    processed = 0
    batch_count = 0

    rows = remaining_df.to_dict("records")

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        try:
            labels = label_batch(client, batch)
        except Exception as e:
            print(f"  Batch hata ({e}), yıldız kuralı kullanılıyor.")
            labels = [None] * len(batch)

        for row, label in zip(batch, labels):
            final_label = label if label else star_label(row["rating"])
            mask = labeled_df["review_id"] == row["review_id"]
            labeled_df.loc[mask, "sentiment_label"] = final_label
            labeled_df.loc[mask, "groq_labeled"] = True

        processed += len(batch)
        batch_count += 1
        pct = processed / total * 100
        msg = f"[{processed}/{total} ({pct:.0f}%)] batch {batch_count} tamamlandı"
        print(msg, flush=True)

        if batch_count % CHECKPOINT_EVERY == 0:
            labeled_df.to_csv(OUTPUT_CSV, index=False)
            # İlerleme log dosyasına yaz (takip için)
            log_path = OUTPUT_CSV.replace(".csv", "_progress.txt")
            with open(log_path, "w") as f:
                f.write(f"{processed}/{total} ({pct:.1f}%) - batch {batch_count}\n")
            print(f"  Kaydedildi ({pct:.0f}% tamamlandı)", flush=True)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    labeled_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTamamlandı. Çıktı: {OUTPUT_CSV}")

    label_counts = labeled_df["sentiment_label"].value_counts()
    print("\nEtiket dağılımı:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,}")


if __name__ == "__main__":
    main()
