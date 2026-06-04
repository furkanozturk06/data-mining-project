"""Groq-powered sentiment prediction and per-app review summarizer."""

import json
import pandas as pd
from groq import Groq
from src.config import GROQ_API_KEY

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY ayarlanmamış. 'export GROQ_API_KEY=...' ile ayarlayın.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def predict_sentiment(text: str) -> dict:
    """Return Groq-based sentiment prediction: {label, positive, negative}."""
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen bir Türkçe duygu analizi asistanısın. "
                    "Verilen yorumu analiz edip SADECE JSON döndür. "
                    'Format: {"label": "positive" or "negative", "positive": float, "negative": float} '
                    "positive + negative = 1.0 olmalı."
                ),
            },
            {"role": "user", "content": f'Yorumu analiz et: "{text}"'},
        ],
        response_format={"type": "json_object"},
        max_tokens=80,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        label = result.get("label", "negative")
        pos = float(result.get("positive", 0.5))
        neg = float(result.get("negative", 0.5))
        total = pos + neg
        if total > 0:
            pos, neg = pos / total, neg / total
        return {"label": label, "positive": round(pos, 3), "negative": round(neg, 3)}
    except Exception:
        return {"label": "negative", "positive": 0.25, "negative": 0.75}


def generate_app_summary(df: pd.DataFrame, app_name: str) -> str:
    """Generate a short Turkish summary for an app based on its reviews."""
    subset = df[df["app_name"].str.lower() == app_name.strip().lower()].copy()
    subset = subset.dropna(subset=["text"])
    subset = subset[subset["text"].str.strip().str.len() > 10]

    if subset.empty:
        return "Bu uygulama için yeterli yorum bulunamadı."

    # Dengeli örnek: en fazla 15 pozitif, 5 nötr, 15 negatif + 15 rastgele
    pos = subset[subset["rating"] >= 4].sample(min(15, len(subset[subset["rating"] >= 4])), random_state=42)
    neu = subset[subset["rating"] == 3].sample(min(5, len(subset[subset["rating"] == 3])), random_state=42)
    neg = subset[subset["rating"] <= 2].sample(min(15, len(subset[subset["rating"] <= 2])), random_state=42)
    rest = subset.sample(min(15, len(subset)), random_state=42)

    sample = pd.concat([pos, neu, neg, rest]).drop_duplicates(subset=["text"]).head(50)

    review_lines = []
    for _, row in sample.iterrows():
        stars = int(row["rating"]) if pd.notna(row["rating"]) else "?"
        text = str(row["text"])[:200]
        review_lines.append(f"[{stars} yıldız] {text}")

    reviews_text = "\n".join(review_lines)

    prompt = f"""{app_name} uygulamasına ait kullanıcı yorumları aşağıda verilmiştir.
Bu yorumları analiz ederek uygulamayı potansiyel kullanıcılara tanıtan kısa, tarafsız ve bilgilendirici bir Türkçe özet yaz.
Özet 3-5 cümle olmalı. Genel memnuniyet düzeyini, öne çıkan olumlu yönleri ve sık dile getirilen şikayetleri belirt.

Yorumlar:
{reviews_text}

Özet:"""

    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
