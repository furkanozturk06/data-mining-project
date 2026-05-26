"""nlptown multilingual BERT sentiment model — yükleme ve tahmin yardımcıları.

Model: nlptown/bert-base-multilingual-uncased-sentiment
5 sınıflı yıldız tahmini (1-5). Türkçe dahil 6 dilde ince ayarlı.
Burada hem 5-sınıflı yıldız hem de 3-sınıflı sentiment (negatif/nötr/pozitif) sunulur.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

STAR_TO_SENTIMENT = {1: "negatif", 2: "negatif", 3: "nötr", 4: "pozitif", 5: "pozitif"}
SENTIMENT_CLASSES = ["negatif", "nötr", "pozitif"]


@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _as_list(texts) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    return [str(t) if t is not None else "" for t in texts]


@torch.no_grad()
def predict_stars(texts, batch_size: int = 16, max_length: int = 256):
    """Her metin için 5 yıldız olasılığı ve argmax yıldız tahminini döndür."""
    tokenizer, model, device = load_model()
    texts = _as_list(texts)
    all_probs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    star_probs = np.vstack(all_probs)  # (n, 5)
    stars = star_probs.argmax(axis=1) + 1
    return star_probs, stars


def stars_to_3class_probs(star_probs: np.ndarray) -> np.ndarray:
    """5-sınıf olasılığı 3-sınıf sentiment olasılığına indir."""
    neg = star_probs[:, 0] + star_probs[:, 1]
    neu = star_probs[:, 2]
    pos = star_probs[:, 3] + star_probs[:, 4]
    return np.column_stack([neg, neu, pos])


def stars_to_sentiment(stars: Iterable[int]) -> List[str]:
    return [STAR_TO_SENTIMENT[int(s)] for s in stars]


def predict_3class(texts, batch_size: int = 16) -> np.ndarray:
    """LIME / SHAP için gerekli (n, 3) olasılık matrisi: [negatif, nötr, pozitif]."""
    star_probs, _ = predict_stars(texts, batch_size=batch_size)
    return stars_to_3class_probs(star_probs)


def predict_sentiment(text: str) -> dict:
    """Tek bir metin için zengin tahmin sözlüğü."""
    star_probs, stars = predict_stars(text)
    star = int(stars[0])
    s3 = stars_to_3class_probs(star_probs)[0]
    sentiment_idx = int(np.argmax(s3))
    return {
        "star": star,
        "sentiment": SENTIMENT_CLASSES[sentiment_idx],
        "sentiment_confidence": float(s3[sentiment_idx]),
        "star_probs": star_probs[0].tolist(),
        "sentiment_probs": s3.tolist(),
    }
