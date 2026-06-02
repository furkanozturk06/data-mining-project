"""TF-IDF based word importance and word cloud generation for reviews."""

import io
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Common Turkish stopwords
TURKISH_STOPWORDS = [
    "bir", "bu", "ve", "da", "de", "ile", "için", "ama", "çok", "var",
    "yok", "olan", "olan", "bu", "şu", "o", "ben", "sen", "biz", "siz",
    "onlar", "ne", "ki", "mi", "mu", "mı", "mü", "ya", "ye", "daha",
    "en", "gibi", "kadar", "her", "hiç", "bile", "benim", "senin",
    "onu", "bunu", "şunu", "oldu", "olur", "olsa", "ise", "iken",
    "hem", "ya", "veya", "ancak", "fakat", "lakin", "sadece", "yani",
    "artık", "zaten", "hep", "hiçbir", "bazı", "birçok", "çünkü",
    "eğer", "şimdi", "sonra", "önce", "hala", "hâlâ", "nasıl", "neden",
    "niye", "nereden", "nerede", "niçin", "hangi", "kendi", "kendisi",
    "uygulama", "app", "the", "is", "in", "it", "of", "to", "and",
]


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\d+", " ", text)
    return text.strip()


def get_tfidf_top_words(
    df: pd.DataFrame,
    sentiment_col: str = "rating",
    text_col: str = "text",
    n: int = 20,
) -> dict:
    """
    Return top-n TF-IDF words for positive (>=4), neutral (==3), and negative (<=2) reviews.
    Returns {'positive': [(word, score)], 'neutral': [...], 'negative': [(word, score)]}.
    """
    df = df.dropna(subset=[text_col, sentiment_col]).copy()
    df["clean"] = df[text_col].apply(_clean_text)

    # rating == 3.0 comparison works for both int and float dtypes
    pos_texts = [t for t in df[df[sentiment_col] >= 4]["clean"].tolist() if t.strip()]
    neu_texts = [t for t in df[df[sentiment_col].between(2.5, 3.5)]["clean"].tolist() if t.strip()]
    neg_texts = [t for t in df[df[sentiment_col] <= 2]["clean"].tolist() if t.strip()]

    results = {}
    for label, texts in [("positive", pos_texts), ("neutral", neu_texts), ("negative", neg_texts)]:
        if len(texts) < 2:
            results[label] = []
            continue
        # min_df=1 so single-occurrence words aren't silently dropped for smaller groups
        min_df = 2 if len(texts) >= 50 else 1
        vectorizer = TfidfVectorizer(
            stop_words=TURKISH_STOPWORDS,
            max_features=500,
            min_df=min_df,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        vocab = vectorizer.get_feature_names_out()
        if len(vocab) == 0:
            results[label] = []
            continue
        scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_idx = scores.argsort()[::-1][:n]
        results[label] = [(vocab[i], float(scores[i])) for i in top_idx]

    return results


def generate_wordcloud_figure(words_scores: list[tuple[str, float]], title: str, color: str = "viridis") -> bytes:
    """Generate a word cloud PNG image from (word, score) list. Returns PNG bytes."""
    freq_dict = {word: score for word, score in words_scores if len(word) > 2}
    if not freq_dict:
        return b""

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=color,
        max_words=100,
        font_path=None,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def get_top_bottom_reviews(df: pd.DataFrame, app_name: str, n: int = 5) -> dict:
    """Return top-n and bottom-n reviews for a given app name (case-insensitive)."""
    filtered = df[df["app_name"].str.lower() == app_name.strip().lower()].copy()
    filtered = filtered.dropna(subset=["text"])
    filtered = filtered[filtered["text"].str.strip() != ""]

    top = (
        filtered.sort_values("rating", ascending=False)
        .drop_duplicates(subset=["text"])
        .head(n)[["author", "rating", "text", "date", "platform"]]
        .to_dict("records")
    )
    bottom = (
        filtered.sort_values("rating", ascending=True)
        .drop_duplicates(subset=["text"])
        .head(n)[["author", "rating", "text", "date", "platform"]]
        .to_dict("records")
    )
    return {"top": top, "bottom": bottom, "total": len(filtered)}


def get_available_apps(df: pd.DataFrame) -> list[str]:
    """Return sorted unique app names from the dataset."""
    return sorted(df["app_name"].dropna().unique().tolist())
