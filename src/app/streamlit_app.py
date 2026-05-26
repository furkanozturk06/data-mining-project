"""Streamlit arayüzü — yorum sentiment tahmini + uygulama bazlı en iyi/en kötü yorum arama.

Çalıştırmak için (proje kök dizininde):
    streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
SENTIMENT_CLASSES = ["negatif", "nötr", "pozitif"]


def _lazy_bert():
    """transformers/torch sadece tahmin anında yüklensin (büyük paketler)."""
    from src.models.bert import load_model, predict_sentiment  # noqa: WPS433
    return load_model, predict_sentiment

SENTIMENT_COLORS = {
    "pozitif": "#2ca02c",
    "negatif": "#d62728",
    "nötr": "#7f7f7f",
}
SENTIMENT_EMOJI = {"pozitif": "✅", "negatif": "❌", "nötr": "⚪"}

st.set_page_config(
    page_title="Müşteri Yorumu Sentiment Analizi",
    page_icon="💬",
    layout="wide",
)


@st.cache_resource(show_spinner="BERT modeli yükleniyor (ilk seferde 1-2 dk sürebilir)...")
def _warm_model():
    load_model, _ = _lazy_bert()
    return load_model()


@st.cache_data(show_spinner="Yorum verisi okunuyor...")
def load_reviews() -> pd.DataFrame:
    raw = PROJECT_ROOT / "data" / "raw"
    frames = []
    for name in ("app_store_reviews.csv", "google_play_reviews.csv"):
        p = raw / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame(
            columns=["review_id", "platform", "app_name", "rating", "title", "text", "full_text"]
        )
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["full_text"] = (df["title"] + " " + df["text"]).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    df["app_name"] = df["app_name"].fillna("Bilinmiyor")
    return df


def render_result(result: dict) -> None:
    sentiment = result["sentiment"]
    color = SENTIMENT_COLORS[sentiment]
    emoji = SENTIMENT_EMOJI[sentiment]

    st.markdown(
        f"<h2 style='color:{color}; margin-bottom:0'>{emoji} {sentiment.upper()}</h2>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Tahmini yıldız", f"{result['star']} / 5")
    col2.metric("Güven", f"{result['sentiment_confidence']:.1%}")

    star_df = pd.DataFrame(
        {"olasılık": result["star_probs"]},
        index=[f"{i}⭐" for i in range(1, 6)],
    )
    sent_df = pd.DataFrame(
        {"olasılık": result["sentiment_probs"]},
        index=SENTIMENT_CLASSES,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Yıldız olasılık dağılımı")
        st.bar_chart(star_df, height=240)
    with c2:
        st.caption("Sentiment olasılık dağılımı")
        st.bar_chart(sent_df, height=240)


def review_card(row: pd.Series) -> None:
    rating = int(row["rating"])
    stars = "⭐" * rating + "☆" * (5 - rating)
    platform = row.get("platform", "")
    text = row["full_text"]
    if len(text) > 400:
        text = text[:400] + "…"
    st.markdown(
        f"**{stars}**  \n*{platform}*  \n{text}",
    )
    st.divider()


def tab_predict() -> None:
    st.subheader("Yorum analizi")
    st.write(
        "Bir müşteri yorumu yazın, BERT modeli sentiment tahmini yapsın. "
        "Türkçe ve İngilizce yorumlar desteklenir."
    )

    default = "Uygulama çok güzel, çok seviyorum"
    text = st.text_area("Yorum metni:", value=default, height=120)
    if st.button("Analiz et", type="primary"):
        if not text.strip():
            st.warning("Lütfen bir yorum girin.")
            return
        try:
            _, predict_sentiment = _lazy_bert()
        except ModuleNotFoundError as exc:
            st.error(
                f"BERT için gerekli paket eksik: `{exc.name}`. "
                "Şu komutu çalıştırın: `pip install transformers torch sentencepiece`"
            )
            return
        with st.spinner("Tahmin yapılıyor (ilk seferde model iniyor, ~700MB)..."):
            result = predict_sentiment(text)
        render_result(result)


def tab_search(df: pd.DataFrame) -> None:
    st.subheader("Uygulama / firma araması")
    if df.empty:
        st.warning(
            "Yorum verisi bulunamadı. `data/raw/` altına "
            "`app_store_reviews.csv` veya `google_play_reviews.csv` ekleyin."
        )
        return

    apps = sorted(df["app_name"].dropna().unique().tolist())
    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input(
            "Uygulama adı ara (kısmi eşleşme):",
            placeholder="örn. youtube, instagram",
        )
    with col2:
        preset = st.selectbox(
            "veya listeden seç:",
            options=[""] + apps,
        )

    target = query.strip() or preset
    if not target:
        st.info("Yukarıdan bir uygulama yazın veya seçin.")
        st.caption(f"Toplam yorumlu uygulama: {len(apps)} — toplam yorum: {len(df):,}")
        return

    matches = df[df["app_name"].str.contains(target, case=False, na=False)]
    if matches.empty:
        st.warning(f"'{target}' için yorum bulunamadı.")
        return

    n_apps = matches["app_name"].nunique()
    st.success(f"Eşleşen yorum: **{len(matches):,}** | Eşleşen uygulama sayısı: {n_apps}")

    summary = (
        matches.groupby("app_name")
        .agg(yorum=("rating", "size"), ortalama=("rating", "mean"))
        .sort_values("yorum", ascending=False)
        .head(10)
    )
    summary["ortalama"] = summary["ortalama"].round(2)
    st.dataframe(summary, use_container_width=True)

    n = st.slider("Kaç adet?", min_value=3, max_value=15, value=5)

    left, right = st.columns(2)
    with left:
        st.markdown("### ✅ En iyi yorumlar")
        best = matches.sort_values(["rating", "full_text"], ascending=[False, True]).head(n)
        for _, row in best.iterrows():
            review_card(row)
    with right:
        st.markdown("### ❌ En kötü yorumlar")
        worst = matches.sort_values(["rating", "full_text"], ascending=[True, True]).head(n)
        for _, row in worst.iterrows():
            review_card(row)


def main() -> None:
    st.title("💬 Müşteri Yorumu Sentiment Analizi")
    st.caption(f"Model: `{MODEL_NAME}` — Veri Madenciliği Dersi Projesi")

    with st.sidebar:
        st.header("Hakkında")
        st.write(
            "Bu uygulama BERT tabanlı multilingual bir sentiment modeli kullanır "
            "ve App Store + Google Play'den toplanan yorumlar üzerinde çalışır."
        )
        if st.button("Modeli önceden yükle"):
            _warm_model()
            st.success("Model hazır.")

    df = load_reviews()

    tab1, tab2 = st.tabs(["🔮 Yorum analiz et", "🔎 Uygulama ara"])
    with tab1:
        tab_predict()
    with tab2:
        tab_search(df)


if __name__ == "__main__":
    main()
