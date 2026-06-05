"""Streamlit UI for Turkish review sentiment analysis."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.nlp.sentiment import predict, get_word_weights
from src.nlp.word_analysis import (
    get_tfidf_top_words,
    generate_wordcloud_figure,
    get_top_bottom_reviews,
    get_available_apps,
)
from src.nlp.groq_summary import generate_app_summary, predict_sentiment
from src.config import GROQ_API_KEY

st.set_page_config(
    page_title="Yorum Duygu Analizi",
    page_icon="🔍",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
LABELED_CSV = os.path.join(PROCESSED_DIR, "all_reviews_labeled.csv")
CLEAN_CSV = os.path.join(PROCESSED_DIR, "all_reviews_clean.csv")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, bool]:
    """Load reviews. Returns (df, is_groq_labeled)."""
    if os.path.exists(LABELED_CSV):
        df = pd.read_csv(LABELED_CSV)
        is_labeled = "sentiment_label" in df.columns
    elif os.path.exists(CLEAN_CSV):
        df = pd.read_csv(CLEAN_CSV)
        is_labeled = False
    else:
        dfs = []
        for fname in ("app_store_reviews.csv", "google_play_reviews.csv"):
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        if not dfs:
            return pd.DataFrame(), False
        df = pd.concat(dfs, ignore_index=True)
        is_labeled = False

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)
    return df, is_labeled


@st.cache_data(ttl=3600)
def cached_app_summary(app_name: str, review_count: int) -> str:
    """Cache per-app Groq summaries for 1 hour."""
    df, _ = load_data()
    return generate_app_summary(df, app_name)


def render_sentiment_badge(label: str, pos: float, neg: float):
    is_positive = label == "positive"
    color = "#27ae60" if is_positive else "#e74c3c"
    emoji = "✅ Olumlu" if is_positive else "❌ Olumsuz"
    st.markdown(
        f"""
        <div style="background:{color};padding:16px 24px;border-radius:10px;text-align:center;margin:10px 0">
            <span style="color:white;font-size:1.6rem;font-weight:700">{emoji}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    col1.metric("Olumlu Skoru", f"{pos:.1%}")
    col2.metric("Olumsuz Skoru", f"{neg:.1%}")


def render_word_weights_chart(weights: list):
    if not weights:
        return
    top = weights[:15]
    words = [w for w, _ in top]
    scores = [s for _, s in top]
    colors = [f"rgba(39,174,96,{0.4 + s * 0.6})" for s in scores]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=words,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Kelime Ağırlıkları (BERT Attention)",
        xaxis_title="Ağırlık",
        yaxis={"categoryorder": "total ascending"},
        height=420,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

df, is_groq_labeled = load_data()

st.title("Yorum Duygu Analizi")
if is_groq_labeled:
    st.caption("BERT tabanlı Türkçe duygu analizi · App Store & Google Play verileri · Groq etiketleri aktif ✓")
else:
    st.caption("BERT tabanlı Türkçe duygu analizi · App Store & Google Play verileri")

tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Duygu Testi",
    "🏢 Firma Arama",
    "☁️ Kelime Bulutu",
    "📊 Kelime Analizi",
])

# ── Tab 1: Duygu Testi ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Yorumunuzu Analiz Edin")
    user_input = st.text_area(
        "Yorumunuzu buraya yazın:",
        placeholder="Örnek: Bu uygulama çok güzel, her şey mükemmel çalışıyor!",
        height=120,
    )
    if st.button("Analiz Et", type="primary", disabled=not user_input.strip()):
        with st.spinner("Analiz ediliyor..."):
            result = predict_sentiment(user_input)
            weights = get_word_weights(user_input)

        render_sentiment_badge(result["label"], result["positive"], result["negative"])

        st.divider()
        st.subheader("Kelime Ağırlıkları")
        st.caption("BERT'in yorumu değerlendirirken hangi kelimelere ne kadar dikkat ettiği")
        render_word_weights_chart(weights)

        with st.expander("Tüm kelime ağırlıklarını gör"):
            for word, score in weights:
                st.write(f"`{word}` → {score:.4f}")

# ── Tab 2: Firma Arama ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Uygulama / Firma Arama")
    if df.empty:
        st.warning("Veri yüklenemedi.")
    else:
        apps = get_available_apps(df)
        selected_app = st.selectbox("Uygulama seçin:", apps)
        n_reviews = st.slider("Kaç yorum listelensin?", 3, 10, 5)

        col_btn, col_summary_btn = st.columns([1, 1])
        get_reviews = col_btn.button("Yorumları Getir", type="primary")
        get_summary = col_summary_btn.button(
            "AI Özeti Oluştur",
            type="secondary",
            disabled=not GROQ_API_KEY,
            help="GROQ_API_KEY gerekli" if not GROQ_API_KEY else "Groq ile uygulama özeti oluştur",
        )

        # Session state ile her iki sonucu da sakla
        if "tab2_summary" not in st.session_state:
            st.session_state.tab2_summary = None
        if "tab2_summary_app" not in st.session_state:
            st.session_state.tab2_summary_app = None
        if "tab2_reviews" not in st.session_state:
            st.session_state.tab2_reviews = None
        if "tab2_reviews_app" not in st.session_state:
            st.session_state.tab2_reviews_app = None

        if get_summary:
            with st.spinner(f"{selected_app} özeti oluşturuluyor..."):
                try:
                    app_subset = df[df["app_name"].str.lower() == selected_app.strip().lower()]
                    st.session_state.tab2_summary = cached_app_summary(selected_app, len(app_subset))
                    st.session_state.tab2_summary_app = selected_app
                except Exception as e:
                    st.session_state.tab2_summary = f"HATA: {e}"
                    st.session_state.tab2_summary_app = selected_app

        if get_reviews:
            with st.spinner("Yorumlar yükleniyor..."):
                st.session_state.tab2_reviews = get_top_bottom_reviews(df, selected_app, n_reviews)
                st.session_state.tab2_reviews_app = selected_app

        # Özet varsa göster
        if st.session_state.tab2_summary:
            if st.session_state.tab2_summary.startswith("HATA:"):
                st.error(st.session_state.tab2_summary)
            else:
                st.info(f"**AI Özeti ({st.session_state.tab2_summary_app}):** {st.session_state.tab2_summary}")

        # Yorumlar varsa göster
        if st.session_state.tab2_reviews:
            result = st.session_state.tab2_reviews
            app_label = st.session_state.tab2_reviews_app
            st.info(f"**{app_label}** için toplam **{result['total']:,}** yorum bulundu.")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ⭐ En İyi Yorumlar")
                for r in result["top"]:
                    with st.container(border=True):
                        st.markdown(f"**{'⭐' * int(r['rating'])}** ({r['rating']}/5) — *{r['platform']}*")
                        st.write(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))
                        st.caption(f"{r['author']} · {str(r['date'])[:10]}")

            with col2:
                st.markdown("### 💔 En Kötü Yorumlar")
                for r in result["bottom"]:
                    with st.container(border=True):
                        stars = "⭐" * max(int(r["rating"]), 1)
                        st.markdown(f"**{stars}** ({r['rating']}/5) — *{r['platform']}*")
                        st.write(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))
                        st.caption(f"{r['author']} · {str(r['date'])[:10]}")

# ── Tab 3: Kelime Bulutu ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Kelime Bulutu")
    if df.empty:
        st.warning("Veri yüklenemedi.")
    else:
        apps_wc = ["Tüm Uygulamalar"] + get_available_apps(df)
        selected_app_wc = st.selectbox("Uygulama (boş bırakın = hepsi):", apps_wc, key="wc_app")

        if st.button("Kelime Bulutu Oluştur", type="primary"):
            with st.spinner("Kelime bulutları oluşturuluyor..."):
                subset = df if selected_app_wc == "Tüm Uygulamalar" else df[df["app_name"] == selected_app_wc]
                top_words = get_tfidf_top_words(subset, n=80)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### Olumlu Yorumların Kelimeleri")
                if top_words.get("positive"):
                    img = generate_wordcloud_figure(top_words["positive"], "Olumlu Kelimeler", "Greens")
                    st.image(img, use_container_width=True)
                else:
                    st.info("Olumlu yorum bulunamadı.")

            with col2:
                st.markdown("### Nötr Yorumların Kelimeleri")
                if top_words.get("neutral"):
                    img = generate_wordcloud_figure(top_words["neutral"], "Nötr Kelimeler", "Blues")
                    st.image(img, use_container_width=True)
                else:
                    st.info("Nötr yorum bulunamadı.")

            with col3:
                st.markdown("### Olumsuz Yorumların Kelimeleri")
                if top_words.get("negative"):
                    img = generate_wordcloud_figure(top_words["negative"], "Olumsuz Kelimeler", "Reds")
                    st.image(img, use_container_width=True)
                else:
                    st.info("Olumsuz yorum bulunamadı.")

# ── Tab 4: Kelime Analizi ─────────────────────────────────────────────────────
with tab4:
    st.subheader("TF-IDF Kelime Ağırlık Analizi")
    if df.empty:
        st.warning("Veri yüklenemedi.")
    else:
        apps_tf = ["Tüm Uygulamalar"] + get_available_apps(df)
        selected_app_tf = st.selectbox("Uygulama:", apps_tf, key="tf_app")
        top_n = st.slider("Kaç kelime gösterilsin?", 5, 30, 15, key="tf_n")

        if st.button("Analizi Başlat", type="primary"):
            with st.spinner("TF-IDF hesaplanıyor..."):
                subset = df if selected_app_tf == "Tüm Uygulamalar" else df[df["app_name"] == selected_app_tf]
                top_words = get_tfidf_top_words(subset, n=top_n)

            col1, col2, col3 = st.columns(3)

            for col, label, color, title in [
                (col1, "positive", "greens", "Olumlu Yorumlarda Öne Çıkan Kelimeler"),
                (col2, "neutral", "blues", "Nötr Yorumlarda Öne Çıkan Kelimeler"),
                (col3, "negative", "reds", "Olumsuz Yorumlarda Öne Çıkan Kelimeler"),
            ]:
                with col:
                    words_data = top_words.get(label, [])
                    if not words_data:
                        st.info("Yeterli veri yok.")
                        continue
                    words = [w for w, _ in words_data]
                    scores = [s for _, s in words_data]
                    fig = px.bar(
                        x=scores,
                        y=words,
                        orientation="h",
                        title=title,
                        labels={"x": "TF-IDF Skoru", "y": "Kelime"},
                        color=scores,
                        color_continuous_scale=color,
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        showlegend=False,
                        height=500,
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Veri İstatistikleri")
            c1, c2, c3, c4, c5 = st.columns(5)
            subset_stat = df if selected_app_tf == "Tüm Uygulamalar" else df[df["app_name"] == selected_app_tf]
            c1.metric("Toplam Yorum", f"{len(subset_stat):,}")
            c2.metric("Olumlu (4-5 ⭐)", f"{len(subset_stat[subset_stat['rating'] >= 4]):,}")
            c3.metric("Nötr (3 ⭐)", f"{len(subset_stat[subset_stat['rating'] == 3]):,}")
            c4.metric("Olumsuz (1-2 ⭐)", f"{len(subset_stat[subset_stat['rating'] <= 2]):,}")
            c5.metric("Ortalama Puan", f"{subset_stat['rating'].mean():.2f}")
