import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# AYARLAR VE CACHE FONKSİYONLARI
# ==========================================
st.set_page_config(layout="wide", page_title="Müşteri Memnuniyeti Analizi", page_icon="📊")

@st.cache_data
def load_main_data():
    try:
        df = pd.read_csv('data/processed/reviews_with_predictions.csv')
        if 'tarih' in df.columns:
            df['tarih'] = pd.to_datetime(df['tarih'], errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_tfidf_data():
    try:
        with open('data/processed/tfidf_top_words.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def load_model_comparison_data():
    try:
        return pd.read_csv('data/processed/model_comparison.csv')
    except FileNotFoundError:
        return pd.DataFrame()

from hybrid_model import HybridEnsembleClassifier

@st.cache_resource
def load_hybrid_model():
    hybrid = HybridEnsembleClassifier()
    return hybrid

df = load_main_data()
tfidf_data = load_tfidf_data()
model_comp_df = load_model_comparison_data()

# Sidebar Navigation
st.sidebar.title("Menü")
page = st.sidebar.radio("Sayfalar", [
    "🔍 Canlı Analiz", 
    "🏢 Uygulama Arama", 
    "📊 Genel Dashboard", 
    "🤖 Model Karşılaştırma"
])

# ==========================================
# SAYFA 1: CANLI DUYGU ANALİZİ
# ==========================================
if page == "🔍 Canlı Analiz":
    st.title("🔍 Canlı Duygu Analizi")
    st.markdown("Yapay zeka (BERT) modelinin girdiğiniz yorumun duygu durumunu nasıl sınıflandırdığını görün.")
    
    user_input = st.text_area("Yorumunuzu girin:", placeholder="Yorumunuzu buraya yazın...")
    
    if st.button("Analiz Et"):
        if user_input.strip() == "":
            st.warning("Lütfen analiz edilecek bir metin girin.")
        else:
            with st.spinner("Analiz ediliyor..."):
                try:
                    hybrid = load_hybrid_model()
                    
                    # 1. Prediction using the Hybrid model
                    res = hybrid.predict_proba(user_input)
                    hybrid_probs = res["hybrid_probs"]
                    pred_idx = res["hybrid_pred_id"]
                    label = hybrid.id2label[pred_idx].lower()
                    confidence = hybrid_probs[pred_idx] * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if label == "positive":
                            st.success("🟢 Olumlu Yorum")
                        elif label == "neutral":
                            st.warning("🟡 Nötr Yorum")
                        else:
                            st.error("🔴 Olumsuz Yorum")
                    with col2:
                        st.metric("Güven Skoru", f"%{confidence:.1f}")
                        
                    # 2. Show sub-model probabilities for transparency
                    st.write("**Model Oylama Dağılımı:**")
                    import pandas as pd
                    dist_df = pd.DataFrame({
                        "Klasik (SVC)": res["svc_probs"] * 100,
                        "Derin Öğrenme (BERT)": res["bert_probs"] * 100,
                        "Nihai Hibrit Karar": hybrid_probs * 100
                    }, index=["Negatif", "Nötr", "Pozitif"])
                    st.dataframe(dist_df.style.format("{:.1f}%"))
                    
                    # 3. For attention maps, use the internal BERT model
                    inputs = hybrid.tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512).to(hybrid.device)
                    with torch.no_grad():
                        outputs = hybrid.bert(**inputs)
                    
                    attentions = outputs.attentions
                    last_layer_attention = attentions[-1].cpu()
                    cls_attention = last_layer_attention[0].mean(dim=0)[0].numpy()
                    
                    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    valid_indices = [i for i, t in enumerate(tokens) if t not in ['[CLS]', '[SEP]', '[PAD]']]
                    clean_tokens = [tokens[i].replace('##', '') for i in valid_indices]
                    clean_attentions = [cls_attention[i] for i in valid_indices]
                    
                    if len(clean_tokens) > 0:
                        sum_att = sum(clean_attentions)
                        norm_att = [a / sum_att for a in clean_attentions] if sum_att > 0 else clean_attentions
                        
                        fig = px.bar(
                            x=clean_tokens, 
                            y=norm_att,
                            labels={'x': 'Kelime', 'y': 'Etki (Attention Ağırlığı)'},
                            title="Hangi kelimeler etkili oldu?",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Tahmin sırasında bir hata oluştu: {str(e)}")

# ==========================================
# SAYFA 2: UYGULAMA ARAMA
# ==========================================
elif page == "🏢 Uygulama Arama":
    st.title("🏢 Uygulama Bazlı Arama ve Detay")
    st.markdown("Spesifik bir uygulamaya ait performansı ve kullanıcı geri bildirimlerini inceleyin.")
    
    if df.empty:
        st.warning("Veri bulunamadı. Lütfen önce veri hazırlama adımlarını tamamlayın.")
    else:
        col_filters, col_results = st.columns([1, 3])
        
        with col_filters:
            st.subheader("Filtreler")
            apps = sorted(df['uygulama'].dropna().unique().tolist())
            selected_app = st.selectbox("Uygulama Seçin:", apps)
            
            platform_filter = st.radio("Platform:", ["Tümü", "Google Play", "App Store"])
            min_score = st.slider("Minimum Puan:", 1, 5, 1)
            
        with col_results:
            filtered_df = df[(df['uygulama'] == selected_app) & (df['puan'] >= min_score)]
            if platform_filter == "Google Play":
                filtered_df = filtered_df[filtered_df['platform'] == 'google_play']
            elif platform_filter == "App Store":
                filtered_df = filtered_df[filtered_df['platform'] == 'app_store']
                
            if filtered_df.empty:
                st.info("Bu filtrelere uygun veri bulunamadı.")
            else:
                m1, m2, m3 = st.columns(3)
                total_reviews = len(filtered_df)
                avg_score = filtered_df['puan'].mean()
                
                pos_reviews = len(filtered_df[filtered_df['bert_label'] == 'positive'])
                sat_rate = (pos_reviews / total_reviews * 100) if total_reviews > 0 else 0
                
                m1.metric("Toplam Yorum Sayısı", f"{total_reviews:,}")
                m2.metric("Ortalama Puan", f"{avg_score:.1f} / 5")
                m3.metric("Memnuniyet Oranı", f"%{sat_rate:.1f}")
                
                st.markdown("---")
                
                c1, c2 = st.columns(2)
                
                def get_top_reviews(data, target_score, n=5):
                    sub = data[data['puan'] == target_score].copy()
                    if not sub.empty:
                        sub['len'] = sub['cleaned_text'].astype(str).apply(len)
                        return sub.sort_values(by='len', ascending=False).head(n)
                    return pd.DataFrame()

                with c1:
                    st.subheader("⭐ En İyi 5 Yorum")
                    best_reviews = get_top_reviews(filtered_df, 5)
                    for _, row in best_reviews.iterrows():
                        date_str = row['tarih'].strftime('%Y-%m-%d') if pd.notnull(row['tarih']) else 'Tarih Yok'
                        st.info(f"{row['yorum']}\n\n**⭐⭐⭐⭐⭐ | {row['platform']} | {date_str}**")
                        
                with c2:
                    st.subheader("👎 En Kötü 5 Yorum")
                    worst_reviews = get_top_reviews(filtered_df, 1)
                    for _, row in worst_reviews.iterrows():
                        date_str = row['tarih'].strftime('%Y-%m-%d') if pd.notnull(row['tarih']) else 'Tarih Yok'
                        st.error(f"{row['yorum']}\n\n**⭐ | {row['platform']} | {date_str}**")
                        
                st.markdown("---")
                
                p1, p2 = st.columns(2)
                
                with p1:
                    score_dist = filtered_df['puan'].value_counts().reset_index()
                    score_dist.columns = ['Puan', 'Yorum Sayısı']
                    score_dist = score_dist.sort_values(by='Puan')
                    
                    fig_dist = px.bar(score_dist, x='Puan', y='Yorum Sayısı', text='Yorum Sayısı',
                                      title="Puan Dağılımı", template="plotly_white")
                    fig_dist.update_traces(textposition='outside')
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                with p2:
                    if 'tarih' in filtered_df.columns and not filtered_df['tarih'].isnull().all():
                        time_df = filtered_df.dropna(subset=['tarih']).copy()
                        time_df['Ay'] = time_df['tarih'].dt.to_period('M').astype(str)
                        monthly_avg = time_df.groupby('Ay')['puan'].mean().reset_index()
                        
                        fig_time = px.line(monthly_avg, x='Ay', y='puan', markers=True,
                                           title="Aylık Ortalama Puan Eğilimi", template="plotly_white")
                        fig_time.update_yaxes(range=[0, 5.5])
                        st.plotly_chart(fig_time, use_container_width=True)
                    else:
                        st.info("Zaman serisi grafiği için geçerli tarih verisi bulunamadı.")

# ==========================================
# SAYFA 3: GENEL DASHBOARD
# ==========================================
elif page == "📊 Genel Dashboard":
    st.title("📊 Genel Dashboard")
    
    if df.empty:
        st.warning("Veri bulunamadı. Lütfen analiz sürecinin tamamlandığından emin olun.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        
        total_rev = len(df)
        total_app = df['uygulama'].nunique()
        gen_avg = df['puan'].mean()
        gen_sat = (len(df[df['bert_label'] == 'positive']) / total_rev * 100) if total_rev > 0 else 0
        
        k1.metric("Toplam Yorum Sayısı", f"{total_rev:,}")
        k2.metric("Toplam Uygulama Sayısı", f"{total_app}")
        k3.metric("Genel Ortalama Puan", f"{gen_avg:.2f} / 5")
        k4.metric("Genel Memnuniyet Oranı", f"%{gen_sat:.1f}")
        
        st.markdown("---")
        
        g1, g2 = st.columns(2)
        
        with g1:
            app_stats = df.groupby('uygulama').apply(
                lambda x: pd.Series({
                    'Memnuniyet Oranı': (len(x[x['bert_label'] == 'positive']) / len(x) * 100),
                    'Yorum Sayısı': len(x)
                })
            ).reset_index()
            app_stats = app_stats[app_stats['Yorum Sayısı'] > 20]
            top10_apps = app_stats.sort_values('Memnuniyet Oranı', ascending=False).head(10)
            
            fig_top10 = px.bar(top10_apps, x='Memnuniyet Oranı', y='uygulama', orientation='h',
                               title="En Memnun 10 Uygulama", template="plotly_white",
                               color='Memnuniyet Oranı', color_continuous_scale="Greens")
            fig_top10.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top10, use_container_width=True)
            
        with g2:
            plat_stats = df.groupby('platform').apply(
                lambda x: pd.Series({
                    'Ortalama Puan': x['puan'].mean(),
                    'Memnuniyet Oranı': (len(x[x['bert_label'] == 'positive']) / len(x) * 100)
                })
            ).reset_index()
            
            fig_plat = go.Figure()
            fig_plat.add_trace(go.Bar(x=plat_stats['platform'], y=plat_stats['Ortalama Puan'], name="Ort. Puan", yaxis='y1'))
            fig_plat.add_trace(go.Bar(x=plat_stats['platform'], y=plat_stats['Memnuniyet Oranı'], name="Memnuniyet %", yaxis='y2'))
            
            fig_plat.update_layout(
                title="Google Play vs App Store Karşılaştırması",
                barmode='group',
                yaxis=dict(title="Puan (0-5)", range=[0, 5]),
                yaxis2=dict(title="Yüzde (%)", overlaying="y", side="right", range=[0, 100]),
                template="plotly_white"
            )
            st.plotly_chart(fig_plat, use_container_width=True)
            
        st.markdown("---")
        
        if tfidf_data:
            st.subheader("En Çok Geçen Kelimeler (TF-IDF Top 20)")
            t1, t2, t3 = st.columns(3)
            with t1:
                pos_tfidf = pd.DataFrame(tfidf_data.get('positive', [])[:20])
                if not pos_tfidf.empty:
                    fig_pos_tfidf = px.bar(pos_tfidf, x='score', y='word', orientation='h',
                                           title="Olumlu Yorumlarda", template="plotly_white", color_discrete_sequence=['#2ecc71'])
                    fig_pos_tfidf.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0), height=400)
                    st.plotly_chart(fig_pos_tfidf, use_container_width=True)
            with t2:
                neu_tfidf = pd.DataFrame(tfidf_data.get('neutral', [])[:20])
                if not neu_tfidf.empty:
                    fig_neu_tfidf = px.bar(neu_tfidf, x='score', y='word', orientation='h',
                                           title="Nötr Yorumlarda", template="plotly_white", color_discrete_sequence=['#f1c40f'])
                    fig_neu_tfidf.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0), height=400)
                    st.plotly_chart(fig_neu_tfidf, use_container_width=True)
            with t3:
                neg_tfidf = pd.DataFrame(tfidf_data.get('negative', [])[:20])
                if not neg_tfidf.empty:
                    fig_neg_tfidf = px.bar(neg_tfidf, x='score', y='word', orientation='h',
                                           title="Olumsuz Yorumlarda", template="plotly_white", color_discrete_sequence=['#e74c3c'])
                    fig_neg_tfidf.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0), height=400)
                    st.plotly_chart(fig_neg_tfidf, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Kelime Bulutları")
        w1, w2, w3 = st.columns(3)
        
        with w1:
            st.markdown("**Olumlu Yorumlarda**")
            try:
                st.image('visuals/wordcloud_positive.png', use_container_width=True)
            except FileNotFoundError:
                st.warning("wordcloud_positive.png bulunamadı.")
                
        with w2:
            st.markdown("**Nötr Yorumlarda**")
            try:
                st.image('visuals/wordcloud_neutral.png', use_container_width=True)
            except FileNotFoundError:
                st.warning("wordcloud_neutral.png bulunamadı.")
                
        with w3:
            st.markdown("**Olumsuz Yorumlarda**")
            try:
                st.image('visuals/wordcloud_negative.png', use_container_width=True)
            except FileNotFoundError:
                st.warning("wordcloud_negative.png bulunamadı.")
                
        st.markdown("---")
        
        st.subheader("Genel Duygu Dağılımı")
        l1, l2 = st.columns([1, 2])
        with l1:
            label_counts = df['label'].value_counts().reset_index()
            label_counts.columns = ['Duygu', 'Sayı']
            fig_pie = px.pie(label_counts, names='Duygu', values='Sayı', hole=0.4,
                             color='Duygu', color_discrete_map={'positive':'#2ecc71', 'neutral':'#f1c40f', 'negative':'#e74c3c'})
            st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# SAYFA 4: MODEL KARŞILAŞTIRMA
# ==========================================
elif page == "🤖 Model Karşılaştırma":
    st.title("🤖 Model Performans Karşılaştırması")
    st.markdown("Bu projede kullanılan tüm modellerin (Klasik Baseline vs. Derin Öğrenme BERT) performans karşılaştırması.")
    
    if model_comp_df.empty:
        st.warning("Model karşılaştırma verisi bulunamadı (data/processed/model_comparison.csv). Lütfen notebook adımını tamamlayın.")
    else:
        df_comp = model_comp_df.sort_values(by='F1', ascending=False)
        st.dataframe(df_comp, use_container_width=True)
        
        c1, c2 = st.columns(2)
        
        with c1:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
            fig_bar = go.Figure()
            for metric in metrics:
                if metric in df_comp.columns:
                    fig_bar.add_trace(go.Bar(
                        x=df_comp['Model'],
                        y=df_comp[metric],
                        name=metric
                    ))
            fig_bar.update_layout(
                title="Model Performans Karşılaştırması",
                barmode='group',
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            fig_radar = go.Figure()
            for i, row in df_comp.iterrows():
                r_vals = [row.get('Accuracy', 0), row.get('Precision', 0), row.get('Recall', 0), row.get('F1', 0)]
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_vals + [r_vals[0]],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1', 'Accuracy'],
                    fill='toself',
                    name=row['Model'],
                    opacity=0.6
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Model Profil Karşılaştırması",
                template="plotly_white"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        st.info("Bu projede canlı tahmin için BERT (savasy/bert-base-turkish-sentiment-clas) kullanılmaktadır. "
                "Hazır Türkçe dil modelidir, fine-tune gerektirmez.")

# requirements.txt içeriği:
# streamlit
# pandas
# plotly
# transformers
# torch
# Pillow
# wordcloud
