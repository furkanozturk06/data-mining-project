<div align="center">

# Müşteri Geribildirimlerine Dayalı Memnuniyet Analizi

**Veri Madenciliği Dersi — Dönem Projesi**

Türkçe mobil uygulama yorumları üzerinde duygu analizi · App Store & Google Play · Klasik ML, Derin Öğrenme, BERT ve LLM karşılaştırması

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![Groq](https://img.shields.io/badge/Groq-Llama%203-black)

### 🌐 Canlı Demo

[![Canlı Demo](https://img.shields.io/badge/🌐_Uygulamayı_Aç-Streamlit_Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://musteri-memnuniyet-data-mining.streamlit.app/)

**[musteri-memnuniyet-data-mining.streamlit.app](https://musteri-memnuniyet-data-mining.streamlit.app/)** — kurulum gerektirmez, tarayıcıdan doğrudan kullanılabilir.

</div>

---

## Projenin Amacı

Bu projenin amacı, App Store ve Google Play'den toplanan Türkçe müşteri yorumları üzerinde doğal dil işleme ve makine öğrenmesi teknikleri kullanarak **otomatik memnuniyet sınıflandırması** (pozitif / nötr / negatif) yapmak ve farklı model ailelerini (klasik ML, derin öğrenme, transformer, büyük dil modeli) karşılaştırmaktır.

## Araştırma Sorusu

> Müşteri yorumlarındaki metin verisi kullanılarak memnuniyet düzeyi (pozitif / nötr / negatif) ne düzeyde doğrulukla tahmin edilebilir? Farklı model aileleri bu görevde nasıl performans gösterir?

## Neden Bu Problem?

Kullanıcılar satın alma ve indirme kararlarını büyük ölçüde diğer kullanıcıların yorumlarına göre verir. Ancak on binlerce yorumun manuel okunması pratik değildir. Otomatik duygu analizi, işletmelerin müşteri memnuniyetini hızlı ve ölçeklenebilir şekilde takip etmesine olanak tanır.

---

## Veri Seti

Veriler iki büyük mobil uygulama mağazasından, ülke `tr` filtresiyle web scraping yöntemiyle toplanmış; ardından temizlenip etiketlenmiştir.

| Özellik | Değer |
|---------|-------|
| **Toplam yorum (temiz)** | **69.002** |
| Google Play | 49.699 |
| App Store | 19.303 |
| Benzersiz uygulama | 55 |
| Kategori | 6 (Sosyal Medya, E-Ticaret, Yemek, Süpermarket, Kariyer, Bankacılık) |
| Ortalama puan | 2.95 / 5 |

**Sınıf dağılımı (etiketli veri seti):**

| Etiket | Yorum Sayısı |
|--------|--------------|
| Negatif | 30.278 |
| Pozitif | 27.760 |
| Nötr | 10.964 |

**Kolonlar:** `review_id`, `platform`, `app_name`, `author`, `rating`, `title`, `text`, `date`, `sentiment_label`, `groq_labeled`

---

## Etiketleme Stratejisi (Hibrit)

Etiketler iki aşamalı bir yaklaşımla oluşturulmuştur:

1. **Yıldız kuralı (baseline):** 1–2 yıldız → negatif, 3 yıldız → nötr, 4–5 yıldız → pozitif.
2. **LLM ile düzeltme:** Metin ile yıldız puanının **çeliştiği** yorumlar (örn. 1 yıldız ama metin olumlu) tespit edilip Groq **Llama 3.1 8B** ile yeniden etiketlenir. Bu sayede ironi, yanlış puanlama ve karışık yorumlar düzeltilir.

> Toplam **27.955** yorum Groq ile yeniden etiketlenmiştir (`groq_labeled = True`).

---

## Modeller ve Sonuçlar

Veri seti %80 eğitim / %20 test olarak ayrılmış (`stratify`, `random_state=42`) ve dört temel model eğitilmiştir. Ayrıca büyük dil modeli (LLM) zero-shot olarak değerlendirilmiş, canlı demoda ise önceden eğitilmiş bir Türkçe BERT modeli kullanılmıştır.

### Eğitilen Modeller (test seti karşılaştırması)

| Model | Aile | Accuracy | Precision | Recall | F1-Score |
|-------|------|:--------:|:---------:|:------:|:--------:|
| **Logistic Regression**  | Klasik ML (TF-IDF) | **%68.87** | %66.00 | %68.87 | **%66.91** |
| SVM (LinearSVC) | Klasik ML (TF-IDF) | %68.42 | %65.47 | %68.42 | %66.42 |
| TextCNN | Derin Öğrenme | %67.36 | %64.06 | %67.36 | %65.11 |
| Bi-LSTM + Attention | Derin Öğrenme | %65.23 | %63.56 | %65.23 | %64.29 |

> **En başarılı model:** TF-IDF + Logistic Regression. Klasik ML yöntemleri, bu veri seti ve özellik temsilinde derin öğrenme modellerini geride bırakmıştır.

### Ek Değerlendirmeler

| Model | Yöntem | Sonuç |
|-------|--------|-------|
| **Llama 3.3 70B** (Groq) | Zero-shot (50 örneklik test) | %82 doğruluk |
| **BERT** (`savasy/bert-base-turkish-sentiment-cased`) | Önceden eğitilmiş transformer | Canlı demo + kelime (attention) ağırlıkları |

---

## Detaylı Model Açıklamaları

- **Logistic Regression / SVM** — TF-IDF (1–3 gram, `max_features=50.000`) vektörleri üzerinde eğitilen klasik ML modelleri. Hızlı, hafif ve bu görevde en yüksek skoru veren yaklaşım.
- **TextCNN** — Yoon Kim (2014) mimarisi. Word embedding üzerinde farklı boyutlarda (2,3,4,5) 1D konvolüsyon filtreleriyle n-gram kalıplarını yakalar.
- **Bi-LSTM + Attention** — Cümleyi iki yönden okuyan LSTM ve hangi kelimelerin duyguyu belirlediğine odaklanan additive (Bahdanau-tarzı) attention mekanizması.
- **BERT** — `savasy/bert-base-turkish-sentiment-cased`; canlı tahmin yapar ve modelin yorumdaki hangi kelimelere dikkat ettiğini attention ağırlıklarıyla görselleştirir.
- **LLM (Llama 3)** — Groq API üzerinden hiç eğitim almadan (zero-shot) yorumları sınıflandırır.

---

## İnteraktif Arayüz (Streamlit)

> 🌐 **Canlı uygulama:** [musteri-memnuniyet-data-mining.streamlit.app](https://musteri-memnuniyet-data-mining.streamlit.app/) — Streamlit Community Cloud üzerinde yayında.

Proje, Streamlit tabanlı 4 sekmeli bir web uygulaması içerir:

| Sekme | İşlev |
|-------|-------|
| **Duygu Testi** | Girilen yorumu 6 farklı modelden (BERT, LLM, LogReg, SVM, TextCNN, Bi-LSTM) biriyle analiz eder; BERT attention kelime ağırlıklarını grafikle gösterir. |
| **Firma Arama** | Bir uygulamayı seçip en iyi/en kötü yorumlarını listeler ve Groq ile yapay zekâ destekli özet üretir. |
| **Kelime Bulutu** | Olumlu / nötr / olumsuz yorumlardaki öne çıkan kelimeleri TF-IDF tabanlı kelime bulutlarıyla görselleştirir. |
| **Kelime Analizi** | Duygu sınıflarına göre en yüksek TF-IDF skorlu kelimeleri çubuk grafiklerle ve özet istatistiklerle sunar. |

---

## Proje Raporu ve Ekran Görüntüleri

- **Final Raporu (IEEE formatında):** [rapor.pdf](rapor.pdf) · [rapor.docx](rapor.docx)
- **Arayüz ve analiz ekran görüntüleri:** [`görseller/`](görseller/) klasöründe yer alır — Duygu Testi, Firma Arama, Kelime Bulutu ve Kelime Analizi sekmelerinin gerçek görünümleri ile sınıf bazında kelime bulutu ve TF-IDF analizleri.

---

## Proje Yapısı

```
musteri-memnuniyet-analizi/
│
├── data/
│   ├── raw/                          # Ham scraping çıktıları
│   │   ├── app_store_reviews.csv
│   │   └── google_play_reviews.csv
│   └── processed/                    # Temizlenmiş + etiketlenmiş veri
│       ├── all_reviews_clean.csv
│       ├── all_reviews_labeled.csv
│       ├── app_store_reviews_clean.csv
│       └── google_play_reviews_clean.csv
│
├── models/                           # Eğitilmiş modeller + sonuç JSON'ları
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── textcnn.pt
│   ├── bilstm_attention.pt
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── vocabulary.pkl
│   ├── training_results.json
│   ├── ml_results.json
│   └── llm_results.json
│
├── src/
│   ├── config.py                     # GROQ_API_KEY yükleme (.env)
│   ├── scraping/
│   │   ├── app_config.py             # 55 uygulama, 6 kategori
│   │   ├── google_play.py            # google-play-scraper
│   │   ├── app_store.py              # app-store-scraper
│   │   └── run_all.py                # Tüm scraping akışı
│   ├── preprocessing/
│   │   ├── clean_reviews.py          # Temizleme, normalize, deduplikasyon
│   │   └── groq_label_reviews.py     # Çelişkili yorumları Groq ile etiketleme
│   ├── modeling/
│   │   ├── train_baseline.py         # 4 modelin eğitimi ve karşılaştırması
│   │   └── evaluate_llm.py           # Llama 3 zero-shot değerlendirme
│   ├── nlp/
│   │   ├── sentiment.py              # BERT tahmini + attention ağırlıkları
│   │   ├── inference.py              # Tüm modeller için ortak tahmin katmanı
│   │   ├── word_analysis.py          # TF-IDF + kelime bulutu
│   │   └── groq_summary.py           # Groq ile özet ve tahmin
│   └── ui/
│       └── app.py                    # Streamlit arayüzü
│
├── görseller/                        # Arayüz ve analiz ekran görüntüleri
├── rapor.docx                        # Final raporu (IEEE formatı)
├── rapor.pdf                         # Final raporu (PDF)
├── requirements.txt
├── Veri_Madenciligi_Ön_Sunum.pptx
└── README.md
```

---

## Teknolojiler

| Kategori | Araçlar |
|----------|---------|
| Dil | Python |
| Veri Toplama | google-play-scraper, app-store-scraper |
| Veri İşleme | Pandas, NumPy |
| Klasik ML | scikit-learn (TF-IDF, Logistic Regression, LinearSVC) |
| Derin Öğrenme | PyTorch (TextCNN, Bi-LSTM + Attention) |
| Transformer | Hugging Face Transformers (Türkçe BERT) |
| LLM | Groq API (Llama 3.1 8B, Llama 3.3 70B) |
| Arayüz | Streamlit, Plotly |
| Görselleştirme | Matplotlib, WordCloud |
| Ortam Değişkenleri | python-dotenv |

---

## Kurulum

```bash
git clone https://github.com/furkanozturk06/musteri-memnuniyet-analizi.git
cd musteri-memnuniyet-analizi
pip install -r requirements.txt
```

**Groq özelliklerini (LLM analizi, AI özet, yeniden etiketleme) kullanmak için** proje kök dizinine bir `.env` dosyası ekleyin:

```env
GROQ_API_KEY=your_api_key_here
```

> Groq API anahtarı yalnızca LLM tabanlı sekmeler ve etiketleme için gereklidir; klasik ML, derin öğrenme ve BERT analizleri anahtarsız çalışır.

---

## Kullanım

```bash
# 1) (Opsiyonel) Yorumları yeniden topla
python -m src.scraping.run_all                  # her iki platform, tüm uygulamalar
python -m src.scraping.run_all --platform gp    # sadece Google Play
python -m src.scraping.run_all --app "Akbank"   # tek uygulama

# 2) Veriyi temizle
python -m src.preprocessing.clean_reviews

# 3) Çelişkili yorumları Groq ile etiketle
python src/preprocessing/groq_label_reviews.py

# 4) Modelleri eğit ve karşılaştır
python -m src.modeling.train_baseline

# 5) LLM'i zero-shot değerlendir
python -m src.modeling.evaluate_llm

# 6) İnteraktif arayüzü başlat
streamlit run src/ui/app.py
```

> Repo, önceden toplanmış veriyi ve eğitilmiş modelleri içerdiğinden, doğrudan **6. adımla** arayüzü çalıştırarak projeyi inceleyebilirsiniz.

---

## Proje Aşamaları

| Aşama | Durum |
|-------|-------|
| Problem tanımı ve planlama | Tamamlandı |
| Veri toplama (Web Scraping) | Tamamlandı |
| Veri temizleme ve ön işleme | Tamamlandı |
| Hibrit etiketleme (yıldız + LLM) | Tamamlandı |
| Modelleme (4 model) | Tamamlandı |
| LLM ve BERT değerlendirmesi | Tamamlandı |
| Görselleştirme ve arayüz | Tamamlandı |
| Final raporu ve sunum | Tamamlandı |

---

## Ekip

| Ad Soyad | Öğrenci No |
|----------|------------|
| Furkan Öztürk | 230229083 |
| Taha Yasin Çiçek | 230229088 |
| Ziyaeddin Ayerden | 210229022 |
