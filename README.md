<div align="center">

# Mobil Uygulama Yorumlarında Müşteri Memnuniyet Analizi

**Veri Madenciliği Dersi — Dönem Projesi**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![BERT](https://img.shields.io/badge/Model-BERT%20Multilingual-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Data](https://img.shields.io/badge/Veri-71K%2B%20Yorum-green)

</div>

---

## Proje Açıklaması

Bu proje, Google Play Store ve Apple App Store'dan toplanan **71.000+** Türkçe müşteri yorumu üzerinde doğal dil işleme (NLP) ve makine öğrenmesi teknikleri kullanarak otomatik duygu sınıflandırması yapmaktadır. 6 farklı kategorideki 55 mobil uygulamadan web scraping yöntemiyle toplanan veriler üzerinde kapsamlı analiz gerçekleştirilmiştir.

### Araştırma Sorusu

> Müşteri yorumlarındaki metin verisi kullanılarak memnuniyet düzeyi (pozitif / negatif / nötr) ne düzeyde doğrulukla tahmin edilebilir?

### Neden Bu Problem?

Günümüzde müşteriler satın alma kararlarını büyük ölçüde diğer kullanıcıların yorumlarına dayanarak vermektedir. Ancak bu yorumların manuel olarak okunup değerlendirilmesi büyük veri hacimlerinde pratik değildir. Otomatik duygu analizi, işletmelerin müşteri memnuniyetini hızlı ve ölçeklenebilir şekilde takip etmesine olanak tanır.

---

## Temel Özellikler

- **Çoklu Platform Veri Toplama:** Google Play ve App Store'dan eş zamanlı veri toplama
- **Dengeli Veri Seti:** Her yıldız sınıfında kontrollü örnekleme ile dengeli dağılım
- **BERT Derin Öğrenme:** Çok dilli BERT modeli ile yüksek doğruluklu duygu tahmini
- **Klasik ML Karşılaştırması:** 5 farklı makine öğrenmesi modelinin karşılaştırmalı analizi (Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost)
- **TF-IDF Analizi:** Pozitif ve negatif yorumlarda öne çıkan kelimelerin tespiti
- **Konu Modelleme (LDA):** Müşteri yorumlarındaki gizli temaların ortaya çıkarılması
- **LIME Yorumlanabilirlik:** Modelin karar mekanizmasının şeffaf hale getirilmesi
- **İnteraktif Dashboard:** Streamlit ile gerçek zamanlı duygu tahmini arayüzü
- **Kapsamlı Görselleştirme:** 20+ grafik ve kelime bulutları ile veri görselleştirme

---

## Veri Kaynakları

| Platform | Yöntem | Yorum Sayısı |
|----------|--------|--------------|
| Google Play Store | google-play-scraper | ~49.925 |
| Apple App Store | iTunes RSS API | ~21.075 |
| **Toplam** | | **~71.000** |

### Kategoriler ve Uygulamalar (55 Uygulama)

| Kategori | Sayı | Örnekler |
|----------|------|----------|
| Sosyal Medya | 10 | YouTube, Instagram, TikTok, WhatsApp, X |
| E-Ticaret | 10 | Trendyol, Hepsiburada, Amazon, n11 |
| Yemek Siparişi | 8 | Yemeksepeti, Getir, Trendyol Go |
| Market | 9 | BİM, ŞOK, Migros, A101 |
| Kariyer | 6 | LinkedIn, Kariyer.net, İŞKUR |
| Bankacılık | 12 | Ziraat, Akbank, Garanti BBVA, Yapı Kredi |

---

## Proje Aşamaları

```
Problem Tanımı → Veri Toplama → Keşifsel Analiz → Ön İşleme → Modelleme → Değerlendirme → Raporlama
```

| Aşama | Durum |
|-------|-------|
| Problem tanımı ve planlama | Tamamlandı |
| Veri toplama (Web Scraping) | Tamamlandı |
| Veri anlama ve keşifsel analiz | Tamamlandı |
| Veri ön işleme | Tamamlandı |
| TF-IDF ve kelime bulutları | Tamamlandı |
| Konu modelleme (LDA) | Tamamlandı |
| BERT ile duygu analizi | Tamamlandı |
| Klasik ML modelleri | Tamamlandı |
| Model karşılaştırması | Tamamlandı |
| LIME yorumlanabilirlik | Tamamlandı |
| Streamlit arayüzü | Tamamlandı |
| Final raporu ve sunum | Tamamlandı |

---

## Teknolojiler

| Kategori | Araçlar |
|----------|---------|
| Dil | Python 3.9+ |
| Veri Toplama | BeautifulSoup, Selenium, google-play-scraper, app-store-scraper |
| Veri İşleme | Pandas, NumPy |
| NLP | NLTK, Scikit-learn, Transformers (Hugging Face) |
| Derin Öğrenme | PyTorch, BERT (nlptown/bert-base-multilingual-uncased-sentiment) |
| Klasik ML | Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost |
| Yorumlanabilirlik | LIME |
| Görselleştirme | Matplotlib, Seaborn, WordCloud |
| Web Arayüzü | Streamlit |
| Ortam | Jupyter Notebook |

---

## Proje Yapısı

```
data-mining-project/
│
├── data/
│   ├── raw/                          # Ham veri setleri
│   │   ├── app_store_reviews.csv     # App Store yorumları (~21K)
│   │   ├── google_play_reviews.csv   # Google Play yorumları (~50K)
│   │   └── scrape_log.txt            # Toplama log dosyası
│   └── processed/                    # İşlenmiş veri setleri
│
├── notebooks/
│   ├── Müşteri_Geribildirimleri_Duygu_Analizi.ipynb  # Ana kapsamlı notebook
│   ├── 01_tfidf_wordcloud.ipynb      # TF-IDF ve kelime bulutları
│   ├── 02_bert_sentiment.ipynb       # BERT duygu analizi
│   └── 03_word_attributions_lime.ipynb # LIME yorumlanabilirlik
│
├── src/
│   ├── scraping/                     # Web scraping modülleri
│   │   ├── app_config.py             # 55 uygulama yapılandırması
│   │   ├── google_play.py            # Google Play scraper
│   │   ├── app_store.py              # App Store scraper
│   │   └── run_all.py                # Orkestratör
│   ├── models/
│   │   └── bert.py                   # BERT tahmin modülü
│   ├── preprocessing/
│   │   └── clean_reviews.py          # Veri temizleme
│   └── app/
│       └── streamlit_app.py          # İnteraktif web arayüzü
│
├── visuals/                          # Grafik ve görselleştirmeler
│   └── lime/                         # LIME HTML açıklamaları
│
├── Teorik_Altyapı/                   # Teorik dokümanlar
│   ├── sistem_tasarımı.md            # Sistem tasarımı açıklaması
│   └── teorik_çerçeve.md            # Teorik altyapı dokümanı
│
├── Veri_Madenciliği_Ön_Sunum.pptx    # Ön sunum dosyası
├── requirements.txt                  # Python bağımlılıklar
└── README.md
```

---

## Kurulum

```bash
git clone https://github.com/furkanozturk06/data-mining-project.git
cd data-mining-project
pip install -r requirements.txt
```

## Kullanım

### 1. Jupyter Notebook (Ana Analiz)
```bash
jupyter notebook notebooks/Müşteri_Geribildirimleri_Duygu_Analizi.ipynb
```

### 2. Streamlit Web Arayüzü
```bash
streamlit run src/app/streamlit_app.py
```

### 3. Veri Toplama (Opsiyonel)
```bash
python src/scraping/run_all.py
```

---

## Sonuçlar ve Bulgular

### Model Performans Karşılaştırması

Projede 6 farklı model test edilmiş ve karşılaştırmalı analiz yapılmıştır:

| Model | Açıklama |
|-------|----------|
| **BERT (multilingual)** | Derin öğrenme tabanlı, en yüksek doğruluk |
| Logistic Regression | Hızlı ve etkili doğrusal sınıflandırıcı |
| Naive Bayes | Metin sınıflandırmada klasik yaklaşım |
| SVM | Yüksek boyutlu veride başarılı |
| Random Forest | Topluluk öğrenmesi yaklaşımı |
| XGBoost | Gradyan artırma tabanlı güçlü model |

### Temel Bulgular

- **TF-IDF Analizi:** Pozitif ve negatif yorumlarda belirgin kelime farklılıkları tespit edilmiştir
- **Konu Modelleme:** Müşterilerin en çok performans, müşteri hizmeti ve fiyatlandırma konularında yorum yaptığı görülmüştür
- **LIME Yorumlanabilirlik:** Modelin karar mekanizması şeffaf hale getirilmiştir

---

## Kısıtlamalar ve Gelecek Çalışma

### Kısıtlamalar
- BERT modeli önceden eğitilmiş olup, Türkçe'ye özgü ince ayar yapılmamıştır
- Veri seti belirli kategorilerdeki uygulamalarla sınırlıdır
- Yıldız puanlarına dayalı etiketleme gerçek duyguyu tam yansıtmayabilir

### Gelecek Çalışma Önerileri
- Türkçe'ye özgü BERT modeli (BERTurk) ile ince ayar
- Daha geniş uygulama ve kategori yelpazesi
- Zaman serisi analizi ile duygu trendlerinin takibi
- Aspect-Based Sentiment Analysis ile alt-konu bazlı analiz

---

## Ekip

| Ad Soyad | Öğrenci No |
|----------|------------|
| Furkan Öztürk | 230229083 |
| Taha Yasin Çiçek | 230229088 |
| Ziyaeddin Ayerden | 210229022 |

---

## Bağlantılar

- [GitHub Deposu](https://github.com/furkanozturk06/data-mining-project)
- [BERT Modeli — Hugging Face](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
- [LIME Kütüphanesi](https://github.com/marcotcr/lime)
- [Streamlit Dokümantasyonu](https://docs.streamlit.io/)

---

## İletişim

Sorularınız veya katkıda bulunmak için: [GitHub Issues](https://github.com/furkanozturk06/data-mining-project/issues)
