<div align="center">

# Müşteri Geribildirimlerine Dayalı Memnuniyet Analizi

**Veri Madenciliği Dersi — Dönem Projesi**

</div>

---

## Problem Tanımı

Müşteri memnuniyetini anlamak, işletmeler için kritik bir öneme sahiptir. Bu projede, farklı platformlardan toplanan müşteri yorumları ve puanlamaları analiz edilerek memnuniyet düzeyi otomatik olarak sınıflandırılmaktadır.

**Araştırma Sorusu:** Müşteri yorumlarındaki metin verisi kullanılarak memnuniyet düzeyi (pozitif / negatif / nötr) ne düzeyde doğrulukla tahmin edilebilir?

---

## Proje Hedefleri

- Farklı platformlardan müşteri yorumlarını ve puanlamalarını toplamak
- Metin verisini analiz edilebilir hale getirmek
- Duygu analizi modelleri geliştirmek ve karşılaştırmak
- Sonuçları görselleştirmek ve yorumlamak

---

## Veri Kaynakları

| Platform | Yöntem | Veri Türü |
|----------|--------|-----------|
| Yelp | Web Scraping | İşletme yorumları + puan |
| Trustpilot | Web Scraping | Ürün/hizmet yorumları + puan |
| Google Reviews | Web Scraping | İşletme yorumları + puan |
| E-ticaret platformları | Web Scraping | Ürün yorumları + puan |

---

## Proje Aşamaları

```
Problem Tanımı → Veri Toplama → Keşifsel Analiz → Ön İşleme → Modelleme → Değerlendirme → Raporlama
```

| Aşama | Durum |
|-------|-------|
| Problem tanımı ve planlama | Tamamlandı |
| Veri toplama (Web Scraping) | Devam ediyor |
| Veri anlama ve keşifsel analiz | Bekliyor |
| Veri ön işleme | Bekliyor |
| Modelleme | Bekliyor |
| Değerlendirme | Bekliyor |
| Görselleştirme | Bekliyor |
| Final raporu ve sunum | Bekliyor |

---

## Teknolojiler

| Kategori | Araçlar |
|----------|---------|
| Dil | Python |
| Veri Toplama | BeautifulSoup, Selenium, Requests |
| Veri İşleme | Pandas, NumPy |
| NLP | NLTK, Scikit-learn |
| Modelleme | Scikit-learn |
| Görselleştirme | Matplotlib, Seaborn |
| Ortam | Jupyter Notebook |

---

## Proje Yapısı

```
data-mining-project/
│
├── data/
│   ├── raw/                # Ham veri setleri
│   └── processed/          # İşlenmiş veri setleri
│
├── notebooks/              # Jupyter notebook dosyaları
│
├── src/                    # Python kaynak kodları (scraping, model vb.)
│
├── visuals/                # Grafik ve görselleştirmeler
│
├── reports/                # Final raporu
│
├── docs/                   # Ek dokümanlar, sunum
│
└── README.md
```

---

## Kurulum

```bash
git clone https://github.com/furkanozturk06/data-mining-project.git
cd data-mining-project
pip install -r requirements.txt
```

---

## Ekip

| Ad Soyad | Öğrenci No |
|----------|------------|
| Furkan Öztürk | 230229083 |
| Taha Yasin Çiçek | 230229088 |
| Ziyaeddin Ayerden | 210229022 |
