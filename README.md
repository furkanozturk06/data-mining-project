<div align="center">

# Müşteri Geribildirimlerine Dayalı Memnuniyet Analizi

**Veri Madenciliği Dersi — Dönem Projesi**

</div>

---

## Problem Tanımı ve Hedef Belirleme

## Projenin Amacı

Bu projenin amacı, farklı platformlardan toplanan müşteri yorumları ve puanlamaları üzerinde doğal dil işleme ve makine öğrenmesi teknikleri kullanarak otomatik memnuniyet sınıflandırması yapmaktır.

## Kapsam

Yelp, Trustpilot, Google Reviews ve çeşitli e-ticaret platformlarından web scraping yöntemiyle müşteri yorumları ve yıldız puanları toplanacaktır. Toplanan veriler üzerinde metin ön işleme, özellik çıkarma ve en az iki farklı makine öğrenmesi modeli uygulanarak karşılaştırmalı analiz yapılacaktır.

## Araştırma Sorusu

Müşteri yorumlarındaki metin verisi kullanılarak memnuniyet düzeyi (pozitif / negatif / nötr) ne düzeyde doğrulukla tahmin edilebilir?

## Neden Bu Problem?

Günümüzde müşteriler satın alma kararlarını büyük ölçüde diğer kullanıcıların yorumlarına dayanarak vermektedir. Ancak bu yorumların manuel olarak okunup değerlendirilmesi büyük veri hacimlerinde pratik değildir. Otomatik duygu analizi, işletmelerin müşteri memnuniyetini hızlı ve ölçeklenebilir şekilde takip etmesine olanak tanır.

## Hedeflenen Çıktı

| Çıktı | Açıklama |
|-------|----------|
| Veri Seti | Farklı platformlardan toplanmış, temizlenmiş ve etiketlenmiş müşteri yorumları ve puanları |
| Model Karşılaştırması | En az iki farklı makine öğrenmesi modelinin performans analizi |
| Görselleştirme | Sonuçları gösteren metrikler ve grafikler |
| Raporlama | Final raporu ve sunum |

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
