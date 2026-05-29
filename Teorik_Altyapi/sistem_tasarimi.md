# Sistem Tasarımı

## Genel Mimari

Bu proje aşağıdaki boru hattı (pipeline) mimarisini takip etmektedir:

```
                    +------------------+
                    |  VERİ TOPLAMA    |
                    |  (Web Scraping)  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Google Play     |
                    |  Scraper         |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  App Store       |
                    |  Scraper         |
                    +--------+---------+
                             |
              +--------------v--------------+
              |      HAM VERİ               |
              |  app_store_reviews.csv      |
              |  google_play_reviews.csv    |
              +--------------+--------------+
                             |
              +--------------v--------------+
              |    VERİ TEMİZLEME           |
              |  - Eksik değer işleme       |
              |  - URL/noktalama kaldırma   |
              |  - Stopword temizleme       |
              |  - Tokenizasyon             |
              +--------------+--------------+
                             |
          +------------------+------------------+
          |                  |                  |
+---------v------+  +--------v-------+  +-------v--------+
| TF-IDF         |  | KONU           |  | ÖZELLİK        |
| Analizi        |  | MODELLEME      |  | MÜHENDİSLİĞİ   |
| + Kelime       |  | (LDA)          |  | - char_count   |
| Bulutları      |  |                |  | - word_count   |
+----------------+  +----------------+  +-------+--------+
                                                |
          +-------------------------------------+
          |                                     |
+---------v----------+            +-------------v-----------+
|   BERT             |            |   KLASİK ML             |
|   (multilingual)   |            |   - Logistic Regression |
|   - 5-sınıf yıldız |            |   - Naive Bayes         |
|   - 3-sınıf duygu  |            |   - SVM                 |
|                    |            |   - Random Forest       |
+--------+-----------+            |   - XGBoost             |
         |                        +-------------+-----------+
         |                                      |
         +------------------+-------------------+
                            |
              +-------------v--------------+
              |    MODEL DEĞERLENDİRME     |
              |  - Confusion Matrix        |
              |  - Accuracy / F1-Score     |
              |  - Classification Report   |
              |  - Cross-Validation        |
              +-------------+--------------+
                            |
              +-------------v--------------+
              |    LIME YORUMLANABİLİRLİK  |
              |  - Kelime ağırlıkları      |
              |  - Karar açıklamaları      |
              +-------------+--------------+
                            |
              +-------------v--------------+
              |    STREAMLİT ARAYÜZÜ       |
              |  - Gerçek zamanlı tahmin   |
              |  - Uygulama arama         |
              |  - İstatistik görüntüleme  |
              +----------------------------+
```

## Kullanılan Model: BERT

**Model Adı:** `nlptown/bert-base-multilingual-uncased-sentiment`

- Milyonlarca ürün yorumu ile ince ayarlanmış
- 6 dilde destek (Türkçe dahil): İngilizce, Almanca, Felemenkçe, İspanyolca, Fransızca, İtalyanca
- 5-sınıflı yıldız tahmini (1-5)
- 3-sınıflı duygu dönüşümü: Negatif (1-2), Nötr (3), Pozitif (4-5)

## Veri Toplama Stratejisi

### Dengeli Örnekleme
Her uygulama için 1-5 yıldız aralığında eşit sayıda yorum toplanmıştır (her yıldızdan ~200 yorum). Bu yaklaşım, sınıf dengesizliği problemini en aza indirmektedir.

### Platform Farklılıkları
- **Google Play:** `google-play-scraper` kütüphanesi, doğrudan API erişimi
- **App Store:** iTunes RSS API, JSON feed parse etme, sayfalama desteği

## Değerlendirme Metrikleri

| Metrik | Açıklama |
|--------|----------|
| Accuracy | Doğru tahminlerin toplam tahminlere oranı |
| Precision | Pozitif tahminlerin doğruluk oranı |
| Recall | Gerçek pozitiflerin yakalanma oranı |
| F1-Score | Precision ve Recall'un harmonik ortalaması |
| Confusion Matrix | Sınıf bazında tahmin performansı |
| Cross-Validation | K-katlama ile genelleme performansı |
