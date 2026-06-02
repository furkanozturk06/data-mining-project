# 📊 App Store / Google Play Yorumları Duygu Analizi Projesi

Bu proje, Türkçe mobil uygulama yorumlarını Doğal Dil İşleme (NLP) yöntemleriyle analiz eden ve kullanıcı memnuniyetini ölçen kapsamlı bir yapay zeka pipeline sunar. Proje kapsamında Klasik Makine Öğrenmesi algoritmaları, Derin Öğrenme (BERT) modelleri ve bu iki modelin gücünü birleştiren yenilikçi bir **Hibrit (Ensemble) Model** kullanılmıştır.

## 🚀 Kurulum ve Çalıştırma

Projeyi bilgisayarınıza indirdikten sonra, hemen çalıştırmak için aşağıdaki adımları izleyin.

### 1. Gereksinimleri Yükleyin
Bir terminal (veya komut satırı) açın ve proje klasörüne gidin. Ardından gerekli tüm kütüphaneleri kurmak için:

```bash
pip install -r requirements.txt
```

### 2. Arayüzü (Dashboard) Başlatın
Analiz sonuçlarını, istatistiksel grafikleri ve canlı tahmin yapan yapay zeka arayüzünü görmek için Streamlit'i başlatın:

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` adresinde proje açılacaktır. "Canlı Analiz" sekmesinde modellerin nasıl ortak karar (Hibrit Karar) verdiğini test edebilirsiniz.

---

## 🧠 Makine Öğrenmesi & Derin Öğrenme Pipeline (Adım Adım Notebooklar)

Projenin yapay zeka modellerini incelemek veya sıfırdan eğitmek isterseniz sırasıyla şu notebook'ları çalıştırabilirsiniz:

1. **`01_veri_kesfi_ve_on_isleme.ipynb`**: Ham verilerin temizlenmesi, NLP önişleme adımları ve veri seti hazırlığı.
2. **`02_kesifsel_veri_analizi.ipynb`**: Uygulama bazlı grafikler, kelime bulutları (WordCloud), yorum uzunluğu gibi metin özelliklerinin (Feature Engineering) çıkarılması.
3. **`03_baseline_modelleme.ipynb`**: TF-IDF ve matematiksel özelliklerle (uzunluk, kelime sayısı, ünlem vb.) Klasik Makine Öğrenmesi (LightGBM, SVC) modellerinin eğitilmesi.
4. **`04_bert_finetuning.ipynb`**: Türkçe ön-eğitimli derin öğrenme modelinin (ELECTRA/BERT) duygu analizi için ince ayarının (Fine-Tuning) yapılması.
5. **`05_model_karsilastirma.ipynb`**: Eğitilen klasik ve derin öğrenme modellerinin performans (F1-Score, Accuracy) kıyaslaması.
6. **`06_konu_modelleme.ipynb`**: Gözetimsiz öğrenme (LDA / BERTopic) ile müşteri şikayetlerinin konulara (Kargo, Güncelleme, Performans vb.) ayrılması.
7. **`07_hibrit_model_degerlendirme.ipynb`**: *[YENİ]* Klasik Modelin matematiksel mantığı ile BERT'in dil anlama kapasitesini birleştiren (Soft Voting Ensemble) **Hibrit Model**in nihai test performansının ölçülmesi.

*(Not: Depoda büyük model dosyaları tutulmadığı için, Canlı Analiz sekmesini kendi bilgisayarınızda çalıştırmak için model ağırlıklarınızı eğitmeniz gerekebilir.)*
