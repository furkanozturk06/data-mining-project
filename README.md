# 📊 App Store / Google Play Yorumları Duygu Analizi Projesi

Bu proje, Türkçe mobil uygulama yorumlarını Doğal Dil İşleme (NLP) yöntemleriyle analiz eden ve kullanıcı memnuniyetini ölçen kapsamlı bir pipeline sunar.

## 🚀 Kurulum ve Çalıştırma

Projeyi bilgisayarınıza indirdikten sonra, hemen çalıştırmak için aşağıdaki adımları izleyin.

### 1. Gereksinimleri Yükleyin
Bir terminal (veya komut satırı) açın ve proje klasörüne gidin. Ardından gerekli tüm kütüphaneleri kurmak için:

```bash
pip install -r requirements.txt
```

### 2. Arayüzü (Dashboard) Başlatın
Analiz sonuçlarını ve canlı tahmin yapan yapay zeka arayüzünü görmek için Streamlit'i başlatın:

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` adresinde proje açılacaktır.

---

## 🧠 Modeli Kendi Bilgisayarınızda Eğitmek İsterseniz (Opsiyonel)

GitHub deposuna **büyük boyutlu model ağırlık dosyaları (yüzlerce MB)** yüklenmediği için, `app.py` içindeki "Canlı Analiz" sekmesindeki BERT modelini ilk kez denediğinizde çalışmayabilir.

Kendi modelinizi sıfırdan eğitmek ve projeyi %100 yerel hale getirmek için sırasıyla şu notebook'ları çalıştırın:
1. `01_veri_kesfi_ve_on_isleme.ipynb` (Veri Temizliği)
2. `02_kesifsel_veri_analizi.ipynb` (Grafikler ve Analiz)
3. `03_baseline_modelleme.ipynb` (Klasik Modeller)
4. `04_bert_finetuning.ipynb` (BERT Model Eğitimi - *Bunu çalıştırdığınızda `models/advanced_bert_model` klasörü oluşacak ve Canlı Analiz ekranı çalışmaya başlayacaktır*)
5. `05_model_karsilastirma.ipynb` (Model Sonuçlarının Kıyaslanması)
6. `06_konu_modelleme.ipynb` (Yorumların konulara ayrılması)
