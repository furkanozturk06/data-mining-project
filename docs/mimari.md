# Mimari ve Veri Akisi

Bu belge, projenin uctan uca is akisini ve kaynak kod modullerinin
sorumluluklarini ozetler. Ayrintili anlatim ve sonuclar icin kok dizindeki
[README.md](../README.md) ve [reports/](../reports) altindaki final raporuna bakiniz.

## Uctan uca akis

```
1. Veri Toplama        ->  2. Temizleme        ->  3. Hibrit Etiketleme
   (App Store &            (normalize,             (yildiz kurali +
    Google Play)           deduplikasyon)          Groq Llama 3.1 duzeltmesi)
        |                                                   |
        v                                                   v
6. Streamlit Arayuzu   <-  5. Degerlendirme     <-  4. Modelleme
   (6 model, gorseller)    (Accuracy/F1, LLM,        (LogReg, SVM,
                            BERT)                      TextCNN, Bi-LSTM)
```

## Modul haritasi (`src/`)

| Modul | Sorumluluk |
|-------|-----------|
| `scraping/google_play.py` | Google Play yorumlarini yildiz bazinda dengeli ceker (`google-play-scraper`) |
| `scraping/app_store.py` | App Store yorumlarini iTunes RSS API uzerinden ceker |
| `scraping/run_all.py` | Tum scraping akisini yonetir; ham CSV ciktilari `data/raw/` |
| `preprocessing/clean_reviews.py` | Normalize, gecersiz kayit eleme, deduplikasyon; cikti `data/processed/` |
| `preprocessing/groq_label_reviews.py` | Yildiz ile celisen yorumlari Groq LLM ile yeniden etiketler |
| `modeling/train_baseline.py` | 4 modeli (LogReg, SVM, TextCNN, Bi-LSTM) egitir ve karsilastirir |
| `modeling/vocabulary.py` | Derin ogrenme icin tokenizer / kelime sozlugu (`Vocabulary`) |
| `modeling/evaluate_llm.py` | Llama 3 ile sifir-atis (zero-shot) degerlendirme |
| `nlp/sentiment.py` | Turkce BERT tahmini + attention agirliklari |
| `nlp/inference.py` | Tum modeller icin ortak tahmin katmani |
| `nlp/word_analysis.py` | TF-IDF tabanli kelime analizi ve kelime bulutu |
| `nlp/groq_summary.py` | Groq ile firma bazinda ozet ve tahmin |
| `ui/app.py` | Streamlit web arayuzu (4 sekme) |

## Veri ve model dosyalari

- `data/raw/` : ham scraping ciktilari (App Store, Google Play)
- `data/processed/` : temizlenmis ve etiketlenmis veri kumeleri
- `models/` : egitilmis modeller (`.pkl`, `.pt`), vektorizer, label encoder,
  kelime sozlugu ve sonuc JSON dosyalari

## Egitilmis modellerin yeniden uretimi

Adim adim komutlar icin [README.md](../README.md) "Kullanim" bolumune bakiniz.
Repo onceden toplanmis veriyi ve egitilmis modelleri icerdiginden, arayuz
dogrudan `streamlit run src/ui/app.py` ile calistirilabilir.
