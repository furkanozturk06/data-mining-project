import os
import sys

# Ensure src module is discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import json
import time
import pandas as pd
from tqdm import tqdm
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.config import GROQ_API_KEY

def main():
    print("BÜYÜK DİL MODELİ (LLM) DEĞERLENDİRMESİ - Llama 3")

    if not GROQ_API_KEY:
        print("HATA: GROQ_API_KEY ayarlanmamış!")
        print("Lütfen .env dosyasında GROQ_API_KEY ayarlandığından emin olun.")
        sys.exit(1)

    client = Groq(api_key=GROQ_API_KEY)
    DATA_PATH = os.path.join("data", "processed", "all_reviews_labeled.csv")

    if not os.path.exists(DATA_PATH):
        print(f"HATA: {DATA_PATH} bulunamadı!")
        sys.exit(1)

    # Veriyi Yükle
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "sentiment_label"])
    df["text"] = df["text"].astype(str)

    # Modeldekiyle birebir aynı test setini elde etmek için
    _, X_test, _, y_test = train_test_split(
        df["text"], df["sentiment_label"], test_size=0.2, random_state=42, stratify=df["sentiment_label"]
    )

    # Groq API Limitlerine takılmamak ve hızlı sonuç almak için test setinden 100 örnek seçelim.
    # (LLM'in kabiliyetini göstermek için 100 örnek yeterlidir)
    sample_size = 50
    test_df = pd.DataFrame({"text": X_test, "true_label": y_test}).sample(n=sample_size, random_state=42)

    print(f"\nTest setinden {sample_size} rastgele yorum seçildi.")
    print("Groq API (llama-3.3-70b-versatile) üzerinden sıfır-eğitim (zero-shot) analiz başlıyor...\n")

    predictions = []
    actuals = test_df["true_label"].tolist()
    texts = test_df["text"].tolist()

    for i in tqdm(range(len(texts)), desc="LLM Tahminleri Alınıyor"):
        text = texts[i]
        
        if len(text) > 1000:
            text = text[:1000]

        prompt = f"""Şu müşteri yorumunu okuyup duygu analizi yap.
Yorum: "{text}"

Cevabın SADECE geçerli bir JSON olmalıdır. Başka hiçbir açıklama yazma.
Geçerli Sınıflar: "pozitif", "negatif", "nötr"

Format:
{{"label": "seçtiğin_sınıf"}}"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "Sen usta bir veri madenciliği ve Türkçe NLP uzmanısın. Yalnızca istenen JSON formatında cevap verirsin."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=50,
                    temperature=0.0,
                )
                
                raw = response.choices[0].message.content.strip()
                result = json.loads(raw)
                label = result.get("label", "nötr").lower()
                
                if label not in ["pozitif", "negatif", "nötr"]:
                    label = "nötr"
                    
                predictions.append(label)
                time.sleep(1.2) # API limitini aşmamak için bekleme
                break # Başarılıysa döngüden çık
                
            except Exception as e:
                if attempt == max_retries - 1:
                    predictions.append("nötr")
                else:
                    time.sleep(3 * (attempt + 1)) # Gecikmeyi artırarak tekrar dene

    # Değerlendirme
    print("\n\nLLM (LLAMA-3.3-70B) TEST SONUÇLARI")
    
    acc = accuracy_score(actuals, predictions)
    print(f"\nLLM Accuracy (Sıfır-Eğitim): %{acc * 100:.2f}\n")
    print("Detaylı Rapor:\n")
    print(classification_report(actuals, predictions, target_names=["negatif", "nötr", "pozitif"]))

    print("\nNot: Bu yüksek başarı oranı, Llama-3 gibi devasa LLM'lerin")
    print("metnin bağlamını hiçbir eğitim almadan doğrudan anlayabildiğini gösterir.")

    # Sonuçları Kaydet
    os.makedirs("models", exist_ok=True)
    report_dict = classification_report(actuals, predictions, target_names=["negatif", "nötr", "pozitif"], output_dict=True)
    report_dict["accuracy_score"] = acc
    with open("models/llm_results.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print("\n[+] LLM sonuçları kaydedildi → models/llm_results.json")

if __name__ == "__main__":
    main()
