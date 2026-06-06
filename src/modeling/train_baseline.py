"""
Model Eğitimleri ve Doğruluk Değerleri
=======================================
Bu script, müşteri yorumları üzerinde 4 farklı modeli eğitir ve kıyaslar:

Klasik Makine Öğrenmesi (ML):
  1. Logistic Regression  (TF-IDF)
  2. SVM - Support Vector Machine  (TF-IDF)

Derin Öğrenme (Deep Learning):
  3. TextCNN  (Word Embedding + Conv1D)
  4. Bi-LSTM + Attention  (Word Embedding + Bidirectional LSTM)

Çıktılar:
  - Her modelin Accuracy, Precision, Recall, F1-Score değerleri
  - Karşılaştırma tablosu
  - Eğitilmiş modeller  →  models/ klasörüne kaydedilir
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Proje ayarları
DATA_PATH = os.path.join("data", "processed", "all_reviews_labeled.csv")
MODELS_DIR = "models"
TEXT_COL = "text"
LABEL_COL = "sentiment_label"

# Derin öğrenme hiperparametreleri
VOCAB_SIZE = 30_000
EMBED_DIM = 128
MAX_LEN = 200
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 7) YAZI TAHMİNİ (TEST AMACIYLA KULLANIM)

def load_data():
    """Veri setini yükler ve temizler."""
    print("VERİ SETİ YÜKLENİYOR")

    if not os.path.exists(DATA_PATH):
        print(f"HATA: {DATA_PATH} bulunamadı!")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)

    print(f"  Toplam yorum sayısı : {len(df):,}")
    print(f"  Sınıf dağılımı      :")
    for label, count in df[LABEL_COL].value_counts().items():
        print(f"    {label:>10s} : {count:,}")

    return df


def split_data(df):
    """Veriyi %80 eğitim - %20 test olarak böler."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COL], df[LABEL_COL],
        test_size=0.2,
        random_state=42,
        stratify=df[LABEL_COL],
    )
    print(f"\n  Eğitim seti : {len(X_train):,} yorum")
    print(f"  Test seti   : {len(X_test):,} yorum")
    return X_train, X_test, y_train, y_test


# 2) MAKİNE ÖĞRENMESİ (ML) MODELLERİ (TF-IDF tabanlı)

def train_ml_models(X_train, X_test, y_train, y_test):
    """TF-IDF + Logistic Regression ve SVM eğitir."""
    print("\nKLASİK ML MODELLERİ EĞİTİLİYOR")

    # TF-IDF Vektörizasyonu
    print("\n  TF-IDF vektörizasyonu yapılıyor (max_features=50000, ngram=(1,3))...")
    tfidf = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Vektörizer'ı kaydet
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print("  TF-IDF vektörizer kaydedildi → models/tfidf_vectorizer.pkl")

    results = {}

    # ── Model 1: Logistic Regression ──
    print("\n  MODEL 1: Logistic Regression")
    lr = LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="lbfgs",
        n_jobs=-1,
    )
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)
    results["Logistic Regression"] = evaluate_model(y_test, y_pred_lr)
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    print("  Model kaydedildi → models/logistic_regression.pkl")

    # ── Model 2: SVM (Support Vector Machine) ──
    print("\n  MODEL 2: SVM (Support Vector Machine)")
    svm = LinearSVC(
        max_iter=3000,
        C=1.0,
        loss="squared_hinge",
        dual=True,
    )
    svm.fit(X_train_vec, y_train)
    y_pred_svm = svm.predict(X_test_vec)
    results["SVM"] = evaluate_model(y_test, y_pred_svm)
    joblib.dump(svm, os.path.join(MODELS_DIR, "svm.pkl"))
    print("  Model kaydedildi → models/svm.pkl")

    return results


# 3) DERİN ÖĞRENME İÇİN VERİ HAZIRLIĞI (Dataset, DataLoader)

class Vocabulary:
    """Basit kelime sözlüğü oluşturur (tokenizer)."""

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, max_size=VOCAB_SIZE):
        self.max_size = max_size
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}

    def build(self, texts):
        """Metinlerden en sık geçen kelimelerin sözlüğünü oluşturur."""
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # En sık geçen kelimeleri seç
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[: self.max_size - 2]:  # PAD ve UNK için 2 yer ayır
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"  Sözlük boyutu: {len(self.word2idx):,} kelime")
        return self

    def encode(self, text, max_len=MAX_LEN):
        """Metni sayısal diziye çevirir."""
        tokens = text.lower().split()[:max_len]
        ids = [self.word2idx.get(w, 1) for w in tokens]  # 1 = UNK
        # Padding
        ids += [0] * (max_len - len(ids))
        return ids


class ReviewDataset(Dataset):
    """PyTorch Dataset sınıfı."""

    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, "iloc") else self.texts[idx]
        label = self.labels[idx]
        encoded = self.vocab.encode(text)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class TextCNN(nn.Module):
    """
    TextCNN: Yoon Kim (2014) mimarisi.
    Farklı boyutlarda 1D konvolüsyon filtreleri kullanarak
    kelime kalıplarını (n-gram) yakalar.
    """

    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=(2, 3, 4, 5), num_filters=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)           # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)          # (batch, embed_dim, seq_len) → Conv1d için
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))      # (batch, num_filters, new_len)
            c = c.max(dim=2).values      # Global Max Pooling → (batch, num_filters)
            conv_outs.append(c)
        x = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)                   # (batch, num_classes)
        return x


# 5) DERİN ÖĞRENME - Bi-LSTM + ATTENTION MODELİ

class Attention(nn.Module):
    """
    Additive Attention (Bahdanau-style).
    LSTM çıktıları üzerinde hangi zaman adımlarının
    (kelimelerin) daha önemli olduğunu öğrenir.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, mask=None):
        # lstm_out: (batch, seq_len, hidden_dim)
        score = self.V(torch.tanh(self.W(lstm_out)))  # (batch, seq_len, 1)
        score = score.squeeze(-1)                      # (batch, seq_len)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(score, dim=1)          # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out)  # (batch, 1, hidden_dim)
        return context.squeeze(1), weights


class BiLSTMAttention(nn.Module):
    """
    Bi-LSTM + Attention: Cümleyi her iki yönden okuyan LSTM
    ve hangi kelimelerin duygu belirlediğine odaklanan Attention mekanizması.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attention = Attention(hidden_dim * 2)  # *2 çünkü bidirectional
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        mask = (x != 0).float()               # Padding mask
        x = self.embedding(x)                  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden_dim*2)
        context, attn_weights = self.attention(lstm_out, mask)  # (batch, hidden_dim*2)
        out = self.dropout(context)
        out = self.fc(out)                     # (batch, num_classes)
        return out


# 6) DERİN ÖĞRENME İÇİN EĞİTİM (TRAIN) DÖNGÜSÜ

def train_deep_model(model, train_loader, val_loader, model_name, num_classes):
    """Derin öğrenme modelini eğitir ve değerlendirir."""

    # Sınıf ağırlıkları hesapla (dengesiz veri seti için)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    model.to(DEVICE)
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
        # ── Eğitim ──
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # ── Doğrulama (Validation) ──
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                preds = output.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total
        val_avg_loss = val_loss / len(val_loader)
        scheduler.step(val_avg_loss)

        print(
            f"    Epoch {epoch + 1:2d}/{EPOCHS} │ "
            f"Loss: {avg_loss:.4f} │ Train Acc: %{train_acc * 100:.2f} │ "
            f"Val Loss: {val_avg_loss:.4f} │ Val Acc: %{val_acc * 100:.2f}"
        )

        # En iyi modeli sakla
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # En iyi modeli yükle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_deep_model(model, test_loader, label_encoder):
    """Derin öğrenme modelini test seti üzerinde değerlendirir."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    # Sayıları tekrar etiketlere çevir
    y_true = label_encoder.inverse_transform(all_labels)
    y_pred = label_encoder.inverse_transform(all_preds)

    return evaluate_model(y_true, y_pred)


def train_dl_models(X_train, X_test, y_train, y_test):
    """TextCNN ve Bi-LSTM+Attention modellerini eğitir."""
    print("\nDERİN ÖĞRENME MODELLERİ EĞİTİLİYOR")
    print(f"  Cihaz: {DEVICE}")

    # Label Encoding (pozitif→0, negatif→1, nötr→2)
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)
    print(f"  Sınıflar: {list(le.classes_)}")

    # Sözlük oluştur
    print("\n  Kelime sözlüğü oluşturuluyor...")
    vocab = Vocabulary(max_size=VOCAB_SIZE)
    vocab.build(X_train)

    # Dataset ve DataLoader
    train_dataset = ReviewDataset(X_train, y_train_enc, vocab)
    test_dataset = ReviewDataset(X_test, y_test_enc, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    actual_vocab_size = len(vocab.word2idx)

    # Label encoder'ı kaydet
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(vocab, os.path.join(MODELS_DIR, "vocabulary.pkl"))

    results = {}

    # ── Model 3: TextCNN ──
    print("\n  MODEL 3: TextCNN")
    textcnn = TextCNN(
        vocab_size=actual_vocab_size,
        embed_dim=EMBED_DIM,
        num_classes=num_classes,
        filter_sizes=(2, 3, 4, 5),
        num_filters=128,
    )
    total_params = sum(p.numel() for p in textcnn.parameters())
    print(f"  Toplam parametre: {total_params:,}")
    print(f"  Eğitim başlıyor ({EPOCHS} epoch)...\n")

    textcnn = train_deep_model(textcnn, train_loader, test_loader, "TextCNN", num_classes)
    results["TextCNN"] = evaluate_deep_model(textcnn, test_loader, le)

    # Modeli kaydet
    torch.save({
        "model_state_dict": textcnn.state_dict(),
        "vocab_size": actual_vocab_size,
        "embed_dim": EMBED_DIM,
        "num_classes": num_classes,
    }, os.path.join(MODELS_DIR, "textcnn.pt"))
    print("  Model kaydedildi → models/textcnn.pt")

    # ── Model 4: Bi-LSTM + Attention ──
    print("\n  MODEL 4: Bi-LSTM + Attention")
    bilstm = BiLSTMAttention(
        vocab_size=actual_vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=128,
        num_classes=num_classes,
        num_layers=2,
    )
    total_params = sum(p.numel() for p in bilstm.parameters())
    print(f"  Toplam parametre: {total_params:,}")
    print(f"  Eğitim başlıyor ({EPOCHS} epoch)...\n")

    bilstm = train_deep_model(bilstm, train_loader, test_loader, "BiLSTM_Attention", num_classes)
    results["Bi-LSTM + Attention"] = evaluate_deep_model(bilstm, test_loader, le)

    # Modeli kaydet
    torch.save({
        "model_state_dict": bilstm.state_dict(),
        "vocab_size": actual_vocab_size,
        "embed_dim": EMBED_DIM,
        "hidden_dim": 128,
        "num_classes": num_classes,
        "num_layers": 2,
    }, os.path.join(MODELS_DIR, "bilstm_attention.pt"))
    print("  Model kaydedildi → models/bilstm_attention.pt")

    return results


# 1) ORTAK YARDIMCI FONKSİYONLAR

def evaluate_model(y_true, y_pred):
    """Model performans metriklerini hesaplar ve yazdırır."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print(f"\n    Doğruluk  (Accuracy)  : %{acc * 100:.2f}")
    print(f"    Kesinlik  (Precision) : %{prec * 100:.2f}")
    print(f"    Duyarlılık (Recall)   : %{rec * 100:.2f}")
    print(f"    F1-Score              : %{f1 * 100:.2f}")
    print(f"\n    Detaylı Sınıf Raporu:")
    print("    " + "-" * 55)
    report = classification_report(y_true, y_pred, digits=4)
    for line in report.split("\n"):
        print(f"    {line}")
    print()

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }


def print_comparison_table(all_results):
    """Tüm modellerin karşılaştırma tablosunu yazdırır."""
    print("\n" + "=" * 70)
    print("  KARŞILAŞTIRMA TABLOSU")
    print("=" * 70)
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
    print("  " + "-" * 65)

    # F1-Score'a göre sırala
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["f1_score"], reverse=True)

    for i, (name, metrics) in enumerate(sorted_results):
        marker = " 🏆" if i == 0 else ""
        print(
            f"  {name:<25s}"
            f" %{metrics['accuracy']:>7.2f}"
            f"  %{metrics['precision']:>7.2f}"
            f"  %{metrics['recall']:>7.2f}"
            f"  %{metrics['f1_score']:>7.2f}{marker}"
        )

    print("  " + "-" * 65)
    best_model = sorted_results[0][0]
    best_f1 = sorted_results[0][1]["f1_score"]
    print(f"\n  ★ En başarılı model: {best_model} (F1-Score: %{best_f1:.2f})")
    print()

    return sorted_results


# 8) ANA FONKSİYON

def main():
    """Tüm modelleri eğitir ve sonuçları karşılaştırır."""
    print("\nMÜŞTERI MEMNUNİYET ANALİZİ - MODEL EĞİTİMLERİ")
    print("Logistic Regression | SVM | TextCNN | Bi-LSTM+Attention\n")

    # Veri yükleme ve bölme
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Klasik ML modelleri
    ml_results = train_ml_models(X_train, X_test, y_train, y_test)

    # Derin öğrenme modelleri
    dl_results = train_dl_models(X_train, X_test, y_train, y_test)

    # Tüm sonuçları birleştir
    all_results = {**ml_results, **dl_results}

    # Karşılaştırma tablosu
    sorted_results = print_comparison_table(all_results)

    # Sonuçları JSON olarak kaydet
    results_path = os.path.join(MODELS_DIR, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  Sonuçlar kaydedildi → {results_path}")

    print("\n  Tüm modeller başarıyla eğitildi ve kaydedildi! ✓\n")


if __name__ == "__main__":
    main()
