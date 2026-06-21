"""Kelime sözlüğü (tokenizer) — torch'a bağımlı olmayan saf Python sınıfı.

Bu sınıf ayrı bir modülde tutulur, çünkü eğitilmiş `vocabulary.pkl`
nesnesi pickle ile bu sınıfa referans verir. Sınıfı torch içeren
ağır eğitim modülünden ayırmak, modeli yükleyen ortamların (ör.
Streamlit arayüzü) sözlüğü torch yüklemeden açabilmesini sağlar.
"""

VOCAB_SIZE = 30_000
MAX_LEN = 200


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
