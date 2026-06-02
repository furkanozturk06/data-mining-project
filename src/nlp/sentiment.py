"""Turkish BERT sentiment analysis using savasy/bert-base-turkish-sentiment-cased."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, attn_implementation="eager"
        )
        _model.eval()


def predict(text: str) -> dict:
    """Return sentiment prediction with label and confidence score."""
    _load_model()
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
    id2label = _model.config.id2label
    label = id2label[int(np.argmax(probs))]
    return {
        "label": label,
        "positive": float(probs[list(id2label.values()).index("positive")]),
        "negative": float(probs[list(id2label.values()).index("negative")]),
    }


def get_word_weights(text: str) -> list[tuple[str, float]]:
    """Return (word, weight) pairs using BERT attention from last layer [CLS] token."""
    _load_model()
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    with torch.no_grad():
        outputs = _model(**inputs, output_attentions=True)

    # Average attention across all heads in the last layer, from [CLS] to each token
    attentions = outputs.attentions[-1]  # (1, num_heads, seq_len, seq_len)
    avg_attention = attentions[0].mean(dim=0)[0]  # (seq_len,) from CLS
    avg_attention = avg_attention.numpy()

    tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Merge subword tokens (## prefix) into original words
    words = []
    weights = []
    i = 1  # skip [CLS]
    while i < len(tokens) - 1:  # skip [SEP]
        token = tokens[i]
        weight = float(avg_attention[i])
        if token.startswith("##"):
            if words:
                words[-1] += token[2:]
                weights[-1] = max(weights[-1], weight)
        else:
            words.append(token)
            weights.append(weight)
        i += 1

    # Normalize weights to [0, 1]
    max_w = max(weights) if weights else 1.0
    normalized = [(w, v / max_w) for w, v in zip(words, weights)]
    return sorted(normalized, key=lambda x: x[1], reverse=True)
