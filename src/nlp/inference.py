import os
import joblib
import torch
import numpy as np
from src.modeling.train_baseline import TextCNN, BiLSTMAttention
from src.nlp.groq_summary import predict_sentiment as groq_predict_sentiment
from src.nlp.sentiment import predict as bert_predict

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")

# Bellekte tutulacak modeller (Singleton pattern benzeri)
_models_cache = {}

def load_pkl(filename):
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path)

def get_ml_resources():
    if "tfidf" not in _models_cache:
        _models_cache["tfidf"] = load_pkl("tfidf_vectorizer.pkl")
        _models_cache["lr"] = load_pkl("logistic_regression.pkl")
        _models_cache["svm"] = load_pkl("svm.pkl")
        _models_cache["le"] = load_pkl("label_encoder.pkl")
    return _models_cache["tfidf"], _models_cache["lr"], _models_cache["svm"], _models_cache["le"]

def get_dl_resources():
    if "vocab" not in _models_cache:
        _models_cache["vocab"] = load_pkl("vocabulary.pkl")
        
        vocab_size = len(_models_cache["vocab"].word2idx)
        embed_dim = 100
        num_classes = 3
        
        # TextCNN Yükle
        textcnn = TextCNN(vocab_size, embed_dim, num_classes)
        ckpt_cnn = torch.load(os.path.join(MODELS_DIR, "textcnn.pt"), map_location=torch.device('cpu'), weights_only=True)
        textcnn.load_state_dict(ckpt_cnn["model_state_dict"])
        textcnn.eval()
        _models_cache["textcnn"] = textcnn
        
        # Bi-LSTM Yükle
        bilstm = BiLSTMAttention(vocab_size, embed_dim, hidden_dim=128, num_classes=num_classes, num_layers=2)
        ckpt_lstm = torch.load(os.path.join(MODELS_DIR, "bilstm_attention.pt"), map_location=torch.device('cpu'), weights_only=True)
        bilstm.load_state_dict(ckpt_lstm["model_state_dict"])
        bilstm.eval()
        _models_cache["bilstm"] = bilstm
        
        if "le" not in _models_cache:
            _models_cache["le"] = load_pkl("label_encoder.pkl")
            
    return _models_cache["vocab"], _models_cache["textcnn"], _models_cache["bilstm"], _models_cache["le"]

def predict_sentiment_all(text: str, model_name: str) -> dict:
    if not text.strip():
        return {"label": "nötr", "positive": 0.0, "negative": 0.0}
        
    if model_name == "BERT (HuggingFace)":
        return bert_predict(text)
        
    elif model_name == "LLM (Llama-3)":
        return groq_predict_sentiment(text)
        
    elif model_name == "Lojistik Regresyon":
        tfidf, lr, svm, le = get_ml_resources()
        X = tfidf.transform([text])
        y_pred = lr.predict(X)[0]
        probs = lr.predict_proba(X)[0]
        
        classes = lr.classes_
        prob_dict = {c: p for c, p in zip(classes, probs)}
        
        return {
            "label": str(y_pred), 
            "positive": float(prob_dict.get("pozitif", 0.0)), 
            "negative": float(prob_dict.get("negatif", 0.0))
        }
        
    elif model_name == "Destek Vektör Makineleri (SVM)":
        tfidf, lr, svm, le = get_ml_resources()
        X = tfidf.transform([text])
        y_pred = svm.predict(X)[0]
        
        # LinearSVC doesn't have predict_proba, use decision_function with softmax
        decisions = svm.decision_function(X)[0]
        exp_decisions = np.exp(decisions - np.max(decisions))
        probs = exp_decisions / exp_decisions.sum()
        
        classes = svm.classes_
        prob_dict = {c: p for c, p in zip(classes, probs)}
        
        return {
            "label": str(y_pred), 
            "positive": float(prob_dict.get("pozitif", 0.0)), 
            "negative": float(prob_dict.get("negatif", 0.0))
        }
        
    elif model_name == "TextCNN (Derin Öğrenme)":
        vocab, textcnn, bilstm, le = get_dl_resources()
        encoded = vocab.encode(text)
        if not encoded:
            encoded = [0] # OOV durumunda hata vermemesi için
        tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            outputs = textcnn(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).numpy()
            pred_idx = torch.argmax(outputs, dim=1).item()
        
        label = le.inverse_transform([pred_idx])[0]
        classes = le.inverse_transform([0, 1, 2])
        prob_dict = {c: float(p) for c, p in zip(classes, probs)}
        
        return {
            "label": label, 
            "positive": prob_dict.get("pozitif", 0.0), 
            "negative": prob_dict.get("negatif", 0.0)
        }
        
    elif model_name == "Bi-LSTM + Attention":
        vocab, textcnn, bilstm, le = get_dl_resources()
        encoded = vocab.encode(text)
        if not encoded:
            encoded = [0]
        tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            outputs = bilstm(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).numpy()
            pred_idx = torch.argmax(outputs, dim=1).item()
            
        label = le.inverse_transform([pred_idx])[0]
        classes = le.inverse_transform([0, 1, 2])
        prob_dict = {c: float(p) for c, p in zip(classes, probs)}
        
        return {
            "label": label, 
            "positive": prob_dict.get("pozitif", 0.0), 
            "negative": prob_dict.get("negatif", 0.0)
        }
        
    return {"label": "nötr", "positive": 0.0, "negative": 0.0}
