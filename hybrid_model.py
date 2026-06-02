import os
import torch
import joblib
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

class HybridEnsembleClassifier:
    def __init__(self, svc_model_path="models/best_baseline_model.pkl", 
                 tfidf_path="models/tfidf_vectorizer.pkl", 
                 bert_model_path="./my_bert_model"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Hybrid Model loading on: {self.device}")
        
        # 1. Load Classical Models
        print("Loading SVC and TF-IDF...")
        self.svc = joblib.load(svc_model_path)
        self.tfidf = joblib.load(tfidf_path)
        
        # 2. Load BERT Model
        print("Loading Fine-Tuned BERT...")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.bert = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
        self.bert.to(self.device)
        self.bert.eval()
        
        # Label Mappings
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        # Check if model has its own config
        if hasattr(self.bert.config, 'id2label') and self.bert.config.id2label:
            self.id2label = {int(k): v for k, v in self.bert.config.id2label.items()}
            
    def _clean_text(self, text):
        """Basic cleaning similar to what tf-idf expects."""
        text = str(text).lower()
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[^a-z0-9ğüşıöç\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_proba(self, text):
        """Returns the probability distribution from both models and the ensemble."""
        
        # --- 1. SVC/LGBM Prediction ---
        cleaned_text = self._clean_text(text)
        X_tfidf = self.tfidf.transform([cleaned_text])
        
        # Pad with 5 zeros for numerical features (StandardScaled mean = 0)
        import scipy.sparse
        zero_pad = scipy.sparse.csr_matrix(np.zeros((1, 5)))
        X_combined = scipy.sparse.hstack([X_tfidf, zero_pad]).tocsr()
        
        # Predict probability
        svc_probs = self.svc.predict_proba(X_combined)[0] 
        svc_classes = self.svc.classes_
        
        # Initialize an empty array for aligned SVC probs
        aligned_svc_probs = np.zeros(3)
        for i, cls in enumerate(svc_classes):
            if cls == 0 or cls == 'negative': aligned_svc_probs[0] = svc_probs[i]
            elif cls == 1 or cls == 'neutral': aligned_svc_probs[1] = svc_probs[i]
            elif cls == 2 or cls == 'positive': aligned_svc_probs[2] = svc_probs[i]
            
        # --- 2. BERT Prediction ---
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        
        bert_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        # --- 3. Ensemble (Soft Voting) ---
        # 50% BERT, 50% SVC
        hybrid_probs = (aligned_svc_probs * 0.5) + (bert_probs * 0.5)
        
        return {
            "svc_probs": aligned_svc_probs,
            "bert_probs": bert_probs,
            "hybrid_probs": hybrid_probs,
            "hybrid_pred_id": np.argmax(hybrid_probs)
        }
    
    def predict(self, text):
        res = self.predict_proba(text)
        pred_id = res["hybrid_pred_id"]
        return self.id2label[pred_id], res["hybrid_probs"][pred_id]

if __name__ == "__main__":
    print("Testing Hybrid Model...")
    hybrid = HybridEnsembleClassifier()
    test_text = "Uygulama aslında fena değil ama kargo çok gecikti"
    label, conf = hybrid.predict(test_text)
    print(f"\nText: {test_text}")
    print(f"Prediction: {label} (Conf: {conf:.2f})")
