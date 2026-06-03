import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import streamlit as st

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

class TextCNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

@st.cache_resource
def load_dl_components():
    if not os.path.exists('models/dl_vocab.json'):
        return None, None, None
        
    with open('models/dl_vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    vocab_size = len(vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    bilstm = None
    if os.path.exists('models/bilstm_model.pt'):
        bilstm = BiLSTMModel(vocab_size, embed_dim=128, hidden_dim=128, output_dim=3)
        bilstm.load_state_dict(torch.load('models/bilstm_model.pt', map_location=device, weights_only=True))
        bilstm.to(device)
        bilstm.eval()
        
    textcnn = None
    if os.path.exists('models/textcnn_model.pt'):
        textcnn = TextCNNModel(vocab_size, embed_dim=128, n_filters=100, filter_sizes=[3,4,5], output_dim=3)
        textcnn.load_state_dict(torch.load('models/textcnn_model.pt', map_location=device, weights_only=True))
        textcnn.to(device)
        textcnn.eval()
        
    return vocab, bilstm, textcnn

@st.cache_resource
def load_xlm_roberta():
    model_path = "models/xlm-roberta-sentiment"
    if not os.path.exists(model_path):
        return None, None
        
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model
