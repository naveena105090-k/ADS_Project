import torch
import torch.nn as nn
import pickle
import numpy as np
from lstm_model import LSTMRecommender

class LSTMPredictor:
    def __init__(self, seq_len=5):
        self.seq_len = seq_len
        
        with open("lstm_label_encoder.pkl", "rb") as f:
            self.le = pickle.load(f)
        with open("lstm_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
            
        input_size = 4  # watch_time, watch_percentage, engagement_score, liked
        hidden_size = 32
        num_classes = len(self.le.classes_)
        
        self.model = LSTMRecommender(input_size, hidden_size, num_classes)
        self.model.load_state_dict(torch.load("lstm_model.pth"))
        self.model.eval()

    def predict_top_k(self, current_input, history_list, k=3):
        # We need a sequence of interactions up to seq_len
        # Combine history + current_input
        features = ["watch_time", "watch_percentage", "engagement_score", "liked"]
        
        full_history = list(history_list) + [current_input]
        
        # Take the last `seq_len` interactions
        recent = full_history[-self.seq_len:]
        
        # Extract features
        seq_data = []
        for item in recent:
            seq_data.append([item[f] for f in features])
            
        # Scale data
        seq_data_scaled = self.scaler.transform(seq_data)
        
        # Pad if less than seq_len
        if len(seq_data_scaled) < self.seq_len:
            pad = np.zeros((self.seq_len - len(seq_data_scaled), len(features)))
            seq_data_scaled = np.vstack([pad, seq_data_scaled])
            
        # Convert to tensor and add batch dimension
        X = torch.FloatTensor(seq_data_scaled).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(X)
            probs = torch.nn.functional.softmax(output, dim=1).numpy()[0]
            
        top_k_indices = probs.argsort()[-k:][::-1]
        
        langs = self.le.inverse_transform(top_k_indices)
        confs = probs[top_k_indices]
        
        return [{"language": lang, "confidence": float(conf)} for lang, conf in zip(langs, confs)]
