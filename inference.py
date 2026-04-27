import pickle
import pandas as pd
from feature_engineering import extract_features_realtime

class LanguagePredictor:
    def __init__(self):
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            self.le = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
            
    def predict_top_k(self, current_input, history_list, k=3):
        # Extract dynamic features
        features_df = extract_features_realtime(history_list, current_input)
        X_scaled = self.scaler.transform(features_df)
        
        # Predict probabilities
        probs = self.model.predict_proba(X_scaled)[0]
        top_k_indices = probs.argsort()[-k:][::-1]
        
        langs = self.le.inverse_transform(top_k_indices)
        confs = probs[top_k_indices]
        
        return [{"language": lang, "confidence": float(conf)} for lang, conf in zip(langs, confs)]
