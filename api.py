from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
import uvicorn

try:
    from lstm_inference import LSTMPredictor
    predictor = LSTMPredictor()
except ImportError:
    from inference import LanguagePredictor
    predictor = LanguagePredictor()

app = FastAPI(title="Smart Language Recommendation API")

try:
    videos_df = pd.read_csv("videos_db.csv")
except Exception:
    videos_df = pd.DataFrame()

user_history_db = {}

class InteractionInput(BaseModel):
    username: str
    watch_time: float
    watch_percentage: float
    engagement_score: float
    liked: int

@app.post("/predict")
def predict(data: InteractionInput):
    user = data.username
    current_input = {
        "watch_time": data.watch_time,
        "watch_percentage": data.watch_percentage,
        "engagement_score": data.engagement_score,
        "liked": data.liked
    }
    
    history = user_history_db.get(user, [])
    top_langs = predictor.predict_top_k(current_input, history)
    
    # Update history for stateful features
    user_history_db.setdefault(user, []).append(current_input)
    
    # Get recommendations
    predicted_lang = top_langs[0]["language"]
    
    recs = []
    if not videos_df.empty:
        if predicted_lang in videos_df["language"].values:
            # Get real videos from the database
            recs = videos_df[videos_df["language"] == predicted_lang].sample(min(5, len(videos_df[videos_df["language"] == predicted_lang])))[["video_id", "title"]].to_dict(orient="records")
        else:
            # Universal fallback for any other language
            recs = [{"video_id": f"vid_{predicted_lang.lower()}_{i}", "title": f"Trending {predicted_lang} Content - Video {i+1}"} for i in range(3)]
        
    return {
        "languages": top_langs,
        "recommendations": recs,
        "history": user_history_db[user]
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
