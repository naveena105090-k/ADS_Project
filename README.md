# Smart Language Recommendation System

A production-grade machine learning pipeline that predicts user language preference from real-time behavioral signals and recommends appropriate content.

## 🚀 Features
- **Data Generation**: Uses real YouTube trending datasets as a content catalog, and generates realistic synthetic user behavior where interaction patterns (watch time, completion rate, engagement) map back to specific language preferences.
- **Feature Engineering**: Calculates session-level trends, user history aggregates, and temporal drift dynamically.
- **Model Training**: Employs an `XGBoost` classifier with structured historical features, achieving solid accuracy by learning the latent behavioral profiles.
- **Backend API**: A fast, asynchronous `FastAPI` application for real-time predictions and content recommendations.
- **Frontend UI**: A `Streamlit` interface simulating user behavior inputs with dynamic charts plotting session trends.

## 🛠️ Project Structure
- `dataset/`: Directory containing the real YouTube trending CSVs.
- `data_processing.py`: Pre-processes real video data and generates user interaction logs.
- `feature_engineering.py`: Computes historical rolling features and session trends.
- `model_training.py`: Trains the XGBoost model and exports `.pkl` artifacts.
- `inference.py`: Helper class to load models and calculate real-time predictions.
- `api.py`: FastAPI application serving predictions.
- `frontend.py`: Streamlit application.
- `run_pipeline.py`: Orchestrates the entire flow from processing to serving.

## 💻 How to Run

### 1. how to Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
You can run the entire pipeline end-to-end with a single script. It will generate the data, train the model, and spin up both the FastAPI backend and Streamlit frontend.

```bash
python run_pipeline.py
```

### 3. Using the App
1. The Streamlit frontend will open in your browser at `http://localhost:8501`.
2. Enter a username (e.g., `demo_user`).
3. Tweak the behavioral signals (Watch Time, Watch Percentage, etc.) and submit interactions.
4. **Observe:** As you submit multiple interactions in a single session, the system computes trends, and you'll see the Top-K predicted languages, updated confidence metrics, and content recommendations!
