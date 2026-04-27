import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from feature_engineering import extract_features
import warnings
warnings.filterwarnings('ignore')

def train_model(data_path="user_interactions.csv"):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print("Extracting features...")
    X, y = extract_features(df, is_training=True)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    preds = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"XGBoost Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))
    
    print("Saving artifacts...")
    with open("model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("Training complete! Pipeline is ready for inference.")

if __name__ == "__main__":
    train_model()
