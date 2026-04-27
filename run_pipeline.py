import subprocess
import time
import sys

def main():
    print("="*50)
    print("🎬 STARTING ML PIPELINE")
    print("="*50)
    
    print("\n[1/4] Processing Real Video Data & Generating Behaviors...")
    subprocess.run([sys.executable, "data_processing.py"], check=True)
    
    print("\n[2/5] Engineering Features & Training XGBoost Model...")
    subprocess.run([sys.executable, "model_training.py"], check=True)
    
    print("\n[3/5] Training LSTM Sequence Model for Live Simulator...")
    subprocess.run([sys.executable, "lstm_model.py"], check=True)
    
    print("\n[4/5] Starting FastAPI Server (Port 8000)...")
    api_process = subprocess.Popen([sys.executable, "api.py"])
    
    print("\n[5/5] Starting Streamlit Frontend...")
    time.sleep(3) # Wait for API to warm up
    frontend_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend.py"])
    
    print("\n✅ System is running! Press Ctrl+C to terminate.")
    
    try:
        api_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        api_process.terminate()
        frontend_process.terminate()
        print("Goodbye!")

if __name__ == "__main__":
    main()
