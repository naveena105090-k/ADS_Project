import requests
import time

def simulate_platform(platform_name, username, watch_time, watch_pct, engagement, liked):
    print(f"\n[{platform_name}] 📡 Sending user behavior to ML API...")
    url = "http://127.0.0.1:8000/predict"
    
    payload = {
        "username": username,
        "watch_time": watch_time,
        "watch_percentage": watch_pct,
        "engagement_score": engagement,
        "liked": liked
    }
    
    try:
        # The platform makes an HTTP POST request to our universal API
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            top_lang = data["languages"][0]
            recs = data["recommendations"]
            
            print(f"[{platform_name}] ✅ Received ML Prediction: User prefers {top_lang['language']} ({top_lang['confidence']:.1%} confidence)")
            print(f"[{platform_name}] 📺 Updating Homepage with Recommended Video: '{recs[0]['title']}'")
        else:
            print(f"[{platform_name}] ❌ API Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"[{platform_name}] ❌ Failed to connect. Is the API running? (Run 'python api.py')")

if __name__ == "__main__":
    print("=== MULTI-PLATFORM API INTEGRATION TEST ===\n")
    
    # 1. Netflix sending a movie interaction
    simulate_platform(
        platform_name="Netflix", 
        username="netflix_subscriber_99", 
        watch_time=5400, # Watched for 90 minutes
        watch_pct=0.9,   # Finished 90% of the movie
        engagement=0.8, 
        liked=1
    )
    time.sleep(1)
    
    # 2. YouTube sending a short clip interaction
    simulate_platform(
        platform_name="YouTube", 
        username="yt_viewer_01", 
        watch_time=120,  # Watched a 2 minute clip
        watch_pct=0.3,   # Only finished 30%
        engagement=0.4, 
        liked=0
    )
    time.sleep(1)
    
    # 3. Spotify Podcasts sending an audio interaction
    simulate_platform(
        platform_name="Spotify", 
        username="audio_listener_x", 
        watch_time=1800, # Listened for 30 minutes
        watch_pct=0.5, 
        engagement=0.6, 
        liked=1
    )
