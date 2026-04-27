import pandas as pd
import numpy as np
import os
import glob

def generate_data(dataset_dir="dataset", output_file="user_interactions.csv", num_users=1000, interactions_per_user=20):
    print("Generating synthetic but realistic user interactions...")
    
    # Real video metadata mapping for recommendations
    lang_map = {
        'CA': 'English', 'US': 'English', 'GB': 'English',
        'DE': 'German', 'FR': 'French', 'IN': 'Telugu',
        'JP': 'Japanese', 'KR': 'Korean', 'MX': 'Spanish', 'RU': 'Russian'
    }
    
    # Behavioral profiles per language (so the model can actually learn!)
    # (mean_duration, std_duration, mean_pct, std_pct, like_prob)
    profiles = {
        'English':  (300, 50, 0.60, 0.15, 0.5),
        'German':   (400, 80, 0.55, 0.20, 0.4),
        'French':   (250, 40, 0.65, 0.10, 0.6),
        'Telugu':    (900, 200, 0.40, 0.25, 0.7),
        'Japanese': (1200, 100, 0.85, 0.05, 0.8), # Anime/Long docs, high completion
        'Korean':   (600, 100, 0.75, 0.15, 0.7),
        'Spanish':  (350, 60, 0.50, 0.20, 0.5),
        'Russian':  (500, 90, 0.45, 0.25, 0.3)
    }
    
    all_videos = []
    csv_files = glob.glob(os.path.join(dataset_dir, "*videos.csv"))
    
    for file in csv_files:
        country_code = os.path.basename(file)[:2].upper()
        if country_code not in lang_map: continue
        lang = lang_map[country_code]
        if lang not in profiles: continue
            
        try:
            df_temp = pd.read_csv(file, encoding="utf-8", usecols=["video_id", "title", "views"], nrows=2000)
        except:
            df_temp = pd.read_csv(file, encoding="latin1", usecols=["video_id", "title", "views"], nrows=2000)
            
        df_temp["language"] = lang
        all_videos.append(df_temp)
        
    if not all_videos:
        print("Warning: No video data found. Using dummy videos.")
        videos_df = pd.DataFrame({"video_id": [f"v{i}" for i in range(100)], "title": ["Dummy Video"]*100, "views": [1000]*100, "language": ["English"]*100})
    else:
        videos_df = pd.concat(all_videos, ignore_index=True).drop_duplicates(subset=["video_id"])

    videos_df.to_csv("videos_db.csv", index=False)
    
    languages = list(profiles.keys())
    users = pd.DataFrame({
        "user_id": np.arange(1, num_users + 1),
        "true_pref_lang": np.random.choice(languages, size=num_users)
    })
    
    interactions = []
    
    for _, user in users.iterrows():
        pref_lang = user["true_pref_lang"]
        prof = profiles[pref_lang]
        
        # User creates a series of interactions in a session
        for step in range(interactions_per_user):
            # Add some temporal evolution (e.g. fatigue)
            fatigue = step / interactions_per_user
            
            vid = videos_df[videos_df["language"] == pref_lang].sample(1).iloc[0]
            
            vid_duration = np.random.normal(prof[0], prof[1])
            watch_pct = np.clip(np.random.normal(prof[2] - fatigue*0.1, prof[3]), 0.05, 1.0)
            watch_time = int(vid_duration * watch_pct)
            liked = int(np.random.rand() < (prof[4] - fatigue*0.1))
            skipped = int(watch_pct < 0.2)
            
            engagement_score = np.clip(0.6 * watch_pct + 0.3 * liked - 0.2 * skipped, 0, 1.0)
            
            interactions.append({
                "user_id": user["user_id"],
                "video_id": vid["video_id"],
                "step": step,
                "watch_time": watch_time,
                "watch_percentage": watch_pct,
                "engagement_score": engagement_score,
                "liked": liked,
                "language": pref_lang # The label we want to predict
            })
            
    df = pd.DataFrame(interactions)
    df.to_csv(output_file, index=False)
    print(f"Dataset generated with {len(df)} records. Saved to {output_file}.")

if __name__ == "__main__":
    generate_data()
