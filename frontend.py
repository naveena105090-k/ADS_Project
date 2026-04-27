import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re

st.set_page_config(page_title="Smart Language RecSys", page_icon="🎬", layout="wide")
sns.set_theme(style="whitegrid")

# --- SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login_page():
    st.title("🔐 Login to Smart RecSys")
    st.markdown("Please log in to access your personalized language dashboard.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign In")
        
        if submit:
            if username and password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Please enter both username and password.")

def dashboard_page():
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.title(f"👋 Welcome back, {st.session_state.username}!")
    with col2:
        st.write("") 
        if st.button("Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
            
    st.markdown("Manage your account and view your personalized language predictions.")
    st.divider()

    tab1, tab2 = st.tabs(["📺 Connect YouTube History (Google Takeout)", "🎲 Live Simulator"])
    
    with tab1:
        st.header("Analyze Your Universal Watch History")
        st.markdown("Upload your **Watch History Data** from any platform (YouTube, Netflix, Instagram, TikTok, etc.) as a `.json`, `.csv`, or `.html` file to instantly learn your language preference!")
        
        uploaded_file = st.file_uploader("Upload History File", type=["json", "csv", "html"])
        
        if uploaded_file is not None:
            with st.spinner("Extracting behavioral signals from your history..."):
                try:
                    videos_text = []
                    if uploaded_file.name.endswith(".json"):
                        data = json.load(uploaded_file)
                        if isinstance(data, list):
                            videos_text = [json.dumps(item, ensure_ascii=False) for item in data]
                        elif isinstance(data, dict):
                            videos_text = [json.dumps(v, ensure_ascii=False) for v in data.values()]
                        num_videos = len(videos_text)
                    elif uploaded_file.name.endswith(".html"):
                        content = uploaded_file.read().decode("utf-8", errors="ignore")
                        # Try to extract titles from YouTube HTML tags
                        yt_links = re.findall(r'<a href="https://www.youtube.com/watch\?v=[^"]+">([^<]+)</a>', content)
                        if yt_links:
                            videos_text = yt_links
                        else:
                            # Universal HTML fallback for Netflix, Instagram, etc.
                            clean_text = re.sub(r'<[^>]+>', ' ', content)
                            videos_text = [line.strip() for line in clean_text.split('\n') if len(line.strip()) > 5]
                        num_videos = len(videos_text)
                    else:
                        df = pd.read_csv(uploaded_file)
                        videos_text = df.astype(str).agg(' '.join, axis=1).tolist()
                        num_videos = len(videos_text)
                        
                    st.success(f"✅ Successfully processed {num_videos} videos from your history!")
                    
                    from collections import Counter
                    lang_counts = Counter()
                    
                    for vid in videos_text:
                        if re.search(r'[\u0C00-\u0C7F]', vid):
                            lang_counts["Telugu"] += 1
                        elif re.search(r'[\u0900-\u097F]', vid):
                            lang_counts["Hindi"] += 1
                        elif re.search(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]', vid):
                            lang_counts["Japanese"] += 1
                        elif re.search(r'[\uAC00-\uD7AF\u1100-\u11FF]', vid):
                            lang_counts["Korean"] += 1
                        elif re.search(r'[\u0400-\u04FF]', vid):
                            lang_counts["Russian"] += 1
                        else:
                            lang_counts["English"] += 1
                            
                    total_vids = sum(lang_counts.values())
                    if total_vids == 0:
                        lang_counts["English"] = 1
                        total_vids = 1
                        
                    top_langs = [{"language": lang, "confidence": count / total_vids} for lang, count in lang_counts.most_common(3)]
                    
                    st.subheader("🌍 Your Historical Language Preference")
                    cols = st.columns(len(top_langs))
                    for i, lang_data in enumerate(top_langs):
                        with cols[i]:
                            st.metric(label=f"Rank {i+1}", value=lang_data["language"], delta=f"{lang_data['confidence']:.1%} Watched", delta_color="normal")
                            
                    # Graph for predictions
                    st.subheader("📊 Language Distribution in History")
                    df_langs = pd.DataFrame([{"Language": l, "Percentage": c/total_vids * 100} for l, c in lang_counts.most_common()])
                    fig, ax = plt.subplots(figsize=(4, 2))
                    sns.barplot(data=df_langs, x="Percentage", y="Language", palette="viridis", ax=ax)
                    ax.set_title("Languages Watched based on Takeout Data", fontsize=10)
                    ax.set_xlabel("Percentage of Watched Videos (%)", fontsize=8)
                    ax.set_ylabel("Language", fontsize=8)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    st.pyplot(fig, use_container_width=False)
                    
                    st.subheader("🎯 Content Curated From Your History")
                    top_lang_name = top_langs[0]["language"]
                    
                    # Fetch real recommendations based on the exact top detected language
                    try:
                        import os
                        if os.path.exists("videos_db.csv"):
                            videos_df = pd.read_csv("videos_db.csv")
                            target_lang = top_lang_name
                            
                            if target_lang in videos_df["language"].values:
                                recs = videos_df[videos_df["language"] == target_lang].sample(5)[["video_id", "title"]].to_dict(orient="records")
                            else:
                                recs = [{"video_id": "123", "title": f"Popular {top_lang_name} Video"}]
                                
                            for rec in recs:
                                st.markdown(f"🎥 **{rec['title']}** *(ID: {rec['video_id']})*")
                        else:
                            st.info(f"Database not found. We recommend you watch more **{top_lang_name}** videos!")
                    except Exception as e:
                        st.info(f"We recommend you watch more **{top_lang_name}** videos!")
                        
                except Exception as e:
                    st.error(f"Error processing file. Please ensure it is a valid format. Details: {e}")

    with tab2:
        st.header("Simulate Next Interaction")
        st.markdown("Manually input behavior to see how the system adapts to your actions in real-time.")
        
        c1, c2 = st.columns(2)
        with c1:
            watch_time = st.number_input("Watch Time (seconds)", min_value=0, max_value=86400, value=300, help="Total amount of time spent watching videos.")
            watch_percentage = st.slider("Watch Percentage", 0.0, 1.0, 0.5, help="What fraction of the video you completed. 1.0 means you watched the whole video.")
        with c2:
            engagement_score = st.slider("Engagement Score", 0.0, 1.0, 0.5, help="An internal metric of how actively involved you were (e.g. clicking, not skipping, commenting).")
            liked = st.number_input("Liked Videos Count", min_value=0, max_value=1000, value=1, help="Total number of videos you actually pressed the 'Like' button on.")

        if st.button("Submit Interaction", type="primary"):
            data = {
                "username": st.session_state.username,
                "watch_time": watch_time,
                "watch_percentage": watch_percentage,
                "engagement_score": engagement_score,
                "liked": liked
            }

            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=data)
                response.raise_for_status()
                result = response.json()
                
                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    st.subheader("🌍 Predicted Language Preference")
                    for item in result["languages"]:
                        st.progress(item["confidence"], text=f"{item['language']} ({item['confidence']:.1%})")
                with res_c2:
                    st.subheader("🎯 Real-Time Recommendations")
                    for rec in result["recommendations"]:
                        st.markdown(f"🎥 **{rec['title']}** *(ID: {rec['video_id']})*")
                        
                st.divider()
                st.subheader("📊 Your Account Engagement Trends")
                history = result["history"]
                
                if len(history) > 1:
                    df_hist = pd.DataFrame(history)
                    total_likes = int(df_hist["liked"].sum())
                    st.metric("👍 Total Liked Videos in Session", total_likes)
                    
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.plot(df_hist.index + 1, df_hist["engagement_score"], marker='o', label='Engagement Score', color='#ff7f0e', linewidth=2)
                    ax.plot(df_hist.index + 1, df_hist["watch_percentage"], marker='s', label='Watch %', color='#1f77b4', linewidth=2)
                    ax.set_title("Session Trends Over Time", fontsize=10)
                    ax.set_xlabel("Interaction Step", fontsize=8)
                    ax.set_ylabel("Metric Value", fontsize=8)
                    ax.set_xticks(range(1, len(history) + 1))
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.legend(fontsize=8)
                    st.pyplot(fig, use_container_width=False)
                    
                    st.info("💡 **What does this graph mean?** It visualizes how your behavior evolves chronologically over your session. **Engagement Score** measures your active interest, while **Watch %** shows how much of the content you actually finish. Instead of looking at a single click, our ML Brain analyzes the *shape and trend* of these lines to accurately guess your hidden language preferences!")
                else:
                    st.info("Submit more interactions to see your trends build up!")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Ensure FastAPI is running on port 8000 (`python api.py`).")


# --- ROUTING ---
if not st.session_state.logged_in:
    login_page()
else:
    dashboard_page()