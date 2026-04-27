import pandas as pd

def extract_features(df, is_training=True):
    """
    Given a dataframe of interactions (sorted by time/step),
    compute rolling/historical features for each user to predict the language.
    """
    df = df.copy()
    
    # Sort by user and step
    if "step" in df.columns:
        df = df.sort_values(by=["user_id", "step"])
        
    # Calculate expanding (historical) features up to the current interaction
    # This simulates "what we know about the user at this exact moment"
    df["hist_mean_watch_pct"] = df.groupby("user_id")["watch_percentage"].transform(lambda x: x.expanding().mean().shift(1).fillna(x.mean()))
    df["hist_mean_eng"] = df.groupby("user_id")["engagement_score"].transform(lambda x: x.expanding().mean().shift(1).fillna(x.mean()))
    df["hist_sum_likes"] = df.groupby("user_id")["liked"].transform(lambda x: x.expanding().sum().shift(1).fillna(0))
    
    # Trends: current minus historical mean
    df["eng_trend"] = df["engagement_score"] - df["hist_mean_eng"]
    
    features = [
        "watch_time", 
        "watch_percentage", 
        "engagement_score", 
        "liked",
        "hist_mean_watch_pct",
        "hist_mean_eng",
        "hist_sum_likes",
        "eng_trend"
    ]
    
    if is_training:
        return df[features], df["language"]
    return df[features]

def extract_features_realtime(history_list, current_input):
    """
    Extracts features for a single realtime prediction given user history.
    """
    if not history_list:
        # First interaction
        hist_mean_watch_pct = current_input["watch_percentage"]
        hist_mean_eng = current_input["engagement_score"]
        hist_sum_likes = 0
    else:
        hist_df = pd.DataFrame(history_list)
        hist_mean_watch_pct = hist_df["watch_percentage"].mean()
        hist_mean_eng = hist_df["engagement_score"].mean()
        hist_sum_likes = hist_df["liked"].sum()
        
    eng_trend = current_input["engagement_score"] - hist_mean_eng
    
    feature_dict = {
        "watch_time": current_input["watch_time"],
        "watch_percentage": current_input["watch_percentage"],
        "engagement_score": current_input["engagement_score"],
        "liked": current_input["liked"],
        "hist_mean_watch_pct": hist_mean_watch_pct,
        "hist_mean_eng": hist_mean_eng,
        "hist_sum_likes": hist_sum_likes,
        "eng_trend": eng_trend
    }
    return pd.DataFrame([feature_dict])
