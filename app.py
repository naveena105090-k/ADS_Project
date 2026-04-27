from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
df = pd.read_csv("final_dataset.csv")

user_history = {}

def predict_top_k(user_input, k=2):
    import pandas as pd

    input_df = pd.DataFrame([user_input], columns=[
        "watch_time", "watch_percentage", "engagement_score", "liked"
    ])

    probs = model.predict_proba(input_df)[0]
    top_k = probs.argsort()[-k:][::-1]

    langs = le.inverse_transform(top_k)
    confs = probs[top_k]

    return list(zip(langs, confs))


def recommend(language):
    recs = df[df["language"] == language].copy()

    if "views" in df.columns:
        recs["score"] = recs["engagement_score"] + recs["views"] / (recs["views"].max() + 1)
    else:
        recs["score"] = recs["engagement_score"]

    return recs.sort_values(by="score", ascending=False).head(5).to_dict(orient="records")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user = data["username"]

    user_input = [
        data["watch_time"],
        data["watch_percentage"],
        data["engagement_score"],
        data["liked"]
    ]

    if user not in user_history:
        user_history[user] = []

    user_history[user].append(user_input)

    top_langs = predict_top_k(user_input)
    recs = recommend(top_langs[0][0])

    return jsonify({
        "languages": top_langs,
        "recommendations": recs,
        "history": user_history[user]
    })


if __name__ == "__main__":
    app.run(debug=True)