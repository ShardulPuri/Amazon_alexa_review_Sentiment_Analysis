from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS, cross_origin
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"], supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/test", methods=["GET"])
@cross_origin()
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
@cross_origin()
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST", "OPTIONS"])
@cross_origin()
def predict():
    if request.method == "OPTIONS":
        return jsonify({'status': 'ok'}), 200

    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(predictor, scaler, cv, data)
            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Access-Control-Allow-Methods'] = 'POST,OPTIONS'
            return response

        elif "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})
    except Exception as e:
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    # Clean text: keep only letters, lowercase, no stemming or stopword removal for now
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower()
    corpus.append(review)

    # Vectorize input
    X_prediction = cv.transform(corpus).toarray()

    # Scale input features
    X_prediction_scl = scaler.transform(X_prediction)

    # Predict class probabilities
    y_proba = predictor.predict_proba(X_prediction_scl)
    print(f"Input text: {text_input}")
    print(f"Processed text: {review}")
    print(f"Class probabilities: {y_proba}")

    # Adjust threshold for positive class if desired
    threshold = 0.5
    positive_proba = y_proba[0][1]
    prediction = "Positive" if positive_proba >= threshold else "Negative"
    print(f"Probability positive: {positive_proba}, Prediction: {prediction}")
    return prediction



def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph


def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph


def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
