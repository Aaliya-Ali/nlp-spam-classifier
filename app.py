from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Routes ---

# The home route will now just return a welcome JSON message


@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Email Spam Detector API!",
        "endpoints": {
            "/predict": "POST request with 'email_content' in form data to classify text.",
            "/training": "POST request to train or retrain the spam detection model."
        }
    })


if __name__ == '__main__':
    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()
    app.run(debug=True)
