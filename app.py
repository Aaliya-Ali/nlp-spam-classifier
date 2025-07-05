from flask import Flask, request, jsonify

# Import functions from our utils
from utils import load_model_and_vectorizer, train_spam_model

app = Flask(__name__)

# Global variables to hold the trained model and vectorizer
# They will be loaded once the app starts or after training
spam_classifier = None
tfidf_vectorizer = None

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


@app.route('/training', methods=['POST'])
def training_endpoint():
    global spam_classifier, tfidf_vectorizer  # Declare as global to modify

    # Call the training function from utils.py
    training_info = train_spam_model()

    # After training, reload the newly trained model into global variables
    # This ensures the /predict endpoint uses the latest trained model
    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()

    # Format the report for JSON output
    report_output = {}
    if training_info.get('report'):
        for class_name, metrics in training_info['report'].items():
            # Convert numpy types to float for JSON serialization if necessary
            if isinstance(metrics, dict):
                report_output[class_name] = {k: float(v) if isinstance(
                    v, (float, int)) else v for k, v in metrics.items()}
            else:
                # For 'accuracy', 'macro avg', etc. keys if directly included
                report_output[class_name] = metrics

    # Construct the JSON response
    response_data = {
        "status": training_info.get('status'),
        "accuracy": f"{training_info['accuracy']:.4f}" if training_info.get('accuracy') else None,
        "classification_report": report_output if report_output else None
    }

    # Determine HTTP status code based on training status
    if "Error" in training_info.get('status', ''):
        return jsonify(response_data), 500  # Internal Server Error
    else:
        return jsonify(response_data), 200  # OK


if __name__ == '__main__':
    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()
    app.run(debug=True)
