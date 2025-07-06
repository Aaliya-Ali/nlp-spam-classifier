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

    # Accept JSON hyperparameters from the request body
    # silent=True returns None if not valid JSON
    hyperparameters = request.get_json(silent=True)

    if hyperparameters is None:
        hyperparameters = {}  # Use default hyperparameters if no JSON is provided or it's invalid

    # Validate incoming hyperparameters (optional, but good practice)
    allowed_params = ['max_features', 'alpha', 'test_size', 'random_state']
    validated_params = {}
    for k, v in hyperparameters.items():
        if k in allowed_params:
            try:
                # Type conversion for expected numerical params
                if k in ['max_features', 'random_state']:
                    validated_params[k] = int(v)
                elif k in ['alpha', 'test_size']:
                    validated_params[k] = float(v)
                else:
                    # Catch-all, though not expected for current params
                    validated_params[k] = v
            except (ValueError, TypeError):
                # Log an error or return a bad request, for now, just skip invalid types
                print(
                    f"Warning: Invalid type for hyperparameter '{k}': {v}. Using default.")
        else:
            print(f"Warning: Unknown hyperparameter '{k}' provided. Ignoring.")

    # Call the training function from utils.py with validated parameters
    training_info = train_spam_model(params=validated_params)

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
        "classification_report": report_output if report_output else None,
        # Include the MLflow run ID
        "mlflow_run_id": training_info.get('mlflow_run_id', 'N/A')
    }

    # Determine HTTP status code based on training status
    if "Error" in training_info.get('status', ''):
        return jsonify(response_data), 500  # Internal Server Error
    else:
        return jsonify(response_data), 200  # OK


if __name__ == '__main__':
    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()
    app.run(debug=True, port=8080)
