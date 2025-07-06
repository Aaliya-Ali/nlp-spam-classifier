from flask import Flask, request, jsonify

# Import functions from our utils
from utils import load_model_and_vectorizer, train_spam_model

app = Flask(__name__)

# Global variables to hold the trained model, vectorizer, and training info
spam_classifier = None
tfidf_vectorizer = None
training_metadata = {}  # Store the training info including best_params


@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Email Spam Detector API!",
        "endpoints": {
            "/prediction": "POST request with 'email_content' in JSON to classify text.",
            "/training": "POST request with optional hyperparameters to train/retrain the model.",
            "/best_model_parameter": "GET request to view best parameters from training"
        }
    })


@app.route('/training', methods=['POST'])
def training_endpoint():
    global spam_classifier, tfidf_vectorizer, training_metadata

    # Accept hyperparameters from request
    hyperparameters = request.get_json(silent=True)
    if hyperparameters is None:
        hyperparameters = {}

    allowed_params = ['max_features', 'alpha', 'test_size', 'random_state']
    validated_params = {}
    for k, v in hyperparameters.items():
        if k in allowed_params:
            try:
                if k in ['max_features', 'random_state']:
                    validated_params[k] = int(v)
                elif k in ['alpha', 'test_size']:
                    validated_params[k] = float(v)
            except (ValueError, TypeError):
                print(f"Warning: Invalid type for '{k}': {v}")
        else:
            print(f"Warning: Unknown hyperparameter '{k}' provided. Ignoring.")

    # Train model
    training_info = train_spam_model(params=validated_params)
    training_metadata = training_info  # Save for future use

    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()

    # Format report for JSON
    report_output = {}
    if training_info.get('report'):
        for class_name, metrics in training_info['report'].items():
            if isinstance(metrics, dict):
                report_output[class_name] = {k: float(v) if isinstance(v, (float, int)) else v for k, v in metrics.items()}
            else:
                report_output[class_name] = metrics

    response_data = {
        "status": training_info.get('status'),
        "accuracy": f"{training_info['accuracy']:.4f}" if training_info.get('accuracy') else None,
        "classification_report": report_output if report_output else None,
        "mlflow_run_id": training_info.get('mlflow_run_id', 'N/A'),
        "best_params": training_info.get('best_params', None)
    }

    if "Error" in training_info.get('status', ''):
        return jsonify(response_data), 500
    else:
        return jsonify(response_data), 200


@app.route('/prediction', methods=['POST'])
def prediction_endpoint():
    global spam_classifier, tfidf_vectorizer

    if spam_classifier is None or tfidf_vectorizer is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 400

    data = request.get_json()
    if not data or 'email_content' not in data:
        return jsonify({"error": "Missing 'email_content' in JSON body."}), 400

    email_text = data['email_content']
    transformed_text = tfidf_vectorizer.transform([email_text])
    prediction = spam_classifier.predict(transformed_text)[0]

    return jsonify({
        "input": email_text,
        "prediction": "spam" if prediction == 1 else "ham"
    })


@app.route('/best_model_parameter', methods=['GET'])
def best_model_parameter():
    global training_metadata

    if not training_metadata or 'best_params' not in training_metadata:
        return jsonify({"error": "No best_params found. Train the model first."}), 400

    return jsonify({
        "best_params": training_metadata['best_params']
    })


if __name__ == '__main__':
    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()
    app.run(debug=True, port=5001)
