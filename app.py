import os
from flask import Flask, request, jsonify

# Import functions from our utils
from utils import load_model_and_vectorizer, train_spam_model, preprocess_text

import mlflow.tracking

app = Flask(__name__)

# Global variables to hold the trained model and vectorizer
spam_classifier = None
tfidf_vectorizer = None


@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Email Spam Detector API!",
        "endpoints": {
            "/prediction": "POST request with 'email_content' in JSON to classify text.",
            "/training": "POST request with optional JSON hyperparameters to train/retrain the model.",
            "/best_model_parameter": "GET request to view parameters of the best model from MLflow history."
        },
    })


@app.route('/training', methods=['POST'])
def training_endpoint():
    global spam_classifier, tfidf_vectorizer

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
                print(
                    f"Warning: Invalid type for '{k}': {v}. Parameter will be ignored or default used.")
        else:
            print(f"Warning: Unknown hyperparameter '{k}' provided. Ignoring.")

    training_info = train_spam_model(params=validated_params)

    # Reload the model to ensure it's the latest trained (though not necessarily "best")
    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()

    report_output = {}
    if training_info.get('report'):
        for class_name, metrics in training_info['report'].items():
            if isinstance(metrics, dict):
                report_output[class_name] = {k: float(v) if isinstance(
                    v, (float, int)) else v for k, v in metrics.items()}
            else:
                report_output[class_name] = metrics

    response_data = {
        "status": training_info.get('status'),
        "accuracy": f"{training_info['accuracy']:.4f}" if training_info.get('accuracy') else None,
        "classification_report": report_output if report_output else None,
        "mlflow_run_id": training_info.get('mlflow_run_id', 'N/A'),
        "parameters_used_in_this_run": training_info.get('used_params', None)
    }

    if "Error" in training_info.get('status', ''):
        return jsonify(response_data), 500
    else:
        return jsonify(response_data), 200


@app.route('/prediction', methods=['POST'])
def prediction_endpoint():
    global spam_classifier, tfidf_vectorizer

    if spam_classifier is None or tfidf_vectorizer is None:
        loaded_classifier, loaded_vectorizer = load_model_and_vectorizer()
        if loaded_classifier and loaded_vectorizer:
            spam_classifier = loaded_classifier
            tfidf_vectorizer = loaded_vectorizer
        else:
            return jsonify({
                "status": "error",
                "message": "Model not trained or loaded. Please train the model first via the /training endpoint."
            }), 503

    data = request.get_json()
    if not data or 'email_content' not in data:
        return jsonify({"error": "Missing 'email_content' in JSON body. Please provide it in the request body as JSON."}), 400

    email_text = data['email_content']
    # Ensure preprocess_text is imported from utils
    processed_text = preprocess_text(email_text)
    transformed_text = tfidf_vectorizer.transform([processed_text])

    prediction = spam_classifier.predict(transformed_text)[0]

    return jsonify({
        "input_email_content": email_text,
        "prediction": "spam" if prediction == 1 else "ham"
    }), 200


@app.route('/best_model_parameter', methods=['GET'])
def best_model_parameter():
    # We will query MLflow for the best run
    try:
        client = mlflow.tracking.MlflowClient()
        experiment_name = "Spam_Classification_Model_Training"

        # Get the experiment ID
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return jsonify({
                "status": "error",
                "message": f"MLflow experiment '{experiment_name}' not found. Train the model at least once."
            }), 404

        # Search for runs, ordering by the desired metric (e.g., accuracy) in descending order
        # 'metrics.accuracy' is how MLflow stores the logged accuracy
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            # Sort by accuracy, highest first
            order_by=["metrics.accuracy DESC"],
            max_results=1  # We only want the top one
        )

        if not runs:
            return jsonify({
                "status": "error",
                "message": "No runs found in MLflow experiment. Train the model first."
            }), 404

        best_run = runs[0]

        # Extract params and metrics from the best run
        best_params = best_run.data.params
        best_accuracy = best_run.data.metrics.get(
            'accuracy')  # Use .get() for safety
        best_run_id = best_run.info.run_id

        return jsonify({
            "status": "success",
            "message": "Retrieved best model parameters from MLflow.",
            "best_model_run_id": best_run_id,
            "best_model_parameters": best_params,
            "best_model_accuracy": f"{best_accuracy:.4f}" if best_accuracy is not None else "N/A"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred while fetching best parameters from MLflow: {str(e)}"
        }), 500


if __name__ == '__main__':
    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()
    app.run(port=5000, debug=True)
