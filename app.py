# app.py

from flask import Flask, request, jsonify
from utils import load_model_and_vectorizer, train_spam_model_with_grid_search

app = Flask(__name__)

# Global variables to hold the trained model and vectorizer
spam_classifier = None
tfidf_vectorizer = None

@app.route('/')
def home():
    """
    Home route to provide basic information about the API endpoints
    """
    return jsonify({
        "message": "Welcome to the Email Spam Detector API!",
        "endpoints": {
            "/predict": "POST request with 'email_content' in form data to classify text.",
            "/training": "POST request with optional hyperparameters to train/retrain the model using Grid Search and MLflow."
        }
    })

@app.route('/training', methods=['POST'])
def training_endpoint():
    """
    Endpoint to trigger model training with optional hyperparameter grid
    Logs results and artifacts to MLflow
    """
    global spam_classifier, tfidf_vectorizer

    # Get hyperparameters grid from incoming JSON request, or use empty dict
    hyperparams_grid = request.get_json() or {}

    # Call training function with hyperparameters grid
    training_info = train_spam_model_with_grid_search(hyperparams_grid)

    # Reload trained model and vectorizer into global variables
    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()

    # Prepare classification report for JSON response
    report_output = {}
    if training_info.get('report'):
        for class_name, metrics in training_info['report'].items():
            if isinstance(metrics, dict):
                report_output[class_name] = {k: float(v) if isinstance(v, (float, int)) else v for k, v in metrics.items()}
            else:
                report_output[class_name] = metrics

    # Prepare the full response data
    response_data = {
        "status": training_info.get('status'),
        "best_hyperparameters": training_info.get('best_params'),
        "accuracy": f"{training_info['accuracy']:.4f}" if training_info.get('accuracy') else None,
        "classification_report": report_output if report_output else None,
        "mlflow_run_id": training_info.get('mlflow_run_id')
    }

    # Return appropriate HTTP response based on status
    if "Error" in training_info.get('status', ''):
        return jsonify(response_data), 500
    else:
        return jsonify(response_data), 200

if __name__ == '__main__':
    # Load model and vectorizer at app startup
    spam_classifier, tfidf_vectorizer = load_model_and_vectorizer()
    app.run(debug=True)


# utils.py

import pickle
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# File paths for model and vectorizer persistence
MODEL_PATH = 'spam_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'


def train_spam_model_with_grid_search(hyperparams_grid):
    """
    Train Random Forest Classifier using Grid Search for hyperparameter tuning
    Log parameters, metrics, and model artifact to MLflow
    """
    try:
        # Load dataset (ensure 'spam_data.csv' exists with 'text' and 'label' columns)
        df = pd.read_csv('spam_data.csv')
        X = df['text']
        y = df['label']

        # Convert text to numerical feature vectors
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_vectorized = vectorizer.fit_transform(X)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        # Set default hyperparameter grid if not provided
        if not hyperparams_grid:
            hyperparams_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }

        # Initialize Random Forest Classifier
        rf = RandomForestClassifier(random_state=42)

        # Setup Grid Search with cross-validation
        grid_search = GridSearchCV(estimator=rf,
                                   param_grid=hyperparams_grid,
                                   cv=3,
                                   scoring='accuracy',
                                   n_jobs=-1)

        # Start MLflow run for experiment tracking
        with mlflow.start_run() as run:
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Make predictions and evaluate
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Log best parameters and accuracy to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metric('accuracy', accuracy)

            # Log classification report metrics to MLflow
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric_name}", value)

            # Log model artifact to MLflow
            mlflow.sklearn.log_model(best_model, "spam_rf_model")
            mlflow.log_artifact('spam_data.csv')

            # Save model and vectorizer locally
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(best_model, f)
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(vectorizer, f)

        return {
            'status': 'Model trained successfully with Grid Search and logged to MLflow',
            'accuracy': accuracy,
            'report': report,
            'best_params': best_params,
            'mlflow_run_id': run.info.run_id
        }

    except Exception as e:
        return {
            'status': f'Error during training: {str(e)}'
        }


def load_model_and_vectorizer():
    """
    Load trained model and vectorizer from disk if available
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer
