import pickle
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# ✅ Set MLflow to use the running MLflow server (Solution 2)
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# File paths for local saving of model and vectorizer
MODEL_PATH = 'spam_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'


def train_spam_model_with_grid_search(hyperparams_grid):
    """
    Train Random Forest Classifier using Grid Search for hyperparameter tuning
    Log parameters, metrics, and model artifact to MLflow server
    """
    try:
        # Load dataset (must be present in the same directory)
        df = pd.read_csv('spam_data.csv')
        X = df['text']
        y = df['label']

        # Convert text to TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_vectorized = vectorizer.fit_transform(X)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42)

        # Set default hyperparameter grid if not provided
        if not hyperparams_grid:
            hyperparams_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }

        # Initialize model and GridSearchCV
        rf = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(estimator=rf,
                                   param_grid=hyperparams_grid,
                                   cv=3,
                                   scoring='accuracy',
                                   n_jobs=-1)

        with mlflow.start_run() as run:
            # Train model
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Evaluate
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Log parameters and metrics
            mlflow.log_params(best_params)
            mlflow.log_metric('accuracy', accuracy)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric_name}", value)

            # Log model artifact
            mlflow.sklearn.log_model(best_model, "spam_rf_model")

            # Log the dataset as artifact (optional)
            mlflow.log_artifact('spam_data.csv')

            # Save locally as well
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(best_model, f)

            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(vectorizer, f)

        return {
            'status': 'Model trained successfully with Grid Search and logged to MLflow server',
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
    Load model and vectorizer from disk if available
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer
