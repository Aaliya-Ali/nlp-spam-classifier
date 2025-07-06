# utils.py
import os
import pandas as pd
import re
import string
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

# Define paths for data and models
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'train.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, 'spam_classifier.pkl')
VECTORIZER_MODEL_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- NLP Preprocessing Globals ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Cleans and preprocesses the input text.
    - Converts to lowercase
    - Removes punctuation
    - Tokenizes
    - Removes stopwords
    - Applies stemming
    """
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(
        word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)


def load_model_and_vectorizer():
    """
    Loads the trained spam classifier and TF-IDF vectorizer from disk.
    Returns (classifier, vectorizer) or (None, None) if files are not found.
    """
    try:
        classifier = joblib.load(CLASSIFIER_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_MODEL_PATH)
        print("Model and vectorizer loaded successfully.")
        return classifier, vectorizer
    except FileNotFoundError:
        print("Model or vectorizer files not found. Please train the model.")
        return None, None
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return None, None


def save_model_and_vectorizer(classifier, vectorizer):
    """
    Saves the trained spam classifier and TF-IDF vectorizer to disk.
    """
    try:
        joblib.dump(classifier, CLASSIFIER_MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_MODEL_PATH)
        print("Model and vectorizer saved successfully.")
    except Exception as e:
        print(f"Error saving model or vectorizer: {e}")


def train_spam_model(params=None):
    """
    Trains the spam classification model and saves it.
    Returns a dictionary with training status, accuracy, and classification report.
    """
    # Define default hyperparameters
    default_params = {
        'max_features': 5000,
        'alpha': 1.0,  # Smoothing parameter for Naive Bayes
        'test_size': 0.2,
        'random_state': 42
    }
    # Override defaults with provided params
    current_params = {**default_params, **(params if params else {})}

    # Set MLflow experiment name
    mlflow.set_experiment("Spam_Classification_Model_Training")

    with mlflow.start_run():  # Start an MLflow run
        try:
            # Log hyperparameters
            mlflow.log_params(current_params)

            df = pd.read_csv(DATA_PATH, encoding='latin-1')

            df = df.rename(columns={'v1': 'label', 'v2': 'sms'})
            df = df.drop(
                columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')

            df['processed_text'] = df['sms'].apply(preprocess_text)

            X = df['processed_text']
            y = df['label']

            # Split data using provided test_size and random_state
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=current_params['test_size'],
                random_state=current_params['random_state'],
                stratify=y
            )

            # Initialize and fit TF-IDF Vectorizer with max_features
            tfidf_vectorizer = TfidfVectorizer(
                max_features=current_params['max_features'])
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)

            # Initialize and train the Classifier (Multinomial Naive Bayes) with alpha
            spam_classifier = MultinomialNB(alpha=current_params['alpha'])
            spam_classifier.fit(X_train_tfidf, y_train)

            # Evaluate the model
            y_pred = spam_classifier.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)

            # Calculate and log individual metrics for both classes
            precision = precision_score(y_test, y_pred, pos_label=0)
            recall = recall_score(y_test, y_pred, pos_label=0)
            f1 = f1_score(y_test, y_pred, pos_label=0)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("spam_precision", precision)
            mlflow.log_metric("spam_recall", recall)
            mlflow.log_metric("spam_f1_score", f1)

            # Generate and log the full classification report
            report = classification_report(y_test, y_pred, output_dict=True)

            # Log metrics for 'ham' class as well
            if 'ham' in report:
                mlflow.log_metric("ham_precision", report['ham']['precision'])
                mlflow.log_metric("ham_recall", report['ham']['recall'])
                mlflow.log_metric("ham_f1_score", report['ham']['f1_score'])

            # Log the models as artifacts
            mlflow.sklearn.log_model(spam_classifier, "spam_classifier_model")
            mlflow.sklearn.log_model(tfidf_vectorizer, "tfidf_vectorizer")

            # Save models locally as well (for direct loading by Flask app)
            save_model_and_vectorizer(spam_classifier, tfidf_vectorizer)

            return {
                "status": "Model trained successfully!",
                "accuracy": accuracy,
                "report": report,
                "mlflow_run_id": mlflow.active_run().info.run_id  # NEW: Return run ID
            }

        except FileNotFoundError:
            mlflow.log_param("error", "FileNotFoundError")
            return {
                "status": f"Error: Dataset not found at {DATA_PATH}. Please make sure 'train.csv' is in the 'data' directory.",
                "accuracy": None,
                "report": None,
                "mlflow_run_id": mlflow.active_run().info.run_id
            }
        except Exception as e:
            mlflow.log_param("error", str(e))
            return {
                "status": f"An error occurred during training: {e}",
                "accuracy": None,
                "report": None,
                "mlflow_run_id": mlflow.active_run().info.run_id
            }
