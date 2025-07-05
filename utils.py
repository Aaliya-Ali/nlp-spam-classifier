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
from sklearn.metrics import accuracy_score, classification_report

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


def train_spam_model():
    """
    Trains the spam classification model and saves it.
    Returns a dictionary with training status, accuracy, and classification report.
    """
    try:
        df = pd.read_csv(DATA_PATH, encoding='latin-1')

        df = df.rename(columns={'v1': 'label', 'v2': 'sms'})
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3',
                     'Unnamed: 4'], errors='ignore')

        df['processed_text'] = df['sms'].apply(preprocess_text)

        X = df['processed_text']
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        spam_classifier = MultinomialNB()
        spam_classifier.fit(X_train_tfidf, y_train)

        y_pred = spam_classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        save_model_and_vectorizer(spam_classifier, tfidf_vectorizer)

        return {
            "status": "Model trained successfully!",
            "accuracy": accuracy,
            "report": report
        }

    except FileNotFoundError:
        return {
            "status": f"Error: Dataset not found at {DATA_PATH}. Please make sure 'train.csv' is in the 'data' directory.",
            "accuracy": None,
            "report": None
        }
    except Exception as e:
        return {
            "status": f"An error occurred during training: {e}",
            "accuracy": None,
            "report": None
        }
