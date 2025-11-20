"""
Train All Text Emotion Models
Trains and saves: Logistic Regression, SVM, XGBoost, Naive Bayes
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def preprocess_text(text):
    """Basic text preprocessing"""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove special characters (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def load_text_dataset():
    """Load text emotion dataset"""
    data_dir = Path('data')

    # Check for train.txt first (emotion dataset format: text;emotion)
    train_txt = data_dir / 'train.txt'
    if train_txt.exists():
        print(f"Loading dataset from: {train_txt}")

        data = []
        with open(train_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ';' in line:
                    parts = line.rsplit(';', 1)  # Split from right to handle ; in text
                    if len(parts) == 2:
                        text, emotion = parts
                        data.append({'text': text, 'emotion': emotion})

        if data:
            df = pd.DataFrame(data)
            print(f"Total samples: {len(df)}")
            print(f"\nEmotion distribution:")
            print(df['emotion'].value_counts())

            # Clean data
            df['cleaned_text'] = df['text'].apply(preprocess_text)
            df = df[df['cleaned_text'].str.len() > 0]
            print(f"After cleaning: {len(df)} samples")

            return df['cleaned_text'].values, df['emotion'].values

    # Find text emotion dataset (CSV format)
    possible_files = [
        'emotion_sentimen_dataset.csv',
        'emotion.csv',
        'text_emotion.csv',
        'emotions.csv',
        'emotion_dataset.csv'
    ]

    # Search for any CSV in data directory
    csv_files = list(data_dir.glob('**/*.csv'))

    dataset_path = None
    for f in csv_files:
        if 'emotion' in f.name.lower() or 'sentiment' in f.name.lower():
            dataset_path = f
            break

    if dataset_path is None and csv_files:
        # Use first CSV found
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if 'text' in df.columns.str.lower().tolist() or 'content' in df.columns.str.lower().tolist():
                    dataset_path = f
                    break
            except:
                continue

    if dataset_path is None:
        print("ERROR: Text emotion dataset not found!")
        print("Please ensure you have a text emotion dataset in the data/ folder")
        print("Expected: train.txt (format: text;emotion) or CSV with text/emotion columns")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_path}")

    # Load dataset
    try:
        if dataset_path.suffix == '.txt':
            # Handle txt format (emotion;text)
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) >= 2:
                        data.append({'text': parts[1], 'emotion': parts[0]})
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Find text and emotion columns
    text_col = None
    emotion_col = None

    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['text', 'content', 'sentence', 'tweet', 'message']:
            text_col = col
        elif col_lower in ['emotion', 'label', 'sentiment', 'class']:
            emotion_col = col

    if text_col is None or emotion_col is None:
        print(f"Columns found: {df.columns.tolist()}")
        print("ERROR: Could not identify text and emotion columns")
        sys.exit(1)

    print(f"Using columns: text='{text_col}', emotion='{emotion_col}'")
    print(f"Total samples: {len(df)}")

    # Clean data
    df = df.dropna(subset=[text_col, emotion_col])
    df['cleaned_text'] = df[text_col].apply(preprocess_text)
    df = df[df['cleaned_text'].str.len() > 0]

    print(f"After cleaning: {len(df)} samples")
    print(f"\nEmotion distribution:")
    print(df[emotion_col].value_counts())

    return df['cleaned_text'].values, df[emotion_col].values

def train_and_save_models(X, y):
    """Train all text models and save them"""

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # TF-IDF Vectorizer (shared)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Encode labels for XGBoost
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    results = []

    # 1. Logistic Regression
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    print("="*50)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_pred = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_pred)

    # Save as pipeline
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)

    lr_path = models_dir / 'text_logreg.pkl'
    joblib.dump(lr_pipeline, lr_path)
    print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")
    print(f"Saved to: {lr_path}")
    results.append(('Logistic Regression', lr_acc))

    # 2. SVM
    print("\n" + "="*50)
    print("Training SVM...")
    print("="*50)

    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    svm_pred = svm_model.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_pred)

    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('model', SVC(kernel='linear', probability=True, random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)

    svm_path = models_dir / 'text_svm.pkl'
    joblib.dump(svm_pipeline, svm_path)
    print(f"SVM Accuracy: {svm_acc*100:.2f}%")
    print(f"Saved to: {svm_path}")
    results.append(('SVM', svm_acc))

    # 3. XGBoost
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train_tfidf.toarray(), y_train_encoded)
    xgb_pred = xgb_model.predict(X_test_tfidf.toarray())
    xgb_acc = accuracy_score(y_test_encoded, xgb_pred)

    # Save XGBoost with vectorizer and label encoder
    xgb_data = {
        'vectorizer': tfidf,
        'model': xgb_model,
        'label_encoder': le
    }

    xgb_path = models_dir / 'text_xgboost.pkl'
    joblib.dump(xgb_data, xgb_path)
    print(f"XGBoost Accuracy: {xgb_acc*100:.2f}%")
    print(f"Saved to: {xgb_path}")
    results.append(('XGBoost', xgb_acc))

    # 4. Naive Bayes
    print("\n" + "="*50)
    print("Training Naive Bayes...")
    print("="*50)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)

    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('model', MultinomialNB())
    ])
    nb_pipeline.fit(X_train, y_train)

    nb_path = models_dir / 'text_nb.pkl'
    joblib.dump(nb_pipeline, nb_path)
    print(f"Naive Bayes Accuracy: {nb_acc*100:.2f}%")
    print(f"Saved to: {nb_path}")
    results.append(('Naive Bayes', nb_acc))

    return results

def main():
    print("="*60)
    print("Training All Text Emotion Models")
    print("="*60)

    # Load dataset
    X, y = load_text_dataset()

    # Train and save models
    results = train_and_save_models(X, y)

    # Summary
    print("\n" + "="*60)
    print("Training Complete - Summary")
    print("="*60)

    print("\nModel Accuracies:")
    for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"  {name}: {acc*100:.2f}%")

    print("\nSaved models in 'models/' directory:")
    print("  - text_logreg.pkl")
    print("  - text_svm.pkl")
    print("  - text_xgboost.pkl")
    print("  - text_nb.pkl")

    print("\nNext: Run python scripts/train_all_facial_models.py")

if __name__ == "__main__":
    main()
