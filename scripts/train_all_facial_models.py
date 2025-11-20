"""
Train All Facial Emotion Models (Traditional ML)
Trains and saves: Logistic Regression, Random Forest, XGBoost
Uses flattened pixel values from FER-2013 dataset
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Constants
IMG_SIZE = 48
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_fer2013_csv():
    """Load FER-2013 from CSV file"""
    data_dir = Path('data')

    # Find CSV file
    csv_paths = [
        data_dir / 'fer2013.csv',
        data_dir / 'fer2013' / 'fer2013.csv',
        data_dir / 'fer2013' / 'fer2013' / 'fer2013.csv'
    ]

    csv_path = None
    for path in csv_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        return None, None

    print(f"Loading from CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Extract pixels
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    X = np.stack(pixels.values)

    # Normalize
    X = X / 255.0

    y = df['emotion'].values

    return X, y

def load_fer2013_folders():
    """Load FER-2013 from image folders"""
    data_dir = Path('data')
    train_dir = data_dir / 'train'

    if not train_dir.exists():
        return None, None

    print(f"Loading from folders: {train_dir}")

    X = []
    y = []

    for emotion_idx, emotion in enumerate(EMOTION_LABELS):
        emotion_dir = train_dir / emotion
        if not emotion_dir.exists():
            continue

        images = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
        print(f"  {emotion}: {len(images)} images")

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten() / 255.0)
            y.append(emotion_idx)

    if len(X) == 0:
        return None, None

    return np.array(X), np.array(y)

def load_dataset():
    """Load FER-2013 dataset"""

    # Try CSV first
    X, y = load_fer2013_csv()

    if X is None:
        # Try folder structure
        X, y = load_fer2013_folders()

    if X is None:
        print("ERROR: FER-2013 dataset not found!")
        print("Please run: python scripts/download_datasets.py")
        sys.exit(1)

    print(f"\nDataset loaded: {len(X)} samples")
    print(f"Feature dimensions: {X.shape}")

    # Show class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  {EMOTION_LABELS[i]}: {count}")

    return X, y

def train_and_save_models(X, y):
    """Train all facial models and save them"""

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # 1. Logistic Regression
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    print("="*50)

    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        solver='lbfgs',
        multi_class='multinomial'
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)

    lr_data = {
        'model': lr_model,
        'scaler': scaler,
        'labels': EMOTION_LABELS
    }

    lr_path = models_dir / 'facial_logreg.pkl'
    joblib.dump(lr_data, lr_path)
    print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")
    print(f"Saved to: {lr_path}")
    results.append(('Logistic Regression', lr_acc))

    # 2. Random Forest
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)  # RF doesn't need scaling
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    rf_data = {
        'model': rf_model,
        'scaler': None,  # RF doesn't need scaling
        'labels': EMOTION_LABELS
    }

    rf_path = models_dir / 'facial_rf.pkl'
    joblib.dump(rf_data, rf_path)
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(f"Saved to: {rf_path}")
    results.append(('Random Forest', rf_acc))

    # 3. XGBoost
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)

    xgb_data = {
        'model': xgb_model,
        'scaler': None,
        'labels': EMOTION_LABELS
    }

    xgb_path = models_dir / 'facial_xgboost.pkl'
    joblib.dump(xgb_data, xgb_path)
    print(f"XGBoost Accuracy: {xgb_acc*100:.2f}%")
    print(f"Saved to: {xgb_path}")
    results.append(('XGBoost', xgb_acc))

    return results

def main():
    print("="*60)
    print("Training All Facial Emotion Models (Traditional ML)")
    print("="*60)
    print("\nNote: These are traditional ML models using flattened pixels.")
    print("DCNN and DeepFace will have better accuracy.\n")

    # Load dataset
    X, y = load_dataset()

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
    print("  - facial_logreg.pkl")
    print("  - facial_rf.pkl")
    print("  - facial_xgboost.pkl")

    print("\nComparison (expected):")
    print("  - DCNN: ~74% (deep learning)")
    print("  - DeepFace: ~70-80% (pre-trained)")
    print("  - Traditional ML: ~40-60%")

    print("\nAll models trained! Restart the API to use them.")

if __name__ == "__main__":
    main()
