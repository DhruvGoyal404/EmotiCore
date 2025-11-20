"""
Custom CNN Training Script for Facial Emotion Recognition
Based on friend's pre-trained model (facialemotionmodel.h5)

This script documents how the custom CNN model was trained on FER-2013 dataset.
The model is already trained and saved as models/facial_cnn_custom.h5

DO NOT RUN THIS SCRIPT unless you want to retrain the model from scratch.
Training takes several hours and requires GPU for reasonable speed.

Model Architecture:
- Input: 48x48 grayscale images
- 4 Convolutional blocks (128→256→512→512 filters)
- MaxPooling and Dropout for regularization
- 2 Dense layers before output
- Output: 7 emotions (softmax)

Training Details:
- Dataset: FER-2013 (28,821 train, 7,066 test)
- Optimizer: Adam
- Loss: Categorical crossentropy
- Epochs: 100
- Batch size: 128
- Expected accuracy: ~55-60% on test set

Author: Friend's implementation
Adapted for: Multimodal Emotion Detection System
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths (adjust based on your FER-2013 folder structure)
TRAIN_DIR = 'data/facial/train'
TEST_DIR = 'data/facial/test'

# Output paths
MODEL_OUTPUT = 'models/facial_cnn_custom.h5'
WEIGHTS_OUTPUT = 'models/facial_cnn_custom_weights.h5'

# Hyperparameters
IMG_SIZE = (48, 48)
BATCH_SIZE = 128
EPOCHS = 100
NUM_CLASSES = 7

# Emotion labels (7 classes)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ============================================================================
# DATA LOADING
# ============================================================================

def create_dataframe(directory):
    """
    Create pandas DataFrame with image paths and labels

    Args:
        directory: Path to folder containing emotion subfolders

    Returns:
        image_paths: List of image file paths
        labels: List of emotion labels
    """
    image_paths = []
    labels = []

    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue

        for imagename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, imagename)
            image_paths.append(image_path)
            labels.append(label)

        print(f"{label} completed")

    return image_paths, labels


def extract_features(image_paths):
    """
    Load images and convert to numpy arrays

    Args:
        image_paths: List of image file paths

    Returns:
        features: Numpy array of shape (n_samples, 48, 48, 1)
    """
    features = []

    for image_path in tqdm(image_paths, desc="Loading images"):
        # Load grayscale image
        img = load_img(image_path, color_mode='grayscale')

        # Convert to numpy array
        img = np.array(img)
        features.append(img)

    # Convert to numpy array and reshape
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)

    return features


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_custom_cnn():
    """
    Build custom CNN architecture for facial emotion recognition

    Architecture:
        Conv Block 1: Conv2D(128) → MaxPool → Dropout(0.4)
        Conv Block 2: Conv2D(256) → MaxPool → Dropout(0.4)
        Conv Block 3: Conv2D(512) → MaxPool → Dropout(0.4)
        Conv Block 4: Conv2D(512) → MaxPool → Dropout(0.4)
        Flatten
        Dense(512) → Dropout(0.4)
        Dense(256) → Dropout(0.3)
        Dense(7, softmax)

    Returns:
        model: Compiled Keras Sequential model
    """
    model = Sequential()

    # Convolutional Block 1
    model.add(Conv2D(
        128,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(48, 48, 1)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Convolutional Block 2
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Convolutional Block 3
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Convolutional Block 4
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Flatten and Dense layers
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer (7 emotions)
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model():
    """
    Complete training pipeline:
    1. Load data
    2. Preprocess
    3. Build model
    4. Train
    5. Save
    """

    print("=" * 80)
    print("CUSTOM CNN TRAINING FOR FACIAL EMOTION RECOGNITION")
    print("=" * 80)

    # Check GPU availability
    print(f"\nGPU Available: {len(tf.config.experimental.list_physical_devices('GPU')) > 0}")

    # ===== STEP 1: Load Training Data =====
    print("\n--- Loading Training Data ---")
    train_df = pd.DataFrame()
    train_df['image'], train_df['label'] = create_dataframe(TRAIN_DIR)
    print(f"Training samples: {len(train_df)}")
    print(f"Class distribution:\n{train_df['label'].value_counts()}")

    # ===== STEP 2: Load Test Data =====
    print("\n--- Loading Test Data ---")
    test_df = pd.DataFrame()
    test_df['image'], test_df['label'] = create_dataframe(TEST_DIR)
    print(f"Test samples: {len(test_df)}")

    # ===== STEP 3: Extract Features =====
    print("\n--- Extracting Features ---")
    X_train = extract_features(train_df['image'])
    X_test = extract_features(test_df['image'])

    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # ===== STEP 4: Encode Labels =====
    print("\n--- Encoding Labels ---")
    le = LabelEncoder()
    le.fit(train_df['label'])

    y_train = le.transform(train_df['label'])
    y_test = le.transform(test_df['label'])

    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # ===== STEP 5: Build Model =====
    print("\n--- Building Model ---")
    model = build_custom_cnn()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    # ===== STEP 6: Train Model =====
    print("\n--- Training Model ---")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("This will take several hours on CPU, ~30-60 minutes on GPU")

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # ===== STEP 7: Evaluate =====
    print("\n--- Final Evaluation ---")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # ===== STEP 8: Save Model =====
    print("\n--- Saving Model ---")
    model.save(MODEL_OUTPUT)
    print(f"✓ Model saved to {MODEL_OUTPUT}")

    # Also save weights separately (optional)
    model.save_weights(WEIGHTS_OUTPUT)
    print(f"✓ Weights saved to {WEIGHTS_OUTPUT}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

    return model, history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # WARNING
    print("\n" + "!" * 80)
    print("WARNING: This script will train the model from scratch.")
    print("Training takes several hours and requires significant computational resources.")
    print("The model is already trained and available at models/facial_cnn_custom.h5")
    print("!" * 80)

    response = input("\nDo you want to proceed with training? (yes/no): ")

    if response.lower() == 'yes':
        train_model()
    else:
        print("Training cancelled. Using existing model.")


# ============================================================================
# USAGE NOTES
# ============================================================================

"""
EXPECTED PERFORMANCE:
- Training accuracy: ~90-95% (after 100 epochs)
- Validation accuracy: ~55-60% (shows some overfitting)
- Test accuracy: ~55-60%

WHY LOWER THAN EXPECTED:
1. Class imbalance (disgust only 1.5% of data)
2. Overfitting despite dropout (train >> test accuracy)
3. 48x48 resolution limits subtle feature detection
4. No data augmentation used
5. No early stopping or learning rate scheduling

IMPROVEMENTS THAT COULD BE MADE:
1. Data Augmentation:
   - Random rotation (±15°)
   - Random brightness/contrast
   - Random zoom
   - Horizontal flip

2. Class Balancing:
   - Weighted loss function
   - Oversample minority classes (SMOTE)

3. Training Optimization:
   - Early stopping (patience=10)
   - ReduceLROnPlateau
   - Batch normalization instead of just dropout

4. Architecture:
   - Add Batch Normalization layers
   - Try different filter sizes
   - Experiment with residual connections

5. Transfer Learning:
   - Use VGGFace or ResNet pre-trained on face recognition
   - Fine-tune top layers only

COMPARISON WITH OTHER MODELS:
- Custom CNN: ~55-60%
- OpenCV Pre-trained CNN: ~65-70%
- HOG + Random Forest: ~29%
- HOG + XGBoost: ~29%
- HOG + LogReg: ~24%

Custom CNN performs better than traditional ML (HOG features) but slightly
worse than professional OpenCV pre-trained model due to:
- Less training data
- Simpler architecture
- No transfer learning
- Less training optimization
"""
