"""
Training Script for 7-Emotion Facial Recognition Model
Extended from the original 3-class model to all 7 FER-2013 emotions
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    Flatten, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Constants
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 50

def load_fer2013_csv(data_path):
    """
    Load FER-2013 dataset from CSV file

    Args:
        data_path: Path to fer2013.csv

    Returns:
        X_train, X_val, y_train, y_val: Training and validation data
    """
    print("Loading FER-2013 from CSV...")

    # Find CSV file
    csv_path = None
    for possible_path in [
        Path(data_path) / 'fer2013.csv',
        Path(data_path) / 'fer2013' / 'fer2013.csv',
        Path(data_path) / 'fer2013' / 'fer2013' / 'fer2013.csv'
    ]:
        if possible_path.exists():
            csv_path = possible_path
            break

    if csv_path is None:
        raise FileNotFoundError(f"fer2013.csv not found in {data_path}")

    print(f"Found CSV at: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Display class distribution
    print("\nClass Distribution:")
    for i, emotion in enumerate(EMOTION_LABELS):
        count = len(df[df['emotion'] == i])
        print(f"  {emotion}: {count} samples")

    # Extract pixels and labels
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    X = np.stack(pixels.values)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X / 255.0  # Normalize

    y = df['emotion'].values
    y = to_categorical(y, NUM_CLASSES)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=df['emotion']
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    return X_train, X_val, y_train, y_val

def load_fer2013_folders(data_path):
    """
    Load FER-2013 from image folders (train/test structure)

    Args:
        data_path: Path to data directory containing train/test folders

    Returns:
        train_generator, val_generator: Data generators
    """
    print("Loading FER-2013 from image folders...")

    train_dir = Path(data_path) / 'train'
    test_dir = Path(data_path) / 'test'

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=0.1
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )

    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )

    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")

    return train_generator, val_generator

def build_dcnn_model():
    """
    Build Deep Convolutional Neural Network for emotion detection
    Architecture based on the original 3-class model, extended to 7 classes
    """
    model = Sequential([
        # Block 1
        Conv2D(64, (5, 5), activation='elu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(64, (5, 5), activation='elu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Block 2
        Conv2D(128, (3, 3), activation='elu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='elu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Block 3
        Conv2D(256, (3, 3), activation='elu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='elu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Block 4 - Additional block for 7 classes
        Conv2D(512, (3, 3), activation='elu', padding='same'),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='elu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Fully connected layers
        Flatten(),
        Dense(256, activation='elu'),
        BatchNormalization(),
        Dropout(0.6),
        Dense(128, activation='elu'),
        BatchNormalization(),
        Dropout(0.6),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def get_callbacks(model_path):
    """Create training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Training')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to: {save_path}")

def evaluate_model(model, X_val, y_val):
    """Evaluate model and print metrics"""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    # Predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes,
                                target_names=EMOTION_LABELS))

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)
    plt.title('Confusion Matrix - 7 Emotion Classes')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_7class.png')
    plt.close()
    print("Confusion matrix saved to: models/confusion_matrix_7class.png")

def main():
    """Main training function"""
    print("="*60)
    print("7-Emotion Facial Recognition Model Training")
    print("="*60)

    # Setup paths
    data_path = Path('data')
    models_path = Path('models')
    models_path.mkdir(exist_ok=True)

    model_save_path = str(models_path / 'facial_emotion_7class.h5')
    history_plot_path = str(models_path / 'training_history_7class.png')

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU available: {gpus}")
        # Memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\nNo GPU found, training on CPU (will be slower)")

    # Load data
    try:
        # Try CSV format first
        X_train, X_val, y_train, y_val = load_fer2013_csv(data_path)
        use_generators = False
    except FileNotFoundError:
        # Try folder format
        print("CSV not found, trying folder structure...")
        train_gen, val_gen = load_fer2013_folders(data_path)
        use_generators = True

    # Build model
    print("\nBuilding DCNN model...")
    model = build_dcnn_model()
    model.summary()

    # Calculate class weights for imbalanced data
    if not use_generators:
        y_integers = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"\nClass weights: {class_weight_dict}")
    else:
        class_weight_dict = None

    # Data augmentation for CSV data
    if not use_generators:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True
        )
        datagen.fit(X_train)

    # Callbacks
    callbacks = get_callbacks(model_save_path)

    # Train model
    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    if use_generators:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight_dict
        )
    else:
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight_dict
        )

    # Plot training history
    plot_training_history(history, history_plot_path)

    # Evaluate model
    if not use_generators:
        evaluate_model(model, X_val, y_val)

    # Final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    if not use_generators:
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
        print(f"Final Validation Loss: {val_loss:.4f}")

    print(f"\nModel saved to: {model_save_path}")
    print("\nNext steps:")
    print("1. Run the API: uvicorn api.main:app --reload")
    print("2. Test with: POST /api/facial-emotion-custom")

if __name__ == "__main__":
    main()
