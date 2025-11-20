"""
Evaluate Facial Emotion Models
Generate confusion matrices, precision, recall, F1-score for all facial models
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support
)
from PIL import Image

# Create output directories
OUTPUT_DIR = Path('evaluation_results/facial')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Label encoding (must match training script)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_to_num = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
num_to_emotion = {idx: emotion for idx, emotion in enumerate(EMOTION_LABELS)}

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image"""
    try:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((48, 48))
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize
        return img_array.flatten()  # Flatten to 2304 features
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def load_test_data():
    """Load facial emotion test dataset"""
    data_dir = Path('data')

    # Try test folder first
    test_dir = data_dir / 'test'
    if test_dir.exists() and test_dir.is_dir():
        print(f"Loading test data from: {test_dir}")

        X_test = []
        y_test = []

        # Expected emotion folders
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        for emotion in emotions:
            emotion_dir = test_dir / emotion
            if emotion_dir.exists():
                print(f"  Loading {emotion}...", end=' ')
                count = 0
                for img_file in emotion_dir.glob('*.jpg'):
                    img_array = load_and_preprocess_image(img_file)
                    if img_array is not None:
                        X_test.append(img_array)
                        y_test.append(emotion_to_num[emotion])  # Convert to numeric
                        count += 1
                print(f"{count} images")
            else:
                print(f"  Warning: {emotion} folder not found")

        if X_test:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            print(f"\nTotal test samples: {len(X_test)}")
            return X_test, y_test

    # If no test folder, try validation
    val_dir = data_dir / 'validation'
    if val_dir.exists() and val_dir.is_dir():
        print(f"Using validation data: {val_dir}")

        X_test = []
        y_test = []

        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        for emotion in emotions:
            emotion_dir = val_dir / emotion
            if emotion_dir.exists():
                print(f"  Loading {emotion}...", end=' ')
                count = 0
                for img_file in emotion_dir.glob('*.jpg'):
                    img_array = load_and_preprocess_image(img_file)
                    if img_array is not None:
                        X_test.append(img_array)
                        y_test.append(emotion_to_num[emotion])  # Convert to numeric
                        count += 1
                print(f"{count} images")

        if X_test:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            print(f"\nTotal validation samples: {len(X_test)}")
            return X_test, y_test

    print("ERROR: No test/validation data found")
    return None, None

def evaluate_model(model_name, model_data, X_test, y_test):
    """Evaluate a single model"""
    print(f"\nEvaluating {model_name}...")

    # Extract model and scaler
    if isinstance(model_data, dict):
        model = model_data['model']
        scaler = model_data.get('scaler', None)

        # Apply scaler if present (for LogReg)
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            y_pred_numeric = model.predict(X_test_scaled)
        else:
            # No scaling for RF and XGBoost
            y_pred_numeric = model.predict(X_test)
    else:
        # Fallback if model is not in dict format
        y_pred_numeric = model_data.predict(X_test)

    # Convert numeric predictions to string labels for display
    y_test_str = np.array([num_to_emotion[int(y)] for y in y_test])
    y_pred_str = np.array([num_to_emotion[int(y)] for y in y_pred_numeric])

    # Get all possible emotions (in string form for display)
    emotions = EMOTION_LABELS  # Use predefined order

    # Calculate metrics using string labels
    accuracy = accuracy_score(y_test_str, y_pred_str)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_str, y_pred_str, average='weighted', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test_str, y_pred_str, labels=emotions)

    # Per-class metrics
    class_report = classification_report(y_test_str, y_pred_str, output_dict=True, zero_division=0)

    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'emotions': emotions,
        'class_report': class_report
    }

    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    return results

def plot_confusion_matrix(cm, emotions, model_name):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=emotions, yticklabels=emotions, ax=ax,
                cbar_kws={'label': 'Normalized Count'})

    ax.set_xlabel('Predicted Emotion', fontsize=12)
    ax.set_ylabel('True Emotion', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name.upper()}\n(Normalized)',
                 fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  * Saved confusion matrix: confusion_matrix_{model_name}.png")

def save_metrics_csv(all_results):
    """Save all metrics to CSV"""
    summary = []

    for result in all_results:
        summary.append({
            'Model': result['model'].upper(),
            'Accuracy': f"{result['accuracy']*100:.2f}%",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1-Score': f"{result['f1_score']:.4f}"
        })

    df = pd.DataFrame(summary)
    df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    print(f"\n* Saved metrics: model_comparison.csv")

    # Also save detailed per-class metrics
    for result in all_results:
        class_df = pd.DataFrame(result['class_report']).transpose()
        class_df.to_csv(OUTPUT_DIR / f'per_class_metrics_{result["model"]}.csv')

    print(f"* Saved per-class metrics for all models")

def create_comparison_chart(all_results):
    """Create comparison bar chart"""
    models = [r['model'].upper() for r in all_results]
    accuracies = [r['accuracy']*100 for r in all_results]

    # Sort by accuracy
    sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
    models_sorted = [x[0] for x in sorted_data]
    accuracies_sorted = [x[1] for x in sorted_data]

    colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db']

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models_sorted, accuracies_sorted, color=colors[:len(models_sorted)],
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Facial Emotion Model Accuracy Comparison\n(Test Set)',
                 fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"* Saved accuracy comparison chart")

def main():
    print("="*70)
    print("FACIAL EMOTION MODELS EVALUATION")
    print("="*70)

    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None:
        return

    print(f"\nTest dataset loaded: {len(X_test)} samples")
    print(f"Emotions: {np.unique(y_test)}")
    print(f"Feature shape: {X_test.shape}")

    # Load models
    models_dir = Path('models')
    all_results = []

    model_files = {
        'logreg': models_dir / 'facial_logreg.pkl',
        'rf': models_dir / 'facial_rf.pkl',
        'xgboost': models_dir / 'facial_xgboost.pkl'
    }

    # Evaluate each model
    for model_name, model_path in model_files.items():
        if model_path.exists():
            try:
                model_data = joblib.load(model_path)
                result = evaluate_model(model_name, model_data, X_test, y_test)
                all_results.append(result)

                # Plot confusion matrix
                plot_confusion_matrix(result['confusion_matrix'],
                                     result['emotions'],
                                     model_name)
            except Exception as e:
                print(f"  X Error evaluating {model_name}: {e}")
        else:
            print(f"  X Model not found: {model_path}")

    if all_results:
        # Save metrics
        save_metrics_csv(all_results)

        # Create comparison chart
        create_comparison_chart(all_results)

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"\nResults saved in: {OUTPUT_DIR.absolute()}")

        # Print summary table
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*70)

        # Sort by accuracy
        all_results.sort(key=lambda x: x['accuracy'], reverse=True)

        for result in all_results:
            print(f"{result['model'].upper():<15} "
                  f"{result['accuracy']*100:>10.2f}%  "
                  f"{result['precision']:>10.4f}  "
                  f"{result['recall']:>10.4f}  "
                  f"{result['f1_score']:>10.4f}")

        print("\nBest Model: " + all_results[0]['model'].upper() +
              f" ({all_results[0]['accuracy']*100:.2f}%)")

if __name__ == "__main__":
    main()
