"""
Evaluate Text Emotion Models
Generate confusion matrices, precision, recall, F1-score for all text models
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
import re

# Create output directories
OUTPUT_DIR = Path('evaluation_results/text')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_text(text):
    """Basic text preprocessing"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def load_test_data():
    """Load text emotion test dataset"""
    data_dir = Path('data')

    # Try test.txt first
    test_txt = data_dir / 'test.txt'
    if test_txt.exists():
        print(f"Loading test data from: {test_txt}")
        data = []
        with open(test_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ';' in line:
                    parts = line.rsplit(';', 1)
                    if len(parts) == 2:
                        text, emotion = parts
                        data.append({'text': text, 'emotion': emotion})

        if data:
            df = pd.DataFrame(data)
            df['cleaned_text'] = df['text'].apply(preprocess_text)
            df = df[df['cleaned_text'].str.len() > 0]
            print(f"Test samples: {len(df)}")
            return df['cleaned_text'].values, df['emotion'].values

    # If no test.txt, use val.txt
    val_txt = data_dir / 'val.txt'
    if val_txt.exists():
        print(f"Using validation data: {val_txt}")
        data = []
        with open(val_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ';' in line:
                    parts = line.rsplit(';', 1)
                    if len(parts) == 2:
                        text, emotion = parts
                        data.append({'text': text, 'emotion': emotion})

        if data:
            df = pd.DataFrame(data)
            df['cleaned_text'] = df['text'].apply(preprocess_text)
            df = df[df['cleaned_text'].str.len() > 0]
            print(f"Validation samples: {len(df)}")
            return df['cleaned_text'].values, df['emotion'].values

    print("ERROR: No test/validation data found")
    return None, None

def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate a single model"""
    print(f"\nEvaluating {model_name}...")

    # Handle XGBoost special case
    if model_name == "xgboost":
        vectorizer = model['vectorizer']
        model_obj = model['model']
        label_encoder = model['label_encoder']

        X_test_vec = vectorizer.transform(X_test).toarray()
        y_pred_idx = model_obj.predict(X_test_vec)
        y_pred = label_encoder.inverse_transform(y_pred_idx)

        # Convert true labels to indices and back for consistency
        y_test_encoded = label_encoder.transform(y_test)
        emotions = label_encoder.classes_
    else:
        # Pipeline models
        y_pred = model.predict(X_test)
        emotions = sorted(np.unique(list(y_test) + list(y_pred)))

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=emotions)

    # Per-class metrics
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

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

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions, ax=ax,
                cbar_kws={'label': 'Normalized Count'})

    ax.set_xlabel('Predicted Emotion', fontsize=12)
    ax.set_ylabel('True Emotion', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name.upper()}\n(Normalized)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved confusion matrix: confusion_matrix_{model_name}.png")

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
    print(f"\n✓ Saved metrics: model_comparison.csv")

    # Also save detailed per-class metrics
    for result in all_results:
        class_df = pd.DataFrame(result['class_report']).transpose()
        class_df.to_csv(OUTPUT_DIR / f'per_class_metrics_{result["model"]}.csv')

    print(f"✓ Saved per-class metrics for all models")

def create_comparison_chart(all_results):
    """Create comparison bar chart"""
    models = [r['model'].upper() for r in all_results]
    accuracies = [r['accuracy']*100 for r in all_results]

    # Sort by accuracy
    sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
    models_sorted = [x[0] for x in sorted_data]
    accuracies_sorted = [x[1] for x in sorted_data]

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#95a5a6']

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models_sorted, accuracies_sorted, color=colors[:len(models_sorted)],
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Text Emotion Model Accuracy Comparison\n(Test Set)',
                 fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved accuracy comparison chart")

def main():
    print("="*70)
    print("TEXT EMOTION MODELS EVALUATION")
    print("="*70)

    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None:
        return

    print(f"\nTest dataset loaded: {len(X_test)} samples")
    print(f"Emotions: {np.unique(y_test)}")

    # Load models
    models_dir = Path('models')
    all_results = []

    model_files = {
        'basic': Path('text_emotion.pkl'),
        'logreg': models_dir / 'text_logreg.pkl',
        'svm': models_dir / 'text_svm.pkl',
        'xgboost': models_dir / 'text_xgboost.pkl',
        'nb': models_dir / 'text_nb.pkl'
    }

    # Evaluate each model
    for model_name, model_path in model_files.items():
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                result = evaluate_model(model_name, model, X_test, y_test)
                all_results.append(result)

                # Plot confusion matrix
                plot_confusion_matrix(result['confusion_matrix'],
                                     result['emotions'],
                                     model_name)
            except Exception as e:
                print(f"  ✗ Error evaluating {model_name}: {e}")
        else:
            print(f"  ✗ Model not found: {model_path}")

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
