# 🎭 Multimodal Emotion Detection System

A comprehensive emotion detection system that combines **text analysis** and **facial recognition** to identify human emotions using state-of-the-art machine learning and deep learning techniques.

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python Version](https://img.shields.io/badge/Python-3.11-blue)
![Models](https://img.shields.io/badge/Models-10%20Working-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Models & Performance](#-models--performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Evaluation Results](#-evaluation-results)
- [Documentation](#-documentation)
- [Contributors](#-contributors)

---

## 🎯 Overview

This project implements a **multimodal emotion detection system** that analyzes emotions from two distinct modalities:

1. **Text Emotion Detection**: Analyzes written text using NLP and machine learning
2. **Facial Emotion Recognition**: Detects emotions from facial expressions using computer vision

The system provides a **production-ready REST API** with a web-based frontend for real-time emotion detection.

### Datasets Used

- **FER-2013**: 35,887 facial images across 7 emotion categories
- **Text Emotion Dataset**: 16,000 labeled text samples across 6 emotion categories

---

## ✨ Key Features

- ✅ **10 Working Models** (5 text + 5 facial)
- ✅ **REST API Backend** (FastAPI)
- ✅ **Interactive Web Frontend** (HTML/CSS/JavaScript)
- ✅ **Model Comparison Dashboard** (switch between models in real-time)
- ✅ **Comprehensive Evaluation** (confusion matrices, precision, recall, F1-score)
- ✅ **Real-time Webcam Detection** (optional)
- ✅ **Docker Support** (containerized deployment)
- ✅ **23 Publication-Ready Graphs** (for academic reports)

---

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Frontend  │────▶│  FastAPI     │────▶│   Models    │
│ (Web UI)    │     │  Backend     │     │  (10 total) │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼─────┐ ┌─────▼─────┐
              │   Text    │ │  Facial   │
              │  Models   │ │  Models   │
              └───────────┘ └───────────┘
```

**Components:**
- **Frontend**: HTML/CSS/JavaScript interface with model selection dropdowns
- **Backend**: FastAPI server handling requests and model inference
- **Models**: Pre-trained and custom-trained emotion classifiers
- **Evaluation**: Automated metrics generation and visualization

---

## 🤖 Models & Performance

### Text Emotion Detection (6 Emotions: joy, sadness, anger, fear, love, surprise)

| Model | Accuracy | Precision | Recall | F1-Score | Type |
|-------|----------|-----------|--------|----------|------|
| **SVM** | **85.60%** | 0.8557 | 0.8560 | 0.8522 | Traditional ML |
| **XGBoost** | 84.00% | 0.8514 | 0.8400 | 0.8409 | Traditional ML |
| **Logistic Regression** | 82.40% | 0.8292 | 0.8240 | 0.8134 | Traditional ML |
| **Naive Bayes** | 70.60% | 0.7317 | 0.7060 | 0.6546 | Traditional ML |
| **HuggingFace Transformers** | ~90%* | - | - | - | Deep Learning (DistilRoBERTa) |

*\*Estimated from pre-trained model*

### Facial Emotion Recognition (7 Emotions: angry, disgust, fear, happy, sad, surprise, neutral)

| Model | Accuracy | Precision | Recall | F1-Score | Type |
|-------|----------|-----------|--------|----------|------|
| **OpenCV CNN** | ~65-70%* | - | - | - | Pre-trained DL |
| **Random Forest** | **29.47%** | 0.2882 | 0.2947 | 0.2652 | Traditional ML (Custom-trained) |
| **XGBoost** | 29.41% | 0.2927 | 0.2941 | 0.2747 | Traditional ML (Custom-trained) |
| **Logistic Regression** | 24.05% | 0.2315 | 0.2405 | 0.2339 | Traditional ML (Custom-trained) |

*\*Estimated from pre-trained model*

**Note**: Lower facial recognition accuracy is expected and documented in literature due to:
- Class imbalance (disgust: 1.5% of dataset)
- Subtle visual differences between emotions
- Limitations of flattened pixel features for traditional ML
- Deep learning models (CNNs) achieve ~70-75% on this task

---

## 🚀 Installation

### Prerequisites

- Python 3.10-3.12 (3.11 recommended)
- conda or virtualenv
- Kaggle API credentials (for dataset download)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multimodal-emotion-detection-system.git
   cd multimodal-emotion-detection-system
   ```

2. **Create conda environment**
   ```bash
   conda create -n emotion python=3.11
   conda activate emotion
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Kaggle API** (for dataset download)
   ```bash
   # Place kaggle.json in .kaggle/ folder
   # Or run: kaggle config set api_token_location .kaggle/kaggle.json
   ```

5. **Download datasets** (optional - only needed for training)
   ```bash
   python scripts/download_datasets.py
   ```

---

## 💻 Usage

### Option 1: Run API + Frontend (Production)

```bash
# Start FastAPI server
uvicorn api.main:app --reload --port 8000

# Open in browser
# http://localhost:8000/frontend/
```

**Features:**
- Select input type (Text or Image)
- Choose model from dropdown
- Get emotion prediction with confidence scores
- View probability distribution

### Option 2: API Endpoints (Programmatic)

```python
import requests

# Text emotion detection
response = requests.post(
    "http://localhost:8000/api/text-emotion?model=svm",
    json={"text": "I am so happy today!"}
)
print(response.json())
# Output: {"emotion": "joy", "confidence": 0.92, ...}

# Facial emotion detection
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/facial-emotion?model=rf",
        files={"file": f}
    )
print(response.json())
# Output: {"emotion": "happy", "confidence": 0.68, ...}
```

### Option 3: Train Your Own Models

```bash
# Train all text models (SVM, XGBoost, LogReg, Naive Bayes)
python scripts/train_all_text_models.py

# Train all facial models (Random Forest, XGBoost, LogReg)
python scripts/train_all_facial_models.py

# Optional: Train custom DCNN (7-class facial)
python scripts/train_7_emotions.py
```

### Option 4: Generate Evaluation Metrics

```bash
# Evaluate all models and generate graphs
python scripts/generate_all_metrics.py

# Results saved in:
# - evaluation_results/text/
# - evaluation_results/facial/
# - report_graphs/
```

### Option 5: Real-time Webcam Detection

```bash
python utils/webcam_handler.py
```

---

## 📁 Project Structure

```
multimodal-emotion-detection-system/
│
├── api/                              # FastAPI Backend
│   ├── main.py                       # Main API server (loads all 10 models)
│   └── __init__.py
│
├── frontend/                         # Web Interface
│   ├── index.html                    # Main UI
│   ├── app.js                        # Frontend logic
│   └── styles.css                    # Styling
│
├── scripts/                          # Training & Evaluation
│   ├── download_datasets.py          # Download FER-2013 & text data from Kaggle
│   ├── train_all_text_models.py     # Train 4 text models (SVM, XGBoost, LogReg, NB)
│   ├── train_all_facial_models.py   # Train 3 facial models (RF, XGBoost, LogReg)
│   ├── train_7_emotions.py          # Train custom DCNN (optional)
│   ├── evaluate_text_models.py      # Generate text model metrics & confusion matrices
│   ├── evaluate_facial_models.py    # Generate facial model metrics & confusion matrices
│   ├── generate_report_graphs.py    # Create 12 template graphs
│   ├── generate_all_metrics.py      # Master script (runs all evaluations)
│   └── update_graphs_with_results.py # Update graphs with actual results
│
├── utils/                            # Utilities
│   └── webcam_handler.py             # Real-time webcam emotion detection
│
├── models/                           # Trained Models (7 .pkl files)
│   ├── text_svm.pkl                  # SVM (85.6% accuracy)
│   ├── text_xgboost.pkl              # XGBoost (84%)
│   ├── text_logreg.pkl               # Logistic Regression (82.4%)
│   ├── text_nb.pkl                   # Naive Bayes (70.6%)
│   ├── facial_rf.pkl                 # Random Forest (29.47%)
│   ├── facial_xgboost.pkl            # XGBoost (29.41%)
│   └── facial_logreg.pkl             # Logistic Regression (24.05%)
│
├── evaluation_results/               # Model Evaluation Outputs
│   ├── text/                         # Text model metrics & confusion matrices
│   └── facial/                       # Facial model metrics & confusion matrices
│
├── report_graphs/                    # Publication-Ready Graphs (23 total)
│   ├── *_ACTUAL.png                  # Graphs with actual trained model results
│   └── *.png                         # Template graphs
│
├── data/                             # Datasets (downloaded via scripts)
│   ├── train/                        # FER-2013 training images
│   ├── test/                         # FER-2013 test images
│   ├── train.txt                     # Text emotion training data
│   └── test.txt                      # Text emotion test data
│
├── .kaggle/
│   └── kaggle.json                   # Kaggle API credentials
│
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
├── .gitignore                        # Git ignore rules
├── README.md                         # This file
│
├── ABOUT_THE_PROJECT.txt             # Detailed project documentation
├── LOCAL_TEST.txt                    # Setup guide for teammates
├── REPORT_GUIDE.txt                  # 20-page academic report structure
└── FINAL_RESULTS_SUMMARY.md          # Complete results summary
```

---

## 📊 Evaluation Results

All evaluation metrics and visualizations are automatically generated and saved in structured folders.

### Available Outputs

#### Text Models (`evaluation_results/text/`)
- ✅ Confusion matrices for all 5 models
- ✅ Model comparison summary (CSV)
- ✅ Per-class precision, recall, F1-score (CSV)
- ✅ Accuracy comparison chart (PNG)

#### Facial Models (`evaluation_results/facial/`)
- ✅ Confusion matrices for all 3 models
- ✅ Model comparison summary (CSV)
- ✅ Per-class precision, recall, F1-score (CSV)
- ✅ Accuracy comparison chart (PNG)

#### Report Graphs (`report_graphs/`)
- ✅ Dataset distribution visualizations
- ✅ Model accuracy comparisons (with ACTUAL results)
- ✅ Deep Learning vs Traditional ML comparison
- ✅ Precision/Recall/F1 analysis
- ✅ Training history curves
- ✅ System architecture diagram
- ✅ All confusion matrices (publication-ready)

**Total: 23 graphs + 8 confusion matrices + 11 CSV files**

---

## 🛠️ Technologies Used

### Core Frameworks
- **Backend**: FastAPI, Uvicorn
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: TensorFlow 2.15, Keras 2.15, HuggingFace Transformers
- **Computer Vision**: OpenCV, DeepFace, PIL
- **NLP**: NLTK

### Data & Visualization
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Altair

### Deployment
- **Containerization**: Docker
- **API Documentation**: Swagger UI (auto-generated)
- **Frontend**: Vanilla JavaScript (no frameworks)

---

## 📖 Documentation

Comprehensive documentation is provided in multiple formats:

1. **[ABOUT_THE_PROJECT.txt](ABOUT_THE_PROJECT.txt)**
   - Detailed explanation of all 10 models
   - How each algorithm works
   - Dataset information
   - Preprocessing steps
   - Performance comparison

2. **[LOCAL_TEST.txt](LOCAL_TEST.txt)**
   - Step-by-step setup guide
   - Conda environment creation
   - Dataset download instructions
   - Model training commands
   - Troubleshooting section

3. **[REPORT_GUIDE.txt](REPORT_GUIDE.txt)**
   - 20-page academic report structure
   - Literature survey (10 papers)
   - Research gaps
   - Methodology with graphs
   - Results and discussion templates

4. **[FINAL_RESULTS_SUMMARY.md](FINAL_RESULTS_SUMMARY.md)**
   - Complete results breakdown
   - All model metrics
   - Graph descriptions
   - Key findings
   - Success metrics

---

## 🎓 Academic Use

This project is suitable for:
- ✅ Machine Learning course projects
- ✅ Deep Learning assignments
- ✅ Computer Vision research
- ✅ NLP applications
- ✅ Academic reports and presentations

### Key Findings for Research Papers

1. **SVM achieves 85.6% accuracy** on text emotion detection (6-class problem)
2. **Traditional ML significantly outperforms on text** (85.6%) vs facial (29.5%)
3. **Class imbalance severely impacts facial recognition** (disgust: 1.5% of data)
4. **Deep learning models show ~30% improvement** over traditional ML for facial tasks
5. **TF-IDF + SVM is highly effective** for emotion detection in short text

<!-- ---

## 🚢 Deployment

### Docker Deployment

```bash
# Build image
docker build -t emotion-detection .

# Run container
docker run -p 8000:8000 emotion-detection -->

<!-- # Access at http://localhost:8000 -->
<!-- ``` -->

<!-- ### Cloud Deployment Options

- **AWS**: Deploy using EC2 + Docker or Lambda (serverless)
- **Google Cloud**: Cloud Run (containerized)
- **Azure**: Azure Container Instances
- **Heroku**: Heroku + Docker

--- -->

## 🔬 Model Comparison Insights

### Why Text Models Perform Better?

| Factor | Text | Facial |
|--------|------|--------|
| **Feature Clarity** | Explicit semantic meaning | Subtle visual cues |
| **Preprocessing** | TF-IDF captures key patterns | Flattened pixels lose spatial info |
| **Class Balance** | Well-balanced | Severe imbalance (disgust 1.5%) |
| **Traditional ML Suitability** | High-dimensional text features | Requires spatial feature extraction |
| **Best Approach** | SVM with TF-IDF | Deep CNNs (not traditional ML) |

### Deep Learning vs Traditional ML

| Task | Best Traditional ML | Best Deep Learning | Gap |
|------|-------------------|-------------------|-----|
| Text Emotion | SVM: 85.6% | HuggingFace: ~90% | +4.4% |
| Facial Emotion | RF: 29.5% | OpenCV CNN: ~68% | +38.5% |

**Conclusion**: Deep learning provides marginal improvement for text but substantial improvement for facial recognition.

---

## 🤝 Contributors

- **[Dhruv Goyal](https://github.com/DhruvGoyal404)**
- **[Jeevant Verma](https://github.com/JeevantVerma)**

<!-- ---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- -->

## 🙏 Acknowledgments

- **FER-2013 Dataset**: Challenges in Representation Learning (ICML 2013)
- **Text Emotion Dataset**: Kaggle Community
- **HuggingFace**: Pre-trained DistilRoBERTa model
- **OpenCV**: Pre-trained emotion recognition models
- **FastAPI**: Modern web framework for building APIs

<!-- ---

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: dhruv.goyal@tiet.ac.in
- **Project Repository**: [GitHub](https://github.com/yourusername/multimodal-emotion-detection-system)
- **Issues**: [GitHub Issues](https://github.com/yourusername/multimodal-emotion-detection-system/issues)

--- -->

## 🔮 Future Enhancements

- [ ] Add audio-based emotion detection (multimodal fusion)
- [ ] Implement ensemble voting across modalities
- [ ] Real-time video emotion tracking
- [ ] Fine-tune BERT models on custom emotion datasets
- [ ] Deploy as mobile app (React Native + TensorFlow Lite)
- [ ] Add explainability (LIME/SHAP for model interpretability)
- [ ] Multi-language support for text emotion detection

<!-- ---

## ⭐ Star This Repository

If you found this project helpful, please consider giving it a star! ⭐

---

**Built with ❤️ by the Emotion Detection Team | TIET Patiala | 2024-2025** -->
