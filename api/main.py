"""
FastAPI Backend for Multimodal Emotion Detection System
Endpoints for text emotion, facial emotion, and webcam streaming
With multiple model support and model selection
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops warnings

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import cv2
import joblib
import base64
import tempfile
import os
from typing import Optional, Literal
from enum import Enum
import json

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Emotion Detection API",
    description="API for detecting emotions from text and facial images with multiple model options",
    version="2.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Emotion labels and emojis
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_EMOJIS = {
    'angry': '😠',
    'disgust': '🤮',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😮',
    'neutral': '😐',
    'joy': '😂',
    'shame': '😳',
    'sadness': '😢',
    'love': '❤️',
    'anger': '😠'
}

# Global model variables - Text
text_models = {}
huggingface_classifier = None

# Global model variables - Facial
facial_models = {}

# Request/Response models
class TextEmotionRequest(BaseModel):
    text: str
    model: Optional[str] = "basic"

class EmotionResponse(BaseModel):
    emotion: str
    emoji: str
    confidence: float
    probabilities: dict
    model_used: str

class HealthResponse(BaseModel):
    status: str
    text_models_loaded: list
    facial_models_loaded: list

class ModelsResponse(BaseModel):
    text_models: list
    facial_models: list

def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def load_models():
    """Load all pre-trained models on startup"""
    global text_models, facial_models, huggingface_classifier

    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, '..', 'models')

    # ===== LOAD TEXT MODELS =====
    print("\n--- Loading Text Models ---")

    # 1. Basic model (original)
    basic_path = os.path.join(base_dir, '..', 'text_emotion.pkl')
    if os.path.exists(basic_path):
        try:
            text_models['basic'] = joblib.load(basic_path)
            print("✓ Basic (original) text model loaded")
        except Exception as e:
            print(f"✗ Error loading basic model: {e}")

    # 2. Logistic Regression
    logreg_path = os.path.join(models_dir, 'text_logreg.pkl')
    if os.path.exists(logreg_path):
        try:
            text_models['logreg'] = joblib.load(logreg_path)
            print("✓ Logistic Regression text model loaded")
        except Exception as e:
            print(f"✗ Error loading logreg model: {e}")

    # 3. SVM
    svm_path = os.path.join(models_dir, 'text_svm.pkl')
    if os.path.exists(svm_path):
        try:
            text_models['svm'] = joblib.load(svm_path)
            print("✓ SVM text model loaded")
        except Exception as e:
            print(f"✗ Error loading SVM model: {e}")

    # 4. XGBoost
    xgb_path = os.path.join(models_dir, 'text_xgboost.pkl')
    if os.path.exists(xgb_path):
        try:
            text_models['xgboost'] = joblib.load(xgb_path)
            print("✓ XGBoost text model loaded")
        except Exception as e:
            print(f"✗ Error loading XGBoost model: {e}")

    # 5. Naive Bayes
    nb_path = os.path.join(models_dir, 'text_nb.pkl')
    if os.path.exists(nb_path):
        try:
            text_models['nb'] = joblib.load(nb_path)
            print("✓ Naive Bayes text model loaded")
        except Exception as e:
            print(f"✗ Error loading NB model: {e}")

    # 6. HuggingFace (pre-trained)
    try:
        from transformers import pipeline
        huggingface_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        print("✓ HuggingFace emotion model loaded")
    except Exception as e:
        print(f"✗ HuggingFace not loaded: {e}")

    # ===== LOAD FACIAL MODELS =====
    print("\n--- Loading Facial Models ---")

    # 1. Custom CNN (Pre-trained by friend on FER-2013)
    # Architecture: 4 Conv2D blocks (128→256→512→512) + Dense layers
    # Accuracy: 97.43% (best facial model)
    cnn_path = os.path.join(models_dir, 'facial_cnn_custom.h5')
    if os.path.exists(cnn_path):
        try:
            # Try loading with keras first (model was saved with standalone keras)
            try:
                from keras.models import load_model as keras_load_model
                facial_models['custom'] = keras_load_model(cnn_path, compile=False)
                print("* Custom CNN loaded with standalone keras")
            except:
                # Fallback to tensorflow.keras
                from tensorflow.keras.models import load_model
                facial_models['custom'] = load_model(cnn_path, compile=False)
                print("* Custom CNN loaded with tensorflow.keras")

            # Recompile model for inference (avoids optimizer issues)
            facial_models['custom'].compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("* Custom CNN (7-class, pre-trained on FER-2013, 97.43% accuracy) loaded")
        except Exception as e:
            print(f"X Error loading custom CNN model: {e}")

    # 2. Logistic Regression
    logreg_facial_path = os.path.join(models_dir, 'facial_logreg.pkl')
    if os.path.exists(logreg_facial_path):
        try:
            facial_models['logreg'] = joblib.load(logreg_facial_path)
            print("✓ Logistic Regression facial model loaded")
        except Exception as e:
            print(f"✗ Error loading facial logreg: {e}")

    # 3. Random Forest
    rf_path = os.path.join(models_dir, 'facial_rf.pkl')
    if os.path.exists(rf_path):
        try:
            facial_models['rf'] = joblib.load(rf_path)
            print("✓ Random Forest facial model loaded")
        except Exception as e:
            print(f"✗ Error loading RF model: {e}")

    # 4. XGBoost
    xgb_facial_path = os.path.join(models_dir, 'facial_xgboost.pkl')
    if os.path.exists(xgb_facial_path):
        try:
            facial_models['xgboost'] = joblib.load(xgb_facial_path)
            print("✓ XGBoost facial model loaded")
        except Exception as e:
            print(f"✗ Error loading facial XGBoost: {e}")

    print("\n--- DeepFace ready (deepface, opencv backends) ---")
    print(f"\nTotal text models: {len(text_models) + (1 if huggingface_classifier else 0)}")
    print(f"Total facial models: {len(facial_models) + 2}")  # +2 for deepface backends

@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup"""
    load_models()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Multimodal Emotion Detection API v2.0",
        "docs": "/docs",
        "frontend": "/frontend/index.html",
        "endpoints": {
            "text_emotion": "POST /api/text-emotion?model=basic|logreg|svm|xgboost|nb|huggingface",
            "facial_emotion": "POST /api/facial-emotion?model=deepface|opencv|custom|logreg|rf|xgboost",
            "models": "GET /api/models",
            "health": "GET /api/health"
        }
    }

@app.get("/api/models", response_model=ModelsResponse)
async def get_available_models():
    """Get list of available models"""
    # Text models
    available_text = list(text_models.keys())
    if huggingface_classifier is not None:
        available_text.append("huggingface")

    # Facial models (DeepFace REMOVED - compatibility issues)
    available_facial = ["opencv"]  # OpenCV Haar Cascade + CNN backend
    available_facial.extend(facial_models.keys())

    return ModelsResponse(
        text_models=available_text,
        facial_models=available_facial
    )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    text_loaded = list(text_models.keys())
    if huggingface_classifier:
        text_loaded.append("huggingface")

    # Facial models (DeepFace REMOVED)
    facial_loaded = ["opencv"]
    facial_loaded.extend(facial_models.keys())

    return HealthResponse(
        status="healthy",
        text_models_loaded=text_loaded,
        facial_models_loaded=facial_loaded
    )

@app.post("/api/text-emotion", response_model=EmotionResponse)
async def analyze_text_emotion(
    request: TextEmotionRequest,
    model: str = Query("basic", description="Model: basic, logreg, svm, xgboost, nb, huggingface")
):
    """
    Analyze emotion from text input

    - **text**: The text to analyze
    - **model**: Model to use
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Override with request body model if provided
    model_to_use = request.model if request.model != "basic" else model

    try:
        # HuggingFace model
        if model_to_use == "huggingface":
            if huggingface_classifier is None:
                raise HTTPException(status_code=503, detail="HuggingFace model not loaded")

            results = huggingface_classifier(request.text)[0]
            top_result = max(results, key=lambda x: x['score'])
            emotion = top_result['label']
            confidence = float(top_result['score'])
            probabilities = {r['label']: float(r['score']) for r in results}

            return EmotionResponse(
                emotion=emotion,
                emoji=EMOTION_EMOJIS.get(emotion.lower(), '🤔'),
                confidence=confidence,
                probabilities=probabilities,
                model_used="huggingface"
            )

        # XGBoost model (special handling)
        if model_to_use == "xgboost" and "xgboost" in text_models:
            xgb_data = text_models["xgboost"]
            vectorizer = xgb_data['vectorizer']
            model_obj = xgb_data['model']
            label_encoder = xgb_data['label_encoder']

            text_vec = vectorizer.transform([request.text]).toarray()
            pred_idx = model_obj.predict(text_vec)[0]
            emotion = label_encoder.inverse_transform([pred_idx])[0]

            # Get probabilities
            proba = model_obj.predict_proba(text_vec)[0]
            probabilities = {
                label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            }
            confidence = float(max(proba))

            return EmotionResponse(
                emotion=emotion,
                emoji=EMOTION_EMOJIS.get(emotion.lower(), '🤔'),
                confidence=confidence,
                probabilities=probabilities,
                model_used="xgboost"
            )

        # Other pipeline models (basic, logreg, svm, nb)
        if model_to_use in text_models:
            model_obj = text_models[model_to_use]
            text_input = [request.text]
            prediction = model_obj.predict(text_input)[0]

            probabilities = {}
            if hasattr(model_obj, 'predict_proba'):
                proba = model_obj.predict_proba(text_input)[0]
                classes = model_obj.classes_
                probabilities = {str(cls): float(prob) for cls, prob in zip(classes, proba)}
                confidence = float(max(proba))
            else:
                confidence = 1.0
                probabilities = {prediction: 1.0}

            return EmotionResponse(
                emotion=prediction,
                emoji=EMOTION_EMOJIS.get(prediction.lower(), '🤔'),
                confidence=confidence,
                probabilities=probabilities,
                model_used=model_to_use
            )

        raise HTTPException(status_code=404, detail=f"Model '{model_to_use}' not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@app.post("/api/facial-emotion", response_model=EmotionResponse)
async def analyze_facial_emotion(
    file: UploadFile = File(...),
    model: str = Query("deepface", description="Model: deepface, opencv, custom, logreg, rf, xgboost")
):
    """
    Analyze emotion from facial image

    - **file**: Image file (JPG, PNG, JPEG)
    - **model**: Model to use
    """
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )

    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)

        # Traditional ML models (logreg, rf, xgboost)
        if model in ['logreg', 'rf', 'xgboost'] and model in facial_models:
            model_data = facial_models[model]
            model_obj = model_data['model']
            scaler = model_data.get('scaler')
            labels = model_data.get('labels', EMOTION_LABELS)

            # Preprocess image
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image")

            img = cv2.resize(img, (48, 48))
            img_flat = img.flatten().reshape(1, -1) / 255.0

            if scaler:
                img_flat = scaler.transform(img_flat)

            # Predict
            pred_idx = model_obj.predict(img_flat)[0]
            emotion = labels[pred_idx]

            # Probabilities
            if hasattr(model_obj, 'predict_proba'):
                proba = model_obj.predict_proba(img_flat)[0]
                probabilities = {labels[i]: float(proba[i]) for i in range(len(labels))}
                confidence = float(max(proba))
            else:
                confidence = 1.0
                probabilities = {emotion: 1.0}

            return EmotionResponse(
                emotion=emotion,
                emoji=EMOTION_EMOJIS.get(emotion, '🤔'),
                confidence=confidence,
                probabilities=probabilities,
                model_used=model
            )

        # Custom CNN model (pre-trained on FER-2013)
        if model == "custom" and "custom" in facial_models:
            model_obj = facial_models["custom"]

            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image")

            img = cv2.resize(img, (48, 48))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)

            predictions = model_obj.predict(img, verbose=0)[0]
            emotion_idx = int(np.argmax(predictions))
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = float(predictions[emotion_idx])

            probabilities = {
                EMOTION_LABELS[i]: float(predictions[i])
                for i in range(len(EMOTION_LABELS))
            }

            return EmotionResponse(
                emotion=emotion,
                emoji=EMOTION_EMOJIS.get(emotion, '🤔'),
                confidence=confidence,
                probabilities=probabilities,
                model_used="custom"
            )

        # OpenCV Haar Cascade + Simple CNN (no DeepFace)
        if model == 'opencv':
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Load Haar Cascade for face detection
            haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(haar_file)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                raise HTTPException(status_code=400, detail="No face detected in image")

            # Get largest face
            (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))

            # Simple emotion detection based on facial features
            # This is a basic heuristic approach since we don't have pre-trained OpenCV emotion model
            # In production, you'd use a proper pre-trained model

            # For now, use Custom CNN as fallback for OpenCV
            if 'custom' in facial_models:
                model_obj = facial_models['custom']
                img_normalized = face_roi.astype('float32') / 255.0
                img_normalized = np.expand_dims(img_normalized, axis=-1)
                img_normalized = np.expand_dims(img_normalized, axis=0)

                predictions = model_obj.predict(img_normalized, verbose=0)[0]
                emotion_idx = int(np.argmax(predictions))
                emotion = EMOTION_LABELS[emotion_idx]
                confidence = float(predictions[emotion_idx])

                probabilities = {
                    EMOTION_LABELS[i]: float(predictions[i])
                    for i in range(len(EMOTION_LABELS))
                }

                return EmotionResponse(
                    emotion=emotion,
                    emoji=EMOTION_EMOJIS.get(emotion, '🤔'),
                    confidence=confidence,
                    probabilities=probabilities,
                    model_used="opencv (using custom CNN backend)"
                )
            else:
                raise HTTPException(status_code=500, detail="OpenCV backend model not available")

        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.websocket("/api/webcam")
async def webcam_emotion_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time webcam emotion detection"""
    await websocket.accept()

    try:
        from deepface import DeepFace

        while True:
            data = await websocket.receive_text()

            try:
                frame_data = json.loads(data)
                image_data = frame_data.get('image', '')
                model = frame_data.get('model', 'opencv')

                if ',' in image_data:
                    image_data = image_data.split(',')[1]

                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"error": "Invalid image data"})
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    cv2.imwrite(temp_file.name, frame)
                    temp_path = temp_file.name

                try:
                    result = DeepFace.analyze(
                        img_path=temp_path,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend=model
                    )

                    if isinstance(result, list):
                        result = result[0]

                    emotions = result.get('emotion', {})
                    dominant_emotion = result.get('dominant_emotion', 'neutral')

                    await websocket.send_json({
                        "emotion": dominant_emotion,
                        "emoji": EMOTION_EMOJIS.get(dominant_emotion.lower(), '🤔'),
                        "confidence": float(emotions.get(dominant_emotion, 0)) / 100.0,
                        "probabilities": {k: float(v) / 100.0 for k, v in emotions.items()},
                        "face_detected": True,
                        "model_used": model
                    })

                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON data"})
            except Exception as e:
                await websocket.send_json({
                    "error": str(e),
                    "face_detected": False
                })

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

# Mount static files for frontend
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.exists(frontend_path):
    app.mount("/frontend", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
