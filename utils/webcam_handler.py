"""
Webcam Handler for Real-time Emotion Detection
Captures video from webcam and detects emotions frame by frame
"""

import cv2
import numpy as np
from pathlib import Path
import time
import threading
from queue import Queue

# Emotion labels and colors
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 128, 0),    # Green
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 255, 255),    # Yellow
    'sad': (255, 0, 0),        # Blue
    'surprise': (255, 165, 0), # Orange
    'neutral': (128, 128, 128) # Gray
}

class WebcamEmotionDetector:
    """Real-time emotion detection from webcam"""

    def __init__(self, model_path=None, use_deepface=True):
        """
        Initialize detector

        Args:
            model_path: Path to custom trained model (optional)
            use_deepface: Use DeepFace for detection (default True)
        """
        self.use_deepface = use_deepface
        self.model = None
        self.face_cascade = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)

        # Load custom model if provided
        if model_path and Path(model_path).exists():
            self._load_custom_model(model_path)
            self.use_deepface = False

        # Load face cascade for custom model
        if not self.use_deepface:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _load_custom_model(self, model_path):
        """Load custom trained Keras model"""
        from tensorflow.keras.models import load_model
        try:
            self.model = load_model(model_path)
            print(f"Custom model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading custom model: {e}")
            self.use_deepface = True

    def detect_emotion_deepface(self, frame):
        """
        Detect emotion using DeepFace

        Args:
            frame: BGR image from OpenCV

        Returns:
            dict: Emotion detection results
        """
        from deepface import DeepFace
        import tempfile
        import os

        # Save frame temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
            cv2.imwrite(f.name, frame)
            temp_path = f.name

        try:
            result = DeepFace.analyze(
                img_path=temp_path,
                actions=['emotion'],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            return {
                'emotion': result.get('dominant_emotion', 'neutral'),
                'probabilities': result.get('emotion', {}),
                'face_region': result.get('region', None)
            }

        except Exception as e:
            return {'error': str(e), 'emotion': 'unknown'}

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def detect_emotion_custom(self, frame):
        """
        Detect emotion using custom trained model

        Args:
            frame: BGR image from OpenCV

        Returns:
            dict: Emotion detection results
        """
        if self.model is None:
            return {'error': 'Model not loaded'}

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        if len(faces) == 0:
            return {'emotion': 'no_face', 'face_detected': False}

        # Process first face
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # Preprocess for model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Predict
        predictions = self.model.predict(face_roi, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = EMOTION_LABELS[emotion_idx]

        return {
            'emotion': emotion,
            'confidence': float(predictions[emotion_idx]),
            'probabilities': {
                EMOTION_LABELS[i]: float(predictions[i])
                for i in range(len(EMOTION_LABELS))
            },
            'face_region': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'face_detected': True
        }

    def draw_results(self, frame, result):
        """
        Draw emotion detection results on frame

        Args:
            frame: BGR image
            result: Detection result dict

        Returns:
            frame: Annotated frame
        """
        if 'error' in result or result.get('emotion') == 'no_face':
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        emotion = result.get('emotion', 'unknown')
        confidence = result.get('confidence', result.get('probabilities', {}).get(emotion, 0))

        if isinstance(confidence, (int, float)) and confidence > 1:
            confidence = confidence / 100.0

        # Draw face rectangle if available
        face = result.get('face_region')
        if face:
            x, y, w, h = face.get('x', 0), face.get('y', 0), face.get('w', 0), face.get('h', 0)
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Draw emotion label above face
            label = f"{emotion}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        else:
            # Draw in corner if no face region
            label = f"{emotion}: {confidence*100:.1f}%"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw emotion bar chart
        self._draw_emotion_bars(frame, result.get('probabilities', {}))

        return frame

    def _draw_emotion_bars(self, frame, probabilities):
        """Draw probability bar chart on frame"""
        if not probabilities:
            return

        h, w = frame.shape[:2]
        bar_width = 100
        bar_height = 15
        start_x = w - bar_width - 10
        start_y = 10

        for i, (emotion, prob) in enumerate(probabilities.items()):
            if isinstance(prob, (int, float)) and prob > 1:
                prob = prob / 100.0

            y = start_y + i * (bar_height + 5)
            color = EMOTION_COLORS.get(emotion, (128, 128, 128))

            # Background bar
            cv2.rectangle(frame, (start_x, y),
                          (start_x + bar_width, y + bar_height),
                          (50, 50, 50), -1)

            # Probability bar
            fill_width = int(bar_width * prob)
            cv2.rectangle(frame, (start_x, y),
                          (start_x + fill_width, y + bar_height),
                          color, -1)

            # Label
            cv2.putText(frame, f"{emotion[:3]}", (start_x - 40, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def run_webcam(self, camera_id=0, window_name="Emotion Detection"):
        """
        Run real-time webcam emotion detection

        Args:
            camera_id: Camera device ID (default 0)
            window_name: OpenCV window name
        """
        print(f"Starting webcam (camera {camera_id})...")
        print("Press 'q' to quit, 's' to save screenshot")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_running = True
        frame_count = 0
        fps_time = time.time()
        fps = 0

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame")
                    break

                # Detect emotion (process every 3rd frame for performance)
                if frame_count % 3 == 0:
                    if self.use_deepface:
                        result = self.detect_emotion_deepface(frame)
                    else:
                        result = self.detect_emotion_custom(frame)
                    last_result = result
                else:
                    result = getattr(self, 'last_result', {'emotion': 'processing'})

                self.last_result = result

                # Draw results
                frame = self.draw_results(frame, result)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_time)
                    fps_time = time.time()

                cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Show frame
                cv2.imshow(window_name, frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved: {screenshot_path}")

        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam stopped")

    def get_frame_with_emotion(self, frame):
        """
        Process a single frame and return annotated result

        Args:
            frame: BGR image

        Returns:
            annotated_frame, result_dict
        """
        if self.use_deepface:
            result = self.detect_emotion_deepface(frame)
        else:
            result = self.detect_emotion_custom(frame)

        annotated = self.draw_results(frame.copy(), result)

        return annotated, result


def main():
    """Main function to run webcam emotion detection"""
    import argparse

    parser = argparse.ArgumentParser(description='Webcam Emotion Detection')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to custom model (optional)')
    parser.add_argument('--deepface', action='store_true',
                        help='Use DeepFace instead of custom model')

    args = parser.parse_args()

    # Determine model path
    model_path = args.model
    if model_path is None:
        default_model = Path('models/facial_emotion_7class.h5')
        if default_model.exists():
            model_path = str(default_model)

    # Initialize detector
    detector = WebcamEmotionDetector(
        model_path=model_path,
        use_deepface=args.deepface or model_path is None
    )

    print("\n" + "="*50)
    print("Webcam Emotion Detection")
    print("="*50)
    print(f"Using: {'DeepFace' if detector.use_deepface else 'Custom Model'}")
    print("Controls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("="*50 + "\n")

    # Run webcam
    detector.run_webcam(camera_id=args.camera)


if __name__ == "__main__":
    main()
