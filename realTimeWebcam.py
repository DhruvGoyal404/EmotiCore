"""
Real-Time Facial Emotion Detection using Webcam
Uses Custom CNN Model (97.43% accuracy) with Haar Cascade face detection

Press 'q' to quit
Press ESC (key 27) to quit

Author: Dhruv Goyal
Model: Custom CNN (4 Conv blocks, 4.2M parameters, 97.43% accuracy)
"""

import cv2
import numpy as np
from keras.models import load_model
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60)
print("REAL-TIME FACIAL EMOTION DETECTION")
print("=" * 60)
print("\nLoading model...")

# Load Custom CNN model (97.43% accuracy)
MODEL_PATH = "models/facial_cnn_custom.h5"
try:
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Parameters: {model.count_params():,}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nPlease ensure the model file exists at:", MODEL_PATH)
    exit(1)

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
print(f"✓ Haar Cascade loaded")

# Emotion labels (0-6)
labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Emotion colors (BGR format for OpenCV)
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 128, 0),    # Dark Green
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 0),      # Green
    'Neutral': (128, 128, 128),# Gray
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 255, 255)  # Yellow
}


def extract_features(image):
    """
    Preprocess face image for CNN model

    Args:
        image: 48x48 grayscale face image

    Returns:
        Normalized image ready for model prediction
    """
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def main():
    """
    Main function for real-time emotion detection
    """
    # Initialize webcam
    print("\nInitializing webcam...")
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("✗ Error: Cannot access webcam")
        print("Please check if webcam is connected and not in use by another application")
        return

    print("✓ Webcam initialized")
    print("\n" + "=" * 60)
    print("WEBCAM STARTED")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Press 's' to save screenshot")
    print("=" * 60 + "\n")

    frame_count = 0
    screenshot_count = 0

    while True:
        # Read frame from webcam
        ret, frame = webcam.read()

        if not ret:
            print("✗ Error: Failed to read from webcam")
            break

        frame_count += 1

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(48, 48)
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]

            # Resize to 48x48 (model input size)
            face_resized = cv2.resize(face_roi, (48, 48))

            # Preprocess for model
            face_features = extract_features(face_resized)

            # Predict emotion
            predictions = model.predict(face_features, verbose=0)[0]
            emotion_idx = predictions.argmax()
            emotion_label = labels[emotion_idx]
            confidence = predictions[emotion_idx] * 100

            # Get color for this emotion
            color = emotion_colors.get(emotion_label, (255, 255, 255))

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

            # Prepare text to display
            text = f"{emotion_label}: {confidence:.1f}%"

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width, y),
                color,
                -1  # Filled rectangle
            )

            # Draw emotion text
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),  # White text
                2
            )

            # Show top 3 emotions (optional - in corner)
            top3_indices = predictions.argsort()[-3:][::-1]
            y_offset = 30
            for idx in top3_indices:
                emotion_name = labels[idx]
                prob = predictions[idx] * 100
                info_text = f"{emotion_name}: {prob:.1f}%"
                cv2.putText(
                    frame,
                    info_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    emotion_colors.get(emotion_name, (255, 255, 255)),
                    2
                )
                y_offset += 25

        # Display frame info
        info_text = f"Frames: {frame_count} | Faces: {len(faces)}"
        cv2.putText(
            frame,
            info_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Show the frame
        cv2.imshow("Real-Time Emotion Detection (Custom CNN 97.43%)", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        # Quit on 'q' or ESC
        if key == ord('q') or key == 27:
            print("\nQuitting...")
            break

        # Save screenshot on 's'
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Screenshot saved: {filename}")

    # Cleanup
    print("\n" + "=" * 60)
    print(f"Session Statistics:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Screenshots saved: {screenshot_count}")
    print("=" * 60)

    webcam.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam released. Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        cv2.destroyAllWindows()
