import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Function to resize the image
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, int(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (int(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow("Gesture Recognition", img)

# Initialize MediaPipe Gesture Recognizer with LIVE_STREAM mode
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.GestureRecognizerOptions.LIVE_STREAM,
    num_hands=2,  # Detect up to 2 hands
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

recognizer = vision.GestureRecognizer.create_from_options(options)

# Callback function for handling recognition results
def result_callback(result, timestamp_ms):
    if result.gestures:
        top_gesture = result.gestures[0][0]  # Get the top gesture
        print(f"Gesture: {top_gesture.category_name}, Score: {top_gesture.score}")

# Open the webcam for live feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert frame to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Recognize gestures
    recognition_result = recognizer.recognize_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # Resize and show the current frame
    resize_and_show(frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
