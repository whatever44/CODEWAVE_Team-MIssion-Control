import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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
    cv2.imshow("Hand Gesture Recognition", img)

# Open the webcam for live feed
cap = cv2.VideoCapture(1)  # Use 0 for default webcam, 1 for external webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up the Hands solution
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)
        
        # Draw hand landmarks on the original frame (BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Resize and show the current frame
        resize_and_show(frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
