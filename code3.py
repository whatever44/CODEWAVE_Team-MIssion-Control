import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import streamlit as st
import subprocess
  # For running external scripts
import requests
from email.message import EmailMessage
from smtplib import SMTP_SSL

IPINFO_TOKEN = '0cf1b52c906317'

IP_ADDRESS = '27.34.49.83'

def get_location_from_ip(ip_address):
    url = f"https://ipinfo.io/{ip_address}?token={IPINFO_TOKEN}"
    response = requests.get(url)
    data = response.json()
    
    location = data.get('loc', '')
    city = data.get('city', 'Unknown')
    region = data.get('region', 'Unknown')
    country = data.get('country', 'Unknown')
    
    if location:
        latitude, longitude = location.split(',')
        return latitude, longitude, city, region, country
    else:
        return None, None, city, region, country

def create_maps_link(latitude, longitude):
    return f"https://www.google.com/maps?q={latitude},{longitude}&t=k"

def send_sos_email(sender_email, mail_password, receiver_email, location_info):
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "SOS Alert"

    latitude, longitude, city, region, country = location_info
    if latitude and longitude:
        maps_link = create_maps_link(latitude, longitude)
        body = f"""
        SOS Alert!
        My current location is:
        Latitude: {latitude}
        Longitude: {longitude}
        City: {city}
        Region: {region}
        Country: {country}
        Google Maps: {maps_link}
        """
    else:
        body = f"""
        SOS Alert!
        Unable to fetch the exact location.
        City: {city}
        Region: {region}
        Country: {country}
        """

    msg.set_content(body)

    with SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, mail_password)
        smtp.send_message(msg)
        print("SOS email sent successfully.")

SENDER_EMAIL = 'useexample73@gmail.com'
MAIL_PASSWORD = 'ldwh xsrc vznm lilo'
    
location_info = get_location_from_ip(IP_ADDRESS)

# Load pre-trained model
model = load_model('action.h5')

# Define the action labels
actions = np.array(['sos', 'notsos'])

# Initialize Mediapipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Helper functions for landmark detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# Streamlit frontend layout
st.title("Gesture Recognition with Streamlit")

# Webcam feed display
FRAME_WINDOW = st.image([])

# Initialize sequence variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

# Start the webcam feed using OpenCV and Mediapipe
cap = cv2.VideoCapture(1)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Make detections and draw landmarks
        image, results = mediapipe_detection(frame, holistic)
        # draw_styled_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-40:]  # Only keep the last 40 frames

        if len(sequence) == 40:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            # Check for prediction stability
            if np.unique(predictions[-10:])[0] == np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Check if the model predicts 'sos'
            if sentence and sentence[-1] == 'sos':
                # Run the external script code2.py
                send_sos_email(SENDER_EMAIL, MAIL_PASSWORD, '03aayush10@gmail.com', location_info)  # Make sure the path to code2.py is correct
                
                cap.release()
                cv2.destroyAllWindows()
                # Display 'SOS' in red text
                break
             # Display for 1 second

            else:
                # Display other actions or nothing
                cv2.putText(image, ' '.join(sentence), (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the image in Streamlit app
        FRAME_WINDOW.image(image)

        # Break loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
