import geocoder
from email.message import EmailMessage
from smtplib import SMTP_SSL
import cv2

SENDER_EMAIL = 'useexample73@gmail.com'
MAIL_PASSWORD = 'ldwh xsrc vznm lilo'  


def get_live_location():
    g = geocoder.ip('me')  
    return g.latlng


def send_sos_email(sender_email, mail_password, receiver_email, location):
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "SOS Alert"

    
    body = f"""
    <html>
    <body>
    <p>SOS Alert!</p>
    <p>My current location is:</p>
    <p>Latitude: {location[0]}</p>
    <p>Longitude: {location[1]}</p>
    </body>
    </html>
    """
    msg.add_alternative(body, subtype="html")

    with SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, mail_password)
        smtp.send_message(msg)
        print("SOS email sent successfully.")


cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from webcam.")
        break
    
    cv2.imshow("Live Feed", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('h') or key == ord('H'):
        print("Triggering SOS...")
        location = get_live_location()
        print(f"Location obtained: {location}")
        send_sos_email(SENDER_EMAIL, MAIL_PASSWORD, '03aayush10@gmail.com', location)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
