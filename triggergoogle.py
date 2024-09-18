import requests
from email.message import EmailMessage
from smtplib import SMTP_SSL
import cv2

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

cap = cv2.VideoCapture(1)  

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
        location_info = get_location_from_ip(IP_ADDRESS)
        print(f"Location obtained: {location_info}")
        send_sos_email(SENDER_EMAIL, MAIL_PASSWORD, 'sitasmathapa99@gmail.com', location_info)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
