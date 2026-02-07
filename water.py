import cv2
import numpy as np
from playsound import playsound
import threading

ALARM_SOUND = "alarm.wav"

def play_alarm():
    playsound(ALARM_SOUND)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defect_found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 3000:
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w)

            if aspect_ratio < 1.5 or aspect_ratio > 4.0:
                defect_found = True
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
                cv2.putText(frame, "DEFECT", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    if defect_found:
        threading.Thread(target=play_alarm).start()

    cv2.imshow("Bottle Inspection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
