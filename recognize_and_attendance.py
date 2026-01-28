# recognize_and_attendance.py

import cv2
import face_recognition
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

DB_PATH = "face_db.pkl"
ATTENDANCE_FILE = "attendance.csv"

def load_database():
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

def mark_attendance(name):
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date = now.strftime("%Y-%m-%d")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Punch In", "Punch Out"])

    today = df[(df["Name"] == name) & (df["Date"] == date)]

    if today.empty:
        df.loc[len(df)] = [name, date, time, ""]
        status = "Punch In"
    elif today.iloc[-1]["Punch Out"] == "":
        df.loc[today.index[-1], "Punch Out"] = time
        status = "Punch Out"
    else:
        status = "Already completed"

    df.to_csv(ATTENDANCE_FILE, index=False)
    return status

def recognize():
    db = load_database()
    known_names = []
    known_encodings = []

    for name, enc_list in db.items():
        for enc in enc_list:
            known_names.append(name)
            known_encodings.append(enc)

    video = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding, box in zip(encodings, boxes):
            distances = face_recognition.face_distance(known_encodings, encoding)
            min_dist = np.min(distances)

            if min_dist < 0.5:
                index = np.argmin(distances)
                name = known_names[index]
                status = mark_attendance(name)
            else:
                name = "Unknown"
                status = ""

            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {status}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Attendance", frame)

        i
