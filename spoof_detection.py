# spoof_detection.py

import cv2
import face_recognition
import numpy as np
from scipy.spatial import distance as dist

# Eye landmarks from face_recognition (68-point model)
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 2

blink_counter = 0
blink_detected = False


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def check_liveness(frame):
    """
    Returns True if blink detected, else False
    """
    global blink_counter, blink_detected

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks_list = face_recognition.face_landmarks(rgb)

    for landmarks in landmarks_list:
        left_eye = np.array(landmarks["left_eye"])
        right_eye = np.array(landmarks["right_eye"])

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                blink_detected = True
                blink_counter = 0

    return blink_detected
