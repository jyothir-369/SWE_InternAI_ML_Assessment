import cv2
import face_recognition
import pickle
import os
import mediapipe as mp
import numpy as np
import time

DB_PATH = "face_db.pkl"

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Load database
def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

# Recognize faces
def recognize_user():
    db = load_database()
    if not db:
        print("[ERROR] No registered users found. Please register first.")
        return

    # Convert database to names and embeddings lists
    names = list(db.keys())
    embeddings = list(db.values())
    embeddings = [np.array(e) for e in embeddings]  # ensure all are numpy arrays

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 600)
    time.sleep(1)  # camera warm-up

    print("[INFO] Press 'q' or ESC to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read webcam.")
                break

            h, w, _ = frame.shape
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            results = face_detection.process(rgb_small)
            boxes = []

            if results.detections:
                for det in results.detections:
                    bboxC = det.location_data.relative_bounding_box
                    sh, sw, _ = small.shape
                    x1 = int(bboxC.xmin * sw)
                    y1 = int(bboxC.ymin * sh)
                    bw = int(bboxC.width * sw)
                    bh = int(bboxC.height * sh)
                    x2 = x1 + bw
                    y2 = y1 + bh
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(sw, x2), min(sh, y2)

                    # scale to full frame
                    x1_full, y1_full = int(x1 * 2), int(y1 * 2)
                    x2_full, y2_full = int(x2 * 2), int(y2 * 2)
                    boxes.append((y1_full, x2_full, y2_full, x1_full))

            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Recognize each detected face
            for box in boxes:
                enc = face_recognition.face_encodings(rgb_full, [box])
                name = "Unknown"
                if enc:
                    enc = enc[0]
                    # Compare with known embeddings
                    sims = [np.dot(e, enc) / (np.linalg.norm(e) * np.linalg.norm(enc)) for e in embeddings]
                    best_idx = np.argmax(sims)
                    if sims[best_idx] > 0.6:  # threshold for recognition
                        name = names[best_idx]

                y1, x2, y2, x1 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed by user.")
                break

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q") or key == 27:
                print("[INFO] Quit pressed.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detection.close()

if __name__ == "__main__":
    recognize_user()
