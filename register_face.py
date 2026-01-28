import cv2
import face_recognition
import pickle
import os
import mediapipe as mp
import time
import numpy as np

DB_PATH = "face_db.pkl"

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)


def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_database(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


def register_user(name):
    name = name.strip()
    if not name:
        print("[ERROR] Username cannot be empty!")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows fix

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "Register Face"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 600)

    time.sleep(1)

    print("[INFO] Press 'c' to capture face, 'q' or ESC to quit")
    encodings = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read webcam.")
                break

            h, w, _ = frame.shape

            # Small frame for faster detection
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            results = face_detection.process(rgb_small)
            boxes = []

            if results.detections:
                for det in results.detections:
                    bboxC = det.location_data.relative_bounding_box

                    # bbox in small frame size
                    sh, sw, _ = small.shape

                    x1 = int(bboxC.xmin * sw)
                    y1 = int(bboxC.ymin * sh)
                    bw = int(bboxC.width * sw)
                    bh = int(bboxC.height * sh)

                    x2 = x1 + bw
                    y2 = y1 + bh

                    # clamp in small frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(sw, x2), min(sh, y2)

                    # scale back to full frame
                    x1_full = int(x1 * 2)
                    y1_full = int(y1 * 2)
                    x2_full = int(x2 * 2)
                    y2_full = int(y2 * 2)

                    boxes.append((y1_full, x2_full, y2_full, x1_full))
                    cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)

            cv2.putText(frame, f"Captured: {len(encodings)}/5", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            # If user closes window manually
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed by user.")
                break

            key = cv2.waitKey(10) & 0xFF

            if key == ord("c"):
                if len(boxes) == 1:
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encoding = face_recognition.face_encodings(rgb_full, boxes)

                    if len(encoding) > 0:
                        encodings.append(encoding[0])
                        print(f"[INFO] Captured sample {len(encodings)}/5")
                    else:
                        print("[WARN] Face detected but encoding failed. Try again.")
                else:
                    print("[WARN] Make sure exactly ONE face is visible.")

                if len(encodings) == 5:
                    break

            elif key == ord("q") or key == 27:
                print("[INFO] Quit pressed.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detection.close()

    # Save if completed
    if len(encodings) == 5:
        db = load_database()
        mean_encoding = np.mean(encodings, axis=0)
        db[name] = mean_encoding
        save_database(db)
        print(f"[SUCCESS] Saved face data for '{name}' into {DB_PATH}")
    else:
        print("[INFO] Registration not completed.")


if __name__ == "__main__":
    username = input("Enter user name: ")
    register_user(username)
