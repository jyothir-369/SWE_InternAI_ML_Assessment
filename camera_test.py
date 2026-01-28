import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow fixes Windows issues

print("Opened:", cap.isOpened())

cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Test", 900, 600)

time.sleep(1)  # allow camera to warm up

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
