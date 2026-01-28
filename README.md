# Face Recognition System  
**Python + OpenCV + Mediapipe + face_recognition**

A real-time face registration and recognition system using your webcam.

- Register users by capturing multiple face samples  
- Recognize registered users live with name + confidence score

Powered by:  
- **Mediapipe** → fast face detection  
- **face_recognition** → dlib-based 128D face encodings  
- **OpenCV** → webcam access & visualization

## Features

- Register new users with 5 face samples  
- Real-time face recognition from webcam  
- Multi-user support  
- Similarity / confidence score displayed  
- Simple controls:  
  - `c` → capture sample (during registration)  
  - `q` / `ESC` → quit  
- Face encodings stored securely in `face_db.pkl`

## Tech Stack

- Python 3.8+  
- `opencv-python`  
- `mediapipe`  
- `face_recognition`  
- `numpy`  
- `pickle` (for storing encodings)

## Project Structure
face-recognition-system/
├── register_face.py       # Script to register new users
├── recognize_face.py      # Real-time recognition script
├── face_db.pkl            # Auto-generated database of face encodings
└── README.md
text## Installation

1. Clone the repository (or download ZIP)

```bash
git clone https://github.com/jyothir-369/face-recognition-system.git
cd face-recognition-system

Install required packages

Bashpip install opencv-python mediapipe face_recognition numpy
Windows users note
If the webcam doesn't open, try changing
cv2.VideoCapture(0) → cv2.VideoCapture(0, cv2.CAP_DSHOW)
in the scripts.
How to Use
1. Register a new user
Bashpython register_face.py

Enter the user's name
Look at the camera (good lighting, front face)
Press c each time to capture a sample (needs 5)
Press q or ESC to exit early

Example output:
textEnter user name: Jyothir
Press 'c' to capture face samples...
[INFO] Captured sample 1/5
[INFO] Captured sample 2/5
...
[SUCCESS] Saved face data for 'Jyothir' into face_db.pkl
2. Start real-time recognition
Bashpython recognize_face.py

Webcam feed opens
Detected faces show name + similarity score
Unknown faces labeled as "Unknown"
Press q or ESC to quit

Example display:
textJyothir (Similarity: 0.96)
Unknown (Similarity: 0.42)
Best Practices & Tips

Use good, even lighting
Face the camera directly during registration
Only one person in frame when registering
Default recognition threshold: 0.5 (change in recognize_face.py if needed)
Harmless warnings about protobuf, pkg_resources, etc. can be ignored

Database Format

File: face_db.pkl (pickle)
Structure: dictionary

Python{
    "Jyothir": numpy.ndarray(128,),    # average encoding
    "Alice":   numpy.ndarray(128,),
    ...
}
Possible Future Improvements

Recognize multiple faces in the same frame
Increase number of samples per user for better accuracy
Replace pickle with JSON / SQLite / TinyDB
Add a simple GUI (Tkinter, PyQt, Streamlit)
Implement liveness detection (blink, head movement)
Add age/gender/emotion estimation

License
Educational / personal use only.
Feel free to fork, modify, and learn from this project.
Author
Jyothir Raghavalu Bhogi
📧 jyothirraghavalu369@gmail.com
🐙 GitHub: https://github.com/jyothir-369
🔗 LinkedIn: https://www.linkedin.com/in/bhogi-jyothir-raghavalu
