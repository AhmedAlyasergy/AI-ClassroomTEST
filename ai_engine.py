import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os

# Configuration
CSV_FILE = "emotion_log.csv"
DB_PATH = "students" 
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=['Student_ID', 'Time', 'Emotion', 'Confidence', 'Lecture_ID']).to_csv(CSV_FILE, index=False)

def open_camera(source):
    """
    Opens a camera from:
    - CAMERA_SOURCE=auto (tries 0,1,2,3)
    - CAMERA_SOURCE=<index> (e.g. 0)
    """
    if source.lower() == "auto":
        for idx in [0, 1, 2, 3]:
            cam = cv2.VideoCapture(idx)
            ok, _ = cam.read()
            if ok:
                print(f"Using camera index {idx}")
                return cam
            cam.release()
        return None

    if source.isdigit():
        cam = cv2.VideoCapture(int(source))
        ok, _ = cam.read()
        if ok:
            print(f"Using camera index {source}")
            return cam
        cam.release()
        return None

    print("Only local camera index is supported now. Set CAMERA_SOURCE to a number (e.g. 0).")
    return None

# Start camera from auto/index
cap = open_camera(CAMERA_SOURCE)
if cap is None:
    raise RuntimeError(
        "Could not open camera. Set CAMERA_SOURCE=0 (or use auto)."
    )

print("AI Engine Started. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret: break

    try:

        # 1. Identify WHO (Attendance) and WHAT (Emotion)
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
        
        for res in results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            
            # Attendance Recognition
            try:
                identities = DeepFace.find(frame[y:y+h, x:x+w], db_path=DB_PATH, enforce_detection=False, silent=True)
                name = os.path.basename(identities[0]['identity'][0]).split('.')[0] if not identities[0].empty else "Unknown"
            except:
                name = "Unknown"

            emotion = res['dominant_emotion'].capitalize()
            conf = round(res['emotion'][res['dominant_emotion']] / 100, 2)
            
            # 2. Log to CSV (Objective 2)
            new_row = [name, datetime.now().strftime("%H:%M:%S"), emotion, conf, "L1"]
            pd.DataFrame([new_row]).to_csv(CSV_FILE, mode='a', header=False, index=False)

            # Draw labels on screen
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except: pass

    cv2.imshow('AI Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
