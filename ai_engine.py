import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os

students_df = pd.read_csv("students.csv", encoding="utf-8-sig")
students_df.columns = students_df.columns.str.strip()

id_to_name = {
    int(row["Student ID"]): row["Student Name"]
    for _, row in students_df.iterrows()
    if not pd.isna(row["Student ID"])
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "emotion_log.csv")
DB_PATH = os.path.join(BASE_DIR, "students")
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
LECTURE_ID = "L1"

if not os.path.exists(CSV_FILE):
    pd.DataFrame(
        columns=["Student_ID", "Time", "Emotion", "Confidence", "Lecture_ID"]
    ).to_csv(CSV_FILE, index=False)

def open_camera(source):
    if source.lower() == "auto":
        for idx in [0, 1, 2, 3]:
            cam = cv2.VideoCapture(idx)
            ok, _ = cam.read()
            if ok:
                return cam
            cam.release()
        return None

    if source.isdigit():
        cam = cv2.VideoCapture(int(source))
        ok, _ = cam.read()
        if ok:
            return cam
        cam.release()

    return None

cap = open_camera(CAMERA_SOURCE)

if cap is None:
    raise RuntimeError("Could not open camera")

print("AI Engine Started")
print("Press Q to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )

        if isinstance(results, dict):
            results = [results]

        for res in results:
            x = res["region"]["x"]
            y = res["region"]["y"]
            w = res["region"]["w"]
            h = res["region"]["h"]

            try:
                identities = DeepFace.find(
                    img_path=frame,
                    db_path=DB_PATH,
                    enforce_detection=False,
                    detector_backend="opencv",
                    model_name="Facenet",
                    distance_metric="cosine",
                    silent=True
                )

                if len(identities) > 0 and not identities[0].empty:
                    best_match = identities[0].iloc[0]

                    if best_match["distance"] > 0.6:
                        name = "Not Registered"
                    else:
                        matched_path = best_match["identity"]
                        student_id = os.path.basename(matched_path).split(".")[0]

                        try:
                            student_id = int(student_id)
                            name = id_to_name.get(student_id, str(student_id))
                        except:
                            name = str(student_id)
                else:
                    name = "Not Registered"

            except Exception as e:
                print("Recognition Error:", e)
                name = "Not Registered"

            emotion = res["dominant_emotion"].capitalize()
            confidence = round(res["emotion"][res["dominant_emotion"]] / 100, 2)

            row = [[
                name,
                datetime.now().strftime("%H:%M:%S"),
                emotion,
                confidence,
                LECTURE_ID
            ]]

            pd.DataFrame(row).to_csv(
                CSV_FILE,
                mode="a",
                header=False,
                index=False
            )

            color = (0, 255, 0)
            if name == "Not Registered":
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(
                frame,
                f"{name} ({emotion} {confidence})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    except Exception as e:
        print("Analyze Error:", e)

    cv2.imshow("AI Classroom Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()