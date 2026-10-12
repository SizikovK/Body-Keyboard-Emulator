import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils, FaceLandmarksConnections


def draw_landmarks(frame, landmarks_list):
    for landmarks in landmarks_list:
        drawing_utils.draw_landmarks(
            frame,
            landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        )

def create_recognizer(model_path: str):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)

recognizer = create_recognizer("models/face_landmarker.task")
cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms += 33

    result = recognizer.detect_for_video(mp_image, frame_timestamp_ms)

    if result.face_landmarks:
        draw_landmarks(frame, result.face_landmarks)

    cv2.imshow("Gesture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Выход.")
        break

cap.release()
cv2.destroyAllWindows()