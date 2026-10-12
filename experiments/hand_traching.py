import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections


def draw_hand_landmarks(frame, hand_landmarks_list):
    for hand_landmarks in hand_landmarks_list:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            HAND_CONNECTIONS,
        )

MODEL_PATH = "../models/gesture_recognizer.task"
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,  # порог обнаружения руки (0.0–1.0)
    min_hand_presence_confidence=0.5,   # порог отслеживания руки между кадрами
    min_tracking_confidence=0.5,        # порог трекинга (если ниже — ищет руку заново)
)

recognizer = mp_vision.GestureRecognizer.create_from_options(options)
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

    result = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)

    if result.hand_landmarks:
        draw_hand_landmarks(frame, result.hand_landmarks)

    if result.gestures:
        top_gesture = result.gestures[0][0]
        label = f"{top_gesture.category_name} ({top_gesture.score:.0%})"

        cv2.putText(
            frame,
            label,
            # нижний левый угол
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            # зелёный
            (0, 255, 0),
            2,
        )

    cv2.putText(frame, "Q - quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Gesture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Выход.")
        break

cap.release()
cv2.destroyAllWindows()