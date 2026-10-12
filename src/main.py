import time
from contextlib import ExitStack
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils, PoseLandmarksConnections
from pynput.keyboard import Listener

if __package__:
    from .body_tracking import BodyTracker
    from .face_tracking import FaceTracker
else:
    from body_tracking import BodyTracker
    from face_tracking import FaceTracker


def draw_pose_landmarks(frame, pose_landmarks_list):
    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            frame,
            pose_landmarks,
            PoseLandmarksConnections.POSE_LANDMARKS,
        )

def create_body_recognizer(model_path: str):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def main():
    models = Path(__file__).resolve().parent.parent / 'models'
    # Все ресурсы освобождаются даже при ошибке создания следующего объекта.
    with ExitStack() as cleanup:
        recognizer = create_body_recognizer(str(models / 'pose_landmarker_full.task'))
        cleanup.callback(recognizer.close)
        cap = cv2.VideoCapture(0)
        cleanup.callback(cap.release)
        cleanup.callback(cv2.destroyAllWindows)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError("Ошибка: не удалось открыть камеру.")
        body_tracker = BodyTracker()
        cleanup.callback(body_tracker.release_all)
        face_tracker = FaceTracker(models / 'face_landmarker.task')
        cleanup.callback(face_tracker.close)
        held_controls = set()

        def on_press(key):
            char = getattr(key, 'char', None)
            if char in held_controls:
                return
            if char == 'c':
                face_tracker.calibrate_requested = True
            elif char == 'v':
                face_tracker.toggle_requested = True
            if char in ('c', 'v'):
                held_controls.add(char)

        def on_release(key):
            held_controls.discard(getattr(key, 'char', None))

        listener = Listener(on_press=on_press, on_release=on_release)
        cleanup.callback(listener.stop)
        listener.start()
        frame_timestamp_ms = 0
        previous_time = time.monotonic()
        fps = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр с камеры.")
                break
            capture_time = time.monotonic()
            frame_timestamp_ms = max(frame_timestamp_ms + 1, int(capture_time * 1000))
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = recognizer.detect_for_video(mp_image, frame_timestamp_ms)
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            body_tracker.frame_aspect = frame.shape[1] / frame.shape[0]
            body_tracker.update(landmarks, capture_time, world)
            now = time.monotonic()
            face_tracker.update(rgb_frame, landmarks, frame_timestamp_ms, now)
            fps = 0.8 * fps + 0.2 / max(now - previous_time, 0.001)
            previous_time = now

            if landmarks is not None:
                draw_pose_landmarks(frame, result.pose_landmarks)
            lines = ('C: center | V: look on/off | Q: quit',
                     face_tracker.status, body_tracker.hand_status,
                     f'FPS: {fps:.1f} | Move: {body_tracker.current_action or "STOP"}')
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Body Tracking', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
