from math import atan2, degrees

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from Xlib import X, display
from Xlib.ext import xtest


class FaceTracker:
    def __init__(self, model_path):
        self.face_sample = None
        self.recognizer = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self.on_result,
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )
        )
        try:
            self.mouse_display = display.Display()
        except Exception:
            self.recognizer.close()
            raise
        self.enabled = True
        self.calibrate_requested = False
        self.toggle_requested = False
        self.center = None
        self.samples = []
        self.calibration_after = 0
        self.last_sample_time = None
        self.missing_since = None
        self.angle_history = []
        self.face_roi = None
        self.filtered = None
        self.last_time = None
        self.last_detection_time = None
        self.detection_interval = 0.05  # До 20 запросов/с, без ожидания результата.
        self.remainder = [0.0, 0.0]
        self.deadzone = (3, 5)  # Градусы поворота лица и движения подбородка.
        self.sensitivity = (20, 30)  # Пикселей/с на градус за пределами зоны.
        self.direction = (1, -1)  # Вертикаль инвертирована под управление камерой.
        self.max_speed = 700
        self.status = 'Calibrating: look straight'

    def reset_motion(self):
        self.filtered = None
        self.last_time = None
        self.remainder = [0.0, 0.0]
        self.samples.clear()
        self.angle_history.clear()

    def update(self, rgb_frame, landmarks, timestamp_ms, now):
        if self.toggle_requested:
            self.enabled = not self.enabled
            self.toggle_requested = False
            self.calibrate_requested = self.enabled
        if self.calibrate_requested:
            self.center = None
            self.calibration_after = now
            self.last_sample_time = None
            self.missing_since = None
            self.face_roi = None
            self.reset_motion()
            self.calibrate_requested = False
        if not self.enabled:
            self.reset_motion()
            self.status = 'Look paused'
            return

        if (self.last_detection_time is None
            or now - self.last_detection_time >= self.detection_interval):
            self.last_detection_time = now
            # Квадратный фрагмент из исходного кадра: без растягивания лица.
            frame = rgb_frame
            if self.face_roi is None and landmarks:
                points = [p for p in landmarks[:11] if p.visibility >= 0.5]
                if len(points) >= 3:
                    h, w = frame.shape[:2]
                    xs, ys = [p.x * w for p in points], [p.y * h for p in points]
                    size = min(min(h, w), max(80, int(max(max(xs) - min(xs), max(ys) - min(ys)) * 2.5)))
                    x = max(0, min(w - size, int((max(xs) + min(xs) - size) / 2)))
                    y = max(0, min(h - size, int((max(ys) + min(ys) - size) / 2)))
                    self.face_roi = (x, y, size)
            if self.face_roi is not None:
                x, y, size = self.face_roi
                frame = frame[y:y + size, x:x + size].copy()
            self.recognizer.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), timestamp_ms
            )
        sample = self.face_sample
        if (landmarks is None or sample is None or sample[0] is None
            or now - sample[1] > 0.25 or sample[1] < self.calibration_after):
            self.reset_motion()
            self.status = 'Waiting for face'
            if self.missing_since is None:
                self.missing_since = now
            if now - self.missing_since >= 0.8:
                # После ухода игрока новый центр определяется автоматически.
                self.center = None
                self.last_sample_time = None
                self.calibration_after = now
                self.face_roi = None
                self.missing_since = now
            return
        self.missing_since = None
        self.update_rotation(sample[0], now, sample[1])

    def on_result(self, result, image, timestamp_ms):
        # Фоновый callback только сохраняет данные, мышью управляет основной цикл.
        rotation = None
        if result.facial_transformation_matrixes:
            matrix = np.asarray(result.facial_transformation_matrixes[0])[:3, :3]
            if matrix.shape == (3, 3) and np.isfinite(matrix).all():
                rotation = self.normalize_rotation(matrix)
        self.face_sample = (rotation, timestamp_ms / 1000)

    @staticmethod
    def normalize_rotation(matrix):
        # Убираем масштаб из матрицы, оставляя только поворот.
        u, _, vt = np.linalg.svd(matrix)
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        return u @ vt

    @staticmethod
    def relative_angles(rotation, center):
        # Поворот в осях нейтральной головы, без смешивания наклона с её разворотом.
        relative = center.T @ rotation
        return (degrees(atan2(relative[0, 2], np.hypot(relative[0, 0], relative[0, 1]))),
                -degrees(atan2(relative[2, 1], relative[2, 2])))

    def update_rotation(self, rotation, now, sample_time):
        if self.center is None:
            self.status = 'Calibrating: hold head still'
            if sample_time == self.last_sample_time:
                return
            self.last_sample_time = sample_time
            if self.samples:
                # Движение во время калибровки начинает её заново.
                reference = self.samples[0][1]
                if np.linalg.norm(rotation - reference) > 0.15:
                    self.samples.clear()
            self.samples.append((sample_time, rotation))
            if len(self.samples) >= 10 and sample_time - self.samples[0][0] >= 1.0:
                self.center = self.normalize_rotation(
                    sum(matrix for _, matrix in self.samples) / len(self.samples)
                )
                self.reset_motion()
            return
        angles = self.relative_angles(rotation, self.center)
        if sample_time != self.last_sample_time:
            self.last_sample_time = sample_time
            self.angle_history.append(angles)
            self.angle_history = self.angle_history[-3:]
        if self.angle_history:
            angles = tuple(np.median(self.angle_history, axis=0))
        if self.last_time is None:
            self.filtered = angles
            self.last_time = now
            return
        dt = max(0, min(now - self.last_time, 0.1))
        self.last_time = now
        alpha = dt / (0.12 + dt)
        self.filtered = tuple(old + alpha * (value - old)
                              for old, value in zip(self.filtered, angles))
        offsets = self.filtered
        self.status = f'Head: {offsets[0]:+.1f}, {offsets[1]:+.1f} deg'
        movement = []
        for axis, offset in enumerate(offsets):
            zone = self.deadzone[axis]
            if abs(offset) <= zone:
                self.remainder[axis] = 0.0
                movement.append(0)
                continue
            offset -= zone if offset > 0 else -zone
            speed = max(-self.max_speed, min(self.max_speed,
                        offset * self.sensitivity[axis] * self.direction[axis]))
            self.remainder[axis] += speed * dt
            pixels = int(self.remainder[axis])
            self.remainder[axis] -= pixels
            movement.append(pixels)
        if any(movement):
            xtest.fake_input(self.mouse_display, X.MotionNotify, detail=1,
                             x=movement[0], y=movement[1])
            self.mouse_display.flush()

    def close(self):
        try:
            self.recognizer.close()
        finally:
            self.mouse_display.close()
