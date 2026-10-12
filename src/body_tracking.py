from enum import StrEnum
from math import acos, degrees, hypot, sqrt
from mediapipe.tasks.python import vision as mp_vision
from pynput.keyboard import Controller, KeyCode, Key
from pynput.mouse import Controller as MouseController, Button
from Xlib import XK


class ActionStateEnum(StrEnum):
    RUNNING = "RUNNING"
    WALKING = "WALKING"


class BodyTracker:
    def __init__(self):
        self.keyboard = Controller()
        self.mouse = MouseController()
        self.punch_speed = 120  # Средняя скорость выпрямления, градусов/с.
        self.punch_timeout = 0.5
        self.shield_delay = 0.4  # Короткий удар остаётся кликом.
        self.hands = {
            button: dict(previous=None, ready=False, last_hit=None, release_at=None,
                         extended_since=None, shield_held=False, bent=None)
            for button in (Button.left, Button.right)
        }

        self.current_action: ActionStateEnum | None = None
        self.last_step_time = None
        self.last_raised_leg = None
        self.movement_missing_since = None
        self.run_threshold = 0.25
        self.walk_threshold = 0.05
        self.stop_timeout = 1.0

        self.pressed_keys = set()
        self.previous_torso = None
        self.last_jump_time = None
        self.jump_until = None
        self.jump_cooldown = 0.6
        self.jump_threshold = 1.2  # Скорость вверх в длинах торса за секунду.

        self.is_crouching = False
        self.crouch_angle = 140  # Чем больше угол, тем легче включить присед.
        self.jump_blocked_until = 0

        self.key_w = KeyCode.from_vk(XK.string_to_keysym("w"))
        self.world_landmarks = None
        self.frame_aspect = 1.0
        self.hand_status = ''

    def release_all(self):
        for button, state in self.hands.items():
            if state["release_at"] is not None:
                self.mouse.release(button)
            state.update(previous=None, ready=False, last_hit=None, release_at=None,
                         extended_since=None, shield_held=False, bent=None)
        if self.jump_until is not None:
            self.keyboard.release(Key.space)
            self.jump_until = None
        for key in self.pressed_keys:
            self.keyboard.release(key)
        self.pressed_keys.clear()

    def detect_punch(self, landmarks, now, indices, state, extension_angle=150, hand_height=0.90):
        points = [landmarks[i] for i in indices] if landmarks is not None else []
        if not points or any(point.visibility < 0.5 for point in points):
            state.update(previous=None, ready=False, last_hit=None, bent=None)
            return False
        shoulder, elbow, wrist = points
        hip = landmarks[indices[0] + 12]
        # Опущенная вдоль тела рука не считается ударом или щитом.
        if hip.visibility < 0.5 or wrist.y >= shoulder.y + (hip.y - shoulder.y) * hand_height:
            state.update(previous=None, ready=False, last_hit=None, bent=None)
            return False
        if self.world_landmarks is not None:
            shoulder, elbow, wrist = [self.world_landmarks[i] for i in indices]
            a = tuple(getattr(shoulder, axis) - getattr(elbow, axis) for axis in ('x', 'y', 'z'))
            b = tuple(getattr(wrist, axis) - getattr(elbow, axis) for axis in ('x', 'y', 'z'))
        else:
            a = ((shoulder.x - elbow.x) * self.frame_aspect, shoulder.y - elbow.y)
            b = ((wrist.x - elbow.x) * self.frame_aspect, wrist.y - elbow.y)
        length = sqrt(sum(v * v for v in a) * sum(v * v for v in b))
        if length < 1e-6:
            state.update(previous=None, ready=False, last_hit=None, bent=None)
            return False
        angle = degrees(acos(max(-1, min(1, sum(x * y for x, y in zip(a, b)) / length))))
        previous = state["previous"]
        state["previous"] = (angle, now)
        if angle < 115:
            state["ready"] = True
            state["bent"] = (angle, now)
        if previous is None or now <= previous[1] or not state["ready"]:
            return False
        if angle > extension_angle:
            state["ready"] = False
            bent_angle, bent_time = state["bent"]
            return (angle - bent_angle) / max(now - bent_time, 0.001) > self.punch_speed
        return False

    def update_hands(self, landmarks, now):
        for button, indices in ((Button.left, (11, 13, 15)),
                                (Button.right, (12, 14, 16))):
            state = self.hands[button]
            hit = self.detect_punch(landmarks, now, indices, state,
                                    extension_angle=140 if button == Button.left else 150,
                                    hand_height=0.90 if button == Button.left else 0.75)
            if button == Button.right:
                angle = state["previous"][0] if state["previous"] else 0
                threshold = 140 if state["shield_held"] else 150
                if angle > threshold:
                    if state["extended_since"] is None:
                        state["extended_since"] = now
                    if now - state["extended_since"] >= self.shield_delay:
                        if state["release_at"] is None:
                            self.mouse.press(button)
                        state["release_at"] = float("inf")
                        state["shield_held"] = True
                        continue
                else:
                    state["extended_since"] = None
                    if state["shield_held"]:
                        self.mouse.release(button)
                        state["release_at"] = None
                        state["shield_held"] = False
            if state["release_at"] is not None and (
                now >= state["release_at"] or state["previous"] is None
            ):
                self.mouse.release(button)
                state["release_at"] = None
            if hit:
                repeated = (state["last_hit"] is not None
                            and now - state["last_hit"] <= self.punch_timeout)
                if state["release_at"] is None:
                    self.mouse.press(button)
                state["release_at"] = now + (self.punch_timeout if repeated else 0.1)
                state["last_hit"] = now
        self.hand_status = ' | '.join(
            f'{name}: {state["previous"][0]:.0f} deg'
            if state["previous"] else f'{name}: arm low / not tracked'
            for name, state in (('LMB', self.hands[Button.left]),
                                ('RMB', self.hands[Button.right]))
        )

    def detect_crouch(self, landmarks):
        if landmarks is None:
            return False
        angles = []
        for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
            hip, knee, ankle = (landmarks[i] for i in (hip, knee, ankle))
            if any(point.visibility < 0.5 for point in (hip, knee, ankle)):
                return False
            a = ((hip.x - knee.x) * self.frame_aspect, hip.y - knee.y)
            b = ((ankle.x - knee.x) * self.frame_aspect, ankle.y - knee.y)
            length = hypot(*a) * hypot(*b)
            if length < 1e-6:
                return False
            cosine = (a[0] * b[0] + a[1] * b[1]) / length
            angles.append(degrees(acos(max(-1, min(1, cosine)))))
        # Небольшой запас при выпрямлении предотвращает дрожание Shift.
        threshold = self.crouch_angle + (15 if self.is_crouching else 0)
        return max(angles) < threshold

    def detect_jump(self, landmarks, now):
        if landmarks is None:
            self.previous_torso = None
            return False

        points = [landmarks[index] for index in (11, 12, 23, 24)]
        if any(point.visibility < 0.5 for point in points):
            self.previous_torso = None
            return False

        torso_y = sum(point.y for point in points) / 4
        torso_size = max(abs((points[2].y + points[3].y - points[0].y - points[1].y) / 2), 0.01)

        previous = self.previous_torso
        self.previous_torso = (torso_y, now)

        if self.is_crouching or now < self.jump_blocked_until:
            return False
        if previous is None or now <= previous[1]:
            return False

        speed = (previous[0] - torso_y) / (now - previous[1]) / torso_size

        if speed > self.jump_threshold and (
            self.last_jump_time is None
            or now - self.last_jump_time > self.jump_cooldown
        ):
            self.last_jump_time = now
            return True
        return False

    def update(self, landmarks, now, world_landmarks=None):
        self.world_landmarks = world_landmarks
        if landmarks is None:
            self.previous_torso = None
            self.last_jump_time = None
        self.update_hands(landmarks, now)
        was_crouching = self.is_crouching
        self.is_crouching = self.detect_crouch(landmarks)

        if was_crouching or self.is_crouching:
            self.jump_blocked_until = now + 0.3

        self.update_movement(landmarks, now)

        if self.jump_until is not None and (
            now >= self.jump_until or landmarks is None or self.is_crouching
        ):
            self.keyboard.release(Key.space)
            self.jump_until = None

        if self.detect_jump(landmarks, now):
            self.keyboard.press(Key.space)
            self.jump_until = now + 0.05

    def detect_movement(self, landmarks, now):
        leg_indices = (23, 24, 25, 26, 27, 28)
        if landmarks is not None and any(
            landmarks[i].visibility < 0.5 for i in leg_indices
        ):
            if self.movement_missing_since is None:
                self.movement_missing_since = now
            # Короткая потеря точек не должна требовать двух новых шагов.
            if (now - self.movement_missing_since < 0.2
                and (self.last_step_time is None
                     or now - self.last_step_time <= self.stop_timeout)):
                return
        else:
            self.movement_missing_since = None
        last_step_time = self.last_step_time
        last_raised_leg = self.last_raised_leg

        if last_step_time is not None and now - last_step_time > self.stop_timeout:
            last_step_time = None
            last_raised_leg = None
            self.current_action = None

        if landmarks is not None:
            left_hip = landmarks[mp_vision.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_vision.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_vision.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_vision.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[mp_vision.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_vision.PoseLandmark.RIGHT_ANKLE]

            if all(point.visibility >= 0.5 for point in
                   (left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle)):
                leg_length = max(
                    abs(left_ankle.y - left_hip.y),
                    abs(right_ankle.y - right_hip.y),
                    0.01,
                )
                knee_difference = ((right_knee.y - right_hip.y)
                                   - (left_knee.y - left_hip.y)) / leg_length
                raised_leg = None

                if knee_difference > self.walk_threshold:
                    raised_leg = "left"
                elif knee_difference < -self.walk_threshold:
                    raised_leg = "right"

                if raised_leg is not None and raised_leg != last_raised_leg:
                    if last_raised_leg is not None:
                        self.current_action = ActionStateEnum.WALKING

                    last_raised_leg = raised_leg
                    last_step_time = now

                if self.current_action is not None and raised_leg is not None:
                    if abs(knee_difference) >= self.run_threshold:
                        self.current_action = ActionStateEnum.RUNNING
                    else:
                        self.current_action = ActionStateEnum.WALKING

            else:
                last_step_time = None
                last_raised_leg = None
                self.current_action = None
        else:
            last_step_time = None
            last_raised_leg = None
            self.current_action = None

        self.last_step_time = last_step_time
        self.last_raised_leg = last_raised_leg

    def update_movement(self, landmarks, now):
        self.detect_movement(landmarks, now)

        desired_keys = set()

        if self.current_action == ActionStateEnum.WALKING:
            desired_keys = {self.key_w}
        elif self.current_action == ActionStateEnum.RUNNING:
            desired_keys = {self.key_w, Key.ctrl_l}

        if self.is_crouching:
            desired_keys.add(Key.shift)

        for key in self.pressed_keys - desired_keys:
            self.keyboard.release(key)

        for key in desired_keys - self.pressed_keys:
            self.keyboard.press(key)

        self.pressed_keys = desired_keys
