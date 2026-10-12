"""Проверки управления без камеры, X-сервера и отправки настоящих клавиш."""
import ast
from enum import StrEnum
from math import acos, atan2, cos, degrees, hypot, radians, sin, sqrt
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np


class Input:
    def __init__(self):
        self.down = set()
        self.events = []

    def press(self, key):
        self.down.add(key)
        self.events.append(('press', key))

    def release(self, key):
        self.down.discard(key)
        self.events.append(('release', key))


def load_classes(filename, namespace):
    # Выполняем настоящие методы, заменяя только аппаратные зависимости.
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'src' / filename).read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(tree, filename, 'exec'), namespace)
    return namespace


def pose(angle=90, raised=True):
    points = [NS(x=0.5, y=0.3, z=0, visibility=1) for _ in range(33)]
    for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
        points[elbow].y = 0.4
        points[wrist].x += 0.1 * sin(radians(angle))
        points[wrist].y = 0.4 + 0.1 * cos(radians(180 - angle)) if raised else 0.9
    for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
        points[hip].y, points[knee].y, points[ankle].y = 0.7, 0.85, 1
    return points


def rx(angle):
    c, s = cos(radians(angle)), sin(radians(angle))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rz(angle):
    c, s = cos(radians(angle)), sin(radians(angle))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(angle):
    c, s = cos(radians(angle)), sin(radians(angle))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class TrackingTests(unittest.TestCase):
    def setUp(self):
        names = ('LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
                 'LEFT_ANKLE', 'RIGHT_ANKLE')
        self.body_env = load_classes('body_tracking.py', dict(
            StrEnum=StrEnum, acos=acos, degrees=degrees, hypot=hypot, sqrt=sqrt,
            Controller=Input, MouseController=Input,
            Button=NS(left='L', right='R'), Key=NS(space='space', shift='shift', ctrl_l='ctrl'),
            KeyCode=NS(from_vk=lambda key: key), XK=NS(string_to_keysym=lambda key: key),
            mp_vision=NS(PoseLandmark=NS(**dict(zip(names, (23, 24, 25, 26, 27, 28))))),
        ))
        self.body = self.body_env['BodyTracker']()
        self.moves = []
        face_env = load_classes('face_tracking.py', dict(
            np=np, atan2=atan2, degrees=degrees, X=NS(MotionNotify=6),
            xtest=NS(fake_input=lambda *a, **kw: self.moves.append(kw)),
            mp=NS(Image=lambda **kw: None, ImageFormat=NS(SRGB=0)),
        ))
        cls = face_env['FaceTracker']
        self.face = cls.__new__(cls)
        self.face.__dict__.update(
            enabled=True, center=None, samples=[], filtered=None, last_time=None,
            last_sample_time=None, angle_history=[], remainder=[0., 0.],
            deadzone=(4, 5), sensitivity=(35, 30), direction=(1, -1), max_speed=700,
            calibrate_requested=False, toggle_requested=False, calibration_after=0,
            missing_since=None, face_roi=None, face_sample=None,
            last_detection_time=None, detection_interval=.05,
            mouse_display=NS(flush=lambda: None),
            recognizer=NS(detect_async=lambda *a: None),
        )

    def test_click_series_and_shield_release(self):
        b = self.body
        b.update_hands(pose(), 0)
        b.update_hands(pose(170), .15)
        self.assertEqual(b.mouse.down, {'L', 'R'})
        b.update_hands(pose(), .3)
        self.assertFalse(b.mouse.down)
        b.update_hands(pose(170), .4)
        self.assertEqual(b.hands['L']['release_at'], .9)
        b.update_hands(pose(170), .85)
        self.assertTrue(b.hands['R']['shield_held'])
        b.update_hands(pose(90, raised=False), .9)
        self.assertFalse(b.mouse.down)

    def test_slow_final_frame_still_detects_fast_extension(self):
        b = self.body
        b.update_hands(pose(), 0)
        b.update_hands(pose(149), .2)
        b.update_hands(pose(152), .25)
        self.assertEqual(b.mouse.down, {'L', 'R'})

    def test_forward_shield_uses_world_angle(self):
        b = self.body
        image = pose()
        world = pose()
        for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
            for i, z in ((shoulder, 0), (elbow, -.2), (wrist, -.4)):
                world[i] = NS(x=0, y=0, z=z, visibility=1)
        b.world_landmarks = world
        b.update_hands(image, 0)
        b.update_hands(image, .45)
        self.assertEqual(b.mouse.down, {'R'})
        b.update_hands(None, .5)
        self.assertFalse(b.mouse.down)

    def test_walking_run_and_stop(self):
        b = self.body
        p = pose(); p[25].y = .8
        b.update_movement(p, 0)
        p = pose(); p[26].y = .8
        b.update_movement(p, .2)
        self.assertEqual(b.keyboard.down, {'w'})
        p[26].y = .73
        b.update_movement(p, .3)
        self.assertEqual(b.keyboard.down, {'w', 'ctrl'})
        p[26].y = .8
        b.update_movement(p, .4)
        self.assertEqual(b.keyboard.down, {'w'})
        p[27].visibility = .1
        b.update_movement(p, .5)
        self.assertEqual(b.keyboard.down, {'w'})
        b.update_movement(None, .6)
        self.assertFalse(b.keyboard.down)

    def test_calibration_counts_unique_results_and_handles_angle_wrap(self):
        f = self.face
        for i in range(20):
            f.update_rotation(rx(179), i * .05, 0)
        self.assertIsNone(f.center)
        self.assertEqual(len(f.samples), 1)
        for i in range(1, 13):
            f.update_rotation(rx(179 if i % 2 else -179), i * .1, i * .1)
        self.assertIsNotNone(f.center)
        self.assertLess(abs(f.relative_angles(rx(179), f.center)[1]), 3)
        for i in range(13, 20):
            f.update_rotation(f.center, i * .1, i * .1)
        self.assertFalse(self.moves)
        for i in range(20, 30):
            f.update_rotation(f.center @ rx(20), i * .1, i * .1)
        self.assertGreater(self.moves[-1]['y'], 0)
        for i in range(30, 50):
            f.update_rotation(f.center @ rx(-20), i * .1, i * .1)
        self.assertLess(self.moves[-1]['y'], 0)

    def test_new_player_is_recalibrated(self):
        f = self.face
        f.center = np.eye(3)
        f.update(None, None, 1000, 1)
        f.update(None, None, 1900, 1.9)
        self.assertIsNone(f.center)
        f.face_sample = (rx(15), 2.05)
        f.update(np.zeros((720, 1280, 3), dtype=np.uint8), pose(), 2100, 2.1)
        self.assertEqual(len(f.samples), 1)
        self.assertFalse(self.moves)

    def test_pitch_uses_neutral_head_axes(self):
        center = rz(35) @ rx(15)
        roll, pitch = self.face.relative_angles(center @ rx(20), center)
        self.assertAlmostEqual(roll, 0)
        self.assertAlmostEqual(pitch, -20)

    def test_horizontal_uses_face_turn_in_both_directions(self):
        center = rz(35) @ rx(15)
        for angle in (-20, 20):
            horizontal, vertical = self.face.relative_angles(center @ ry(angle), center)
            self.assertAlmostEqual(horizontal, angle)
            self.assertAlmostEqual(vertical, 0)

    def test_clicks_allow_lower_hands(self):
        b = self.body
        image = pose()
        for i in (15, 16):
            image[i].y = .64  # Ниже старого ограничения .60, выше нового .66.
        b.world_landmarks = pose()
        b.update_hands(image, 0)
        b.world_landmarks = pose(170)
        b.update_hands(image, .15)
        self.assertEqual(b.mouse.down, {'L'})

    def test_shield_keeps_original_extension_threshold(self):
        b = self.body
        b.update_hands(pose(145), 0)
        b.update_hands(pose(145), .45)
        self.assertFalse(b.mouse.down)
        b.update_hands(pose(170), .6)
        b.update_hands(pose(170), 1.05)
        self.assertEqual(b.mouse.down, {'R'})
        b.update_hands(pose(120), 1.1)
        self.assertFalse(b.mouse.down)

    def test_jump_is_blocked_during_crouch_and_history_is_updated(self):
        b = self.body
        b.detect_jump(pose(), 0)
        raised = pose()
        for i in (11, 12, 23, 24):
            raised[i].y -= .1
        b.is_crouching = True
        self.assertFalse(b.detect_jump(raised, .1))
        self.assertEqual(b.previous_torso[1], .1)
        b.is_crouching = False
        b.jump_blocked_until = .5
        for i in (11, 12, 23, 24):
            raised[i].y -= .1
        self.assertFalse(b.detect_jump(raised, .2))
        b.detect_jump(None, .3)
        self.assertFalse(b.detect_jump(pose(), .6))

    def test_crouch_and_cleanup(self):
        b = self.body
        p = pose()
        self.assertFalse(b.detect_crouch(p))
        for i in (25, 26):
            p[i].x += .15
        self.assertTrue(b.detect_crouch(p))
        b.is_crouching = True
        b.update_movement(p, 0)
        self.assertIn('shift', b.keyboard.down)
        b.mouse.press('R')
        b.hands['R']['release_at'] = float('inf')
        b.keyboard.press('space')
        b.jump_until = 1
        b.release_all()
        self.assertFalse(b.mouse.down)
        self.assertFalse(b.keyboard.down)


if __name__ == '__main__':
    unittest.main()
