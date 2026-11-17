"""Numerical tests that run without MuJoCo/Gym/SB3."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from locomotion_math import signed_radial_progress_speed, filter_servo_targets


class TestLocomotionMath(unittest.TestCase):
    def test_back_and_forth_has_zero_accumulated_progress(self):
        positions = [(0, 0), (0.1, 0), (0.2, 0), (0.1, 0), (0, 0)]
        score = sum(signed_radial_progress_speed(positions[0], prev, now, 0.02)
                    for prev, now in zip(positions[:-1], positions[1:]))
        self.assertAlmostEqual(score, 0)

    def test_direction_free_progress_and_retreat(self):
        self.assertAlmostEqual(signed_radial_progress_speed((0, 0), (0, 0), (0, 0.02), 0.02), 1)
        self.assertAlmostEqual(signed_radial_progress_speed((0, 0), (0, -0.02), (0, 0), 0.02), -1)

    def test_target_filter_limits_rate_and_moves_toward_goal(self):
        now = np.zeros(12)
        desired = np.full(12, 3.)
        new = filter_servo_targets(now, desired, dt=.02, time_constant=.10, max_rate=2.5)
        self.assertTrue(np.all(new > now))
        self.assertTrue(np.all(new - now <= .05 + 1e-12))

    def test_fast_toggle_not_full_range_jump(self):
        now = np.zeros(12)
        target_1 = filter_servo_targets(now, np.ones(12), .02, .10, 2.5)
        target_2 = filter_servo_targets(target_1, -np.ones(12), .02, .10, 2.5)
        self.assertLess(np.max(np.abs(target_2 - target_1)), .0500001)


if __name__ == '__main__':
    unittest.main()
