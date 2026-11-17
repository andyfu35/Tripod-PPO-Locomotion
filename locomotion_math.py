"""Pure, simulator-independent math for direction-free locomotion rewards.

No gait, symmetry, contact, energy, or action penalties are hidden here.
"""
from __future__ import annotations

import numpy as np


def signed_radial_progress_speed(start_xy, prev_xy, now_xy, dt: float) -> float:
    """Signed change in net distance from episode start, in m/s.

    Summing over an undiscounted episode telescopes to net distance / dt.
    Returning toward the start is negative; shaking in place has ~0 total.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    start = np.asarray(start_xy, dtype=np.float64)
    prev = np.asarray(prev_xy, dtype=np.float64)
    now = np.asarray(now_xy, dtype=np.float64)
    return float((np.linalg.norm(now - start) - np.linalg.norm(prev - start)) / dt)


def filter_servo_targets(current, desired, dt: float, time_constant: float,
                         max_rate) -> np.ndarray:
    """First-order target smoothing with an exact per-step angular speed bound.

    This modifies the control interface, NOT the reward or desired gait.
    """
    current = np.asarray(current, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    max_rate = np.asarray(max_rate, dtype=np.float64)
    if dt <= 0 or time_constant < 0 or np.any(max_rate <= 0):
        raise ValueError("dt > 0, time_constant >= 0, and max_rate > 0 required")
    alpha = dt / (time_constant + dt)
    increment = np.clip(alpha * (desired - current), -max_rate * dt, max_rate * dt)
    return current + increment
