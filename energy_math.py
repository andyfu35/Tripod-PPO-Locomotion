"""Motor mechanical work proxy shared by PPO and SEGB energy experiments.

Not electrical battery energy: holding current, losses, and regeneration are unknown.
For joint-servo actuation with transmission gear 1, actuator_force is joint
motor torque (N m), and the mapped joint qvel is angular speed (rad/s).
"""
from __future__ import annotations

import numpy as np


def absolute_mechanical_power_w(torque_nm, speed_rad_s) -> float:
    torque = np.asarray(torque_nm, dtype=np.float64)
    speed = np.asarray(speed_rad_s, dtype=np.float64)
    if torque.shape != speed.shape or torque.ndim != 1:
        raise ValueError("Torque and joint speed must be matching 1-D vectors")
    if not np.all(np.isfinite(torque)) or not np.all(np.isfinite(speed)):
        raise ValueError("Motor power cannot be evaluated for nonfinite signals")
    return float(np.abs(torque * speed).sum())


def energy_penalty(energy_joules: float, duration_s: float, *,
                   weight: float, reference_power_w: float) -> float:
    if (duration_s <= 0 or reference_power_w <= 0 or weight < 0
            or not np.isfinite([energy_joules, duration_s, weight, reference_power_w]).all()
            or energy_joules < 0):
        raise ValueError("Invalid mechanical-energy penalty parameters")
    return float(weight * energy_joules / (duration_s * reference_power_w))


def energy_aware_progress(distance_m: float, energy_j: float, *,
                          horizon_s: float, weight: float,
                          reference_power_w: float) -> float:
    """Net displacement per FIXED horizon minus normalized total motor work.

    Unlike a pure negative-energy axis, zero-motion low-power policies do not
    automatically dominate genuinely walking genes on this axis.
    """
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("Distance must be a finite nonnegative net displacement")
    return distance_m / horizon_s - energy_penalty(
        energy_j, horizon_s, weight=weight, reference_power_w=reference_power_w)
