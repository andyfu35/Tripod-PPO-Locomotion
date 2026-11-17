"""Three-legged MuJoCo locomotion with an optional motor-work penalty.

Reward: signed radial speed + 0.2 alive - weighted absolute motor power.
No prescribed gait, forward-heading command, action penalty, foot-contact
reward, gait bonus, target posture reward, or movement symmetry reward.

Control: 12 desired joint positions, with servo-target lowpass/rate limiting.
This avoids teleporting servo references every 20 ms; the physical robot
still follows the torque-limited MuJoCo position actuators.
"""
from __future__ import annotations

from pathlib import Path
import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from generate_model import MODEL_PATH, generate_model
from locomotion_math import signed_radial_progress_speed, filter_servo_targets
from energy_math import absolute_mechanical_power_w, energy_penalty


class TripodEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1000, frame_skip: int = 10,
                 speed_weight: float = 1.0, alive_bonus: float = 0.2,
                 action_tau: float = 0.10, max_target_rate: float = 2.5,
                 energy_weight: float = 0.0, energy_power_ref_w: float = 50.0,
                 model_path: str | Path | None = None):
        super().__init__()
        xml_path = Path(model_path) if model_path is not None else MODEL_PATH
        if not xml_path.is_file() and model_path is None:
            generate_model(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        self.dt = float(self.model.opt.timestep) * self.frame_skip
        self.max_episode_steps = int(max_episode_steps)
        self.speed_weight = float(speed_weight)
        self.alive_bonus = float(alive_bonus)
        self.action_tau = float(action_tau)
        self.max_target_rate = float(max_target_rate)
        self.energy_weight = float(energy_weight)
        self.energy_power_ref_w = float(energy_power_ref_w)
        if (not math.isfinite(self.energy_weight) or self.energy_weight < 0
                or not math.isfinite(self.energy_power_ref_w)
                or self.energy_power_ref_w <= 0):
            raise ValueError("energy_weight must be >=0 and energy_power_ref_w >0")
        # Joint/gear assumptions are checked rather than indexing qvel[6:18]
        # on faith. Position servos in generate_model.py have gear=1.
        if not np.all(self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Motor work proxy only supports joint actuators")
        if not np.allclose(self.model.actuator_gear[:, 0], 1.0):
            raise ValueError("Motor work proxy requires gear 1 joint servos")
        self.actuated_dof_ids = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]].copy()
        if len(set(map(int, self.actuated_dof_ids))) != self.model.nu:
            raise ValueError("Expected one servo per independently driven joint")
        self.physics_dt = float(self.model.opt.timestep)
        self.episode_motor_energy_j = 0.0
        self.torso_id = self.model.body("torso").id
        self.ground_id = self.model.geom("ground").id
        self.foot_geom_ids = tuple(self.model.geom(f"leg_{tag}_foot").id for tag in "abc")
        assert (self.model.nu, self.model.nq, self.model.nv) == (12, 19, 18)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        # Local base velocities (6), gravity (3), height (1), joint pos (12),
        # joint velocity (12), applied servo target (12), foot contacts (3),
        # local displacement from reset (2) -> 51.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(51,), dtype=np.float32)
        self.neutral = np.tile(np.array([0.0, 0.0, -0.15, 0.30]), 3)
        self.actuator_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.actuator_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self.applied_targets = self.neutral.copy()
        self.episode_start_xy = np.zeros(2, dtype=np.float64)
        self.episode_step = 0

    def _action_to_targets(self, action: np.ndarray) -> np.ndarray:
        return self.neutral + np.where(
            action >= 0.0,
            action * (self.actuator_hi - self.neutral),
            action * (self.neutral - self.actuator_lo),
        )

    def _targets_as_normalized(self) -> np.ndarray:
        deviation = self.applied_targets - self.neutral
        span = np.where(deviation >= 0,
                        self.actuator_hi - self.neutral,
                        self.neutral - self.actuator_lo)
        return np.divide(deviation, np.maximum(span, 1e-8))

    def _orientation_mat(self) -> np.ndarray:
        rot = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        return rot.reshape(3, 3)

    def _foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(3, dtype=np.float32)
        mapping = {int(gid): k for k, gid in enumerate(self.foot_geom_ids)}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if int(c.geom1) == self.ground_id and int(c.geom2) in mapping:
                contacts[mapping[int(c.geom2)]] = 1.0
            elif int(c.geom2) == self.ground_id and int(c.geom1) in mapping:
                contacts[mapping[int(c.geom1)]] = 1.0
        return contacts

    def _nonfoot_ground_contact_count(self) -> int:
        count = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.ground_id in (int(c.geom1), int(c.geom2)):
                other = int(c.geom2) if int(c.geom1) == self.ground_id else int(c.geom1)
                if other not in self.foot_geom_ids:
                    count += 1
        return count

    def _observation(self) -> np.ndarray:
        rot = self._orientation_mat()
        local_linear_velocity = rot.T @ self.data.qvel[0:3]
        local_angular_velocity = self.data.qvel[3:6]
        local_gravity_direction = rot.T @ np.array([0.0, 0.0, -1.0])
        local_displacement = rot[:2, :2].T @ (self.data.qpos[:2] - self.episode_start_xy)
        obs = np.concatenate([
            np.clip(local_linear_velocity, -20.0, 20.0),
            np.clip(local_angular_velocity, -40.0, 40.0),
            local_gravity_direction,
            np.array([self.data.qpos[2]]),
            self.data.qpos[7:19],
            np.clip(self.data.qvel[6:18], -35.0, 35.0),
            self._targets_as_normalized(),
            self._foot_contacts(),
            np.clip(local_displacement, -20.0, 20.0),
        ])
        # Avoid NaN observations if physics becomes unstable on a terminal step.
        return np.nan_to_num(obs, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)

    def _is_alive(self) -> bool:
        z = float(self.data.qpos[2])
        upright = float(self._orientation_mat()[2, 2])
        return bool(0.25 <= z <= 0.85 and upright > 0.35
                    and np.all(np.isfinite(self.data.qpos))
                    and np.all(np.isfinite(self.data.qvel)))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.46]
        yaw = float(self.np_random.uniform(-math.pi, math.pi))
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.data.qpos[7:19] = self.neutral + self.np_random.normal(0.0, 0.015, 12)
        self.data.qvel[:] = 0.0
        self.applied_targets[:] = self.neutral
        self.data.ctrl[:] = self.applied_targets
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=25)
        self.episode_step = 0
        self.episode_motor_energy_j = 0.0
        self.episode_start_xy = self.data.qpos[0:2].copy()
        if not self._is_alive():
            raise RuntimeError("Robot is already fallen at reset. Check geometry and neutral pose.")
        return self._observation(), {}

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if a.shape != (12,):
            raise ValueError(f"Expected action shape (12,), got {a.shape}")
        prev_xy = self.data.qpos[0:2].copy()
        prev_targets = self.applied_targets.copy()
        desired_targets = self._action_to_targets(a)
        self.applied_targets[:] = np.clip(
            filter_servo_targets(prev_targets, desired_targets, self.dt,
                                 self.action_tau, self.max_target_rate),
            self.actuator_lo, self.actuator_hi,
        )
        self.data.ctrl[:] = self.applied_targets
        motor_energy_j = 0.0
        invalid_motor_state = False
        if self.energy_weight > 0:
            # Integrate actual abs(torque * angular_velocity) at EVERY physics
            # substep (10 x 0.002 s), not only the final 50 Hz sample.
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
                try:
                    motor_energy_j += absolute_mechanical_power_w(
                        self.data.actuator_force,
                        self.data.qvel[self.actuated_dof_ids],
                    ) * self.physics_dt
                except ValueError:
                    # If a solver state becomes non-finite, terminate instead
                    # of poisoning PPO/SEGB observations or reward with NaN.
                    invalid_motor_state = True
            self.episode_motor_energy_j += motor_energy_j
        else:
            # Preserve existing PPO and H2K6 physics execution exactly.
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        motor_power_w = motor_energy_j / self.dt
        penalty = energy_penalty(motor_energy_j, self.dt,
                                 weight=self.energy_weight,
                                 reference_power_w=self.energy_power_ref_w)
        self.episode_step += 1
        speed = signed_radial_progress_speed(
            self.episode_start_xy, prev_xy, self.data.qpos[0:2], self.dt,
        )
        alive = self._is_alive() and not invalid_motor_state
        if not np.isfinite(speed):
            # Broken physics must not poison PPO with NaN rewards.
            speed = 0.0
            alive = False
        reward = self.speed_weight * speed + self.alive_bonus * float(alive) - penalty
        terminated = not alive
        truncated = self.episode_step >= self.max_episode_steps
        net_distance = float(np.linalg.norm(self.data.qpos[0:2] - self.episode_start_xy))
        info = {
            "actuator_energy_j": float(motor_energy_j),
            "actuator_power_w": float(motor_power_w),
            "episode_actuator_energy_j": float(self.episode_motor_energy_j),
            "energy_penalty": float(penalty),
            "progress_speed_mps": speed,
            "speed_mps": speed,  # backwards-compatible key; now SIGNED progress
            "net_distance_m": net_distance,
            "alive": alive,
            "body_height_m": float(self.data.qpos[2]),
            "feet_on_ground": int(np.sum(self._foot_contacts())),
            "nonfoot_ground_contacts": self._nonfoot_ground_contact_count(),
            "servo_target_change_rad": float(np.linalg.norm(self.applied_targets - prev_targets)),
            "episode_duration_s": self.episode_step * self.dt,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        pass
