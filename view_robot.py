"""Live MuJoCo viewer for the tripod, with optional v2 PPO policy.

macOS interactive viewer: mjpython view_robot.py [--model ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import mujoco.viewer

from tripod_env import TripodEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Optional trained v2 PPO .zip")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--camera", choices=["fixed", "tracking"], default="fixed")
    args = p.parse_args()
    policy = None
    env_kwargs = {}
    if args.model:
        from stable_baselines3 import PPO
        config_file = Path(args.model).parent / "train_config.json"
        if not config_file.is_file():
            raise FileNotFoundError("v2 training config not found beside model; v1 checkpoints are incompatible")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if not str(config.get("version", "")).startswith("tripod-v2-"):
            raise ValueError("Expected a v2 checkpoint")
        env_kwargs = config["env_kwargs"]
        policy = PPO.load(args.model, device="cpu")
    env = TripodEnv(**env_kwargs)
    if policy is not None and policy.observation_space.shape != env.observation_space.shape:
        raise ValueError("Checkpoint has incompatible observation space")
    obs, _ = env.reset(seed=args.seed)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
        viewer.cam.distance = 4.0 if args.camera == "fixed" else 1.8
        viewer.cam.azimuth = 125
        viewer.cam.elevation = -25
        while viewer.is_running():
            start = time.monotonic()
            if policy is None:
                action = np.zeros(12, dtype=np.float32)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            if args.camera == "tracking":
                viewer.cam.lookat[:] = env.data.qpos[:3]
            viewer.sync()
            time.sleep(max(0.0, env.dt - (time.monotonic() - start)))
    env.close()


if __name__ == "__main__":
    main()
