"""PPO training for the three-leg robot, experiment C (16x1024 rollout)."""
from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tripod_env import TripodEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length per world; experiment C uses 1024")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-steps", type=int, default=1000)
    p.add_argument("--alive-bonus", type=float, default=0.2)
    p.add_argument("--energy-weight", type=float, default=0.0,
                   help="Penalty for motor power relative to energy-power-ref-w; 0 preserves baseline")
    p.add_argument("--energy-power-ref-w", type=float, default=50.0)
    p.add_argument("--action-tau", type=float, default=0.10)
    p.add_argument("--max-target-rate", type=float, default=2.5)
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", default=None, help="Only resume policies trained with v2 (51 obs)")
    p.add_argument("--out-dir", default="runs_c_16x1024_seed0")
    args = p.parse_args()
    if args.energy_weight < 0 or args.energy_power_ref_w <= 0:
        p.error("energy-weight must be nonnegative and energy-power-ref-w positive")
    if args.n_envs < 1 or args.timesteps < 1 or args.n_steps < 1:
        p.error("--n-envs, --n-steps and --timesteps must be positive")
    rollout = args.n_envs * args.n_steps
    if rollout < 256 or rollout % 256 != 0:
        p.error("n-envs * n-steps must be at least 256 and divisible by batch_size=256")
    updates = ceil(args.timesteps / rollout)
    print(f"Experiment C: {args.n_envs} worlds x {args.n_steps} steps = "
          f"{rollout:,} samples/update, {updates} PPO updates, "
          f"{updates * rollout:,} actual interaction steps")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = out_dir / "monitor"
    monitor_dir.mkdir(exist_ok=True)
    env_kwargs = {
        "max_episode_steps": args.episode_steps,
        "alive_bonus": args.alive_bonus,
        "energy_weight": args.energy_weight,
        "energy_power_ref_w": args.energy_power_ref_w,
        "action_tau": args.action_tau,
        "max_target_rate": args.max_target_rate,
    }
    if args.resume:
        previous_config = Path(args.resume).parent / "train_config.json"
        if not previous_config.exists():
            raise FileNotFoundError("Cannot resume: v2 train_config.json was not found next to checkpoint")
        previous = json.loads(previous_config.read_text(encoding="utf-8"))
        old_kwargs = previous["env_kwargs"]
        if previous.get("n_envs") != args.n_envs or previous.get("ppo", {}).get("n_steps") != args.n_steps:
            raise ValueError("Cannot resume with a different n_envs or n_steps; start a new run")
        if old_kwargs != env_kwargs:
            raise ValueError(f"Environment parameters differ from checkpoint: {old_kwargs} != {env_kwargs}")
    (out_dir / "train_config.json").write_text(json.dumps({
        "version": "tripod-v2-signed-progress-smoothed-servo",
        "experiment": "C-many-worlds-long-rollout",
        "reward": "signed_radial_progress + alive_bonus - energy_weight * abs_joint_motor_work / (dt * energy_power_ref_w)",
        "requested_total_timesteps": args.timesteps,
        "rollout_samples_per_update": rollout,
        "expected_ppo_updates": updates,
        "expected_total_timesteps": updates * rollout,
        "env_kwargs": env_kwargs,
        "ppo": {"n_steps": args.n_steps, "batch_size": 256, "n_epochs": 10,
                "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.001,
                "log_std_init": -1.5},
        "seed": args.seed, "n_envs": args.n_envs,
    }, indent=2), encoding="utf-8")

    vector_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    vec_kwargs = {"start_method": "spawn"} if args.n_envs > 1 else None
    env = make_vec_env(
        TripodEnv, n_envs=args.n_envs, seed=args.seed,
        env_kwargs=env_kwargs,
        monitor_dir=str(monitor_dir), vec_env_cls=vector_cls,
        vec_env_kwargs=vec_kwargs,
    )
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=args.device)
        else:
            model = PPO(
                "MlpPolicy", env,
                policy_kwargs={"net_arch": [256, 256], "log_std_init": -1.5},
                learning_rate=3e-4, n_steps=args.n_steps, batch_size=256,
                n_epochs=10, gamma=0.995, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.001,
                seed=args.seed, device=args.device, verbose=1,
            )
        checkpoint = CheckpointCallback(
            save_freq=max(1, args.save_every // args.n_envs),
            save_path=str(out_dir), name_prefix="tripod_ppo",
        )
        model.learn(
            total_timesteps=args.timesteps, callback=checkpoint,
            reset_num_timesteps=not bool(args.resume), progress_bar=False,
        )
        model.save(str(out_dir / "tripod_ppo_final"))
        print(f"Saved policy: {out_dir / 'tripod_ppo_final.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
