"""Evaluate PPO, SEGB, neutral/random control with identical gait metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tripod_env import TripodEnv


def plot_gait(rows: list[dict], save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray([r["time_s"] for r in rows], dtype=float)
    x = np.asarray([r["x_m"] for r in rows], dtype=float)
    y = np.asarray([r["y_m"] for r in rows], dtype=float)
    foot = np.asarray([[r[f"foot_{letter}"] for letter in "abc"] for r in rows], dtype=float).T
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 7.5), layout="constrained")
    ax_top.scatter(x, y, c=t, cmap="viridis", s=7)
    ax_top.plot(x[0], y[0], "o", markersize=9, label="Start")
    ax_top.plot(x[-1], y[-1], "x", markersize=9, label="End")
    ax_top.set_title("Body planar trajectory (color = time)")
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.axis("equal")
    ax_top.grid(alpha=0.2)
    ax_top.legend()
    if len(t) > 1:
        ax_bottom.imshow(foot, aspect="auto", interpolation="nearest", origin="upper",
                         cmap="Greys", extent=[t[0], t[-1], 2.5, -0.5], vmin=0, vmax=1)
    ax_bottom.set_yticks([0, 1, 2], ["Leg A", "Leg B", "Leg C"])
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_title("Foot-ground contacts: dark = stance, light = swing")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def open_video(env: TripodEnv, filename: Path, fps: int, width: int, height: int, camera_mode: str):
    import imageio.v2 as imageio
    import mujoco

    buffer_width = int(env.model.vis.global_.offwidth)
    buffer_height = int(env.model.vis.global_.offheight)
    if width > buffer_width or height > buffer_height:
        raise ValueError(f"Video {width}x{height} exceeds model framebuffer "
                         f"{buffer_width}x{buffer_height}; edit models/tripod.xml")
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    if camera_mode == "tracking":
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = env.torso_id
        camera.distance = 1.7
    else:
        # Fixed world-space view: show whether the body actually travels.
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.0, 0.3]
        camera.distance = 4.0
    camera.azimuth = 130
    camera.elevation = -28
    writer = imageio.get_writer(str(filename), fps=fps, codec="libx264", quality=7)
    return renderer, camera, writer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="v2 SB3 PPO .zip; omit for neutral baseline")
    p.add_argument("--segb-model", help="SEGB evolution champion .npz (same environment/reward)")
    p.add_argument("--random", action="store_true", help="Random baseline instead of neutral")
    p.add_argument("--stochastic", action="store_true", help="Sample PPO actions (default: deterministic mean)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out-dir", default="results_v2")
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-episode", type=int, default=0)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--camera", choices=["fixed", "tracking"], default="fixed")
    p.add_argument("--video-width", type=int, default=720)
    p.add_argument("--video-height", type=int, default=540)
    args = p.parse_args()
    if sum(bool(x) for x in (args.model, args.segb_model, args.random)) > 1:
        p.error("Choose at most one of --model, --segb-model, --random")
    if args.episodes < 1:
        p.error("--episodes must be positive")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {}
    policy = None
    if args.model:
        from stable_baselines3 import PPO
        model_path = Path(args.model)
        config_file = model_path.parent / "train_config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Missing {config_file}. This evaluator is for v2 policies with 51 "
                "observations. To inspect the old 49-observation policy, use v1 evaluate.py."
            )
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if not str(config.get("version", "")).startswith("tripod-v2-"):
            raise ValueError(f"Not a v2 training config: {config_file}")
        env_kwargs = config["env_kwargs"]
        policy = PPO.load(args.model, device="cpu")
    elif args.segb_model:
        from segb_core import SEGBPredictor
        policy = SEGBPredictor.load(args.segb_model)

    env = TripodEnv(**env_kwargs)
    if policy is not None and policy.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            f"Policy obs {policy.observation_space.shape} != env obs "
            f"{env.observation_space.shape}; v1 checkpoints are not compatible."
        )
    episode_summaries = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            rng = np.random.default_rng(args.seed + ep)
            rows = []
            total_reward = 0.0
            distance_walked = 0.0
            last_xy = env.data.qpos[:2].copy()
            writer = renderer = camera = None
            if args.video and ep == args.video_episode:
                video_path = out / f"tripod_episode_{ep:02d}.mp4"
                renderer, camera, writer = open_video(
                    env, video_path, args.fps, args.video_width, args.video_height, args.camera,
                )
            try:
                for step in range(env.max_episode_steps):
                    if policy is not None:
                        action, _ = policy.predict(obs, deterministic=not args.stochastic)
                    elif args.random:
                        action = rng.uniform(-1, 1, 12).astype(np.float32)
                    else:
                        action = np.zeros(12, dtype=np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    xy = env.data.qpos[:2].copy()
                    distance_walked += float(np.linalg.norm(xy - last_xy))
                    last_xy = xy
                    total_reward += reward
                    contacts = env._foot_contacts()
                    row = {
                        "time_s": round((step + 1) * env.dt, 4),
                        "x_m": float(xy[0]), "y_m": float(xy[1]),
                        "z_m": float(env.data.qpos[2]),
                        "progress_speed_mps": info["progress_speed_mps"],
                        "servo_target_change_rad": info["servo_target_change_rad"],
                        "mean_abs_joint_speed_radps": float(np.mean(np.abs(env.data.qvel[6:18]))),
                        "nonfoot_ground_contacts": info["nonfoot_ground_contacts"],
                        "foot_a": int(contacts[0]), "foot_b": int(contacts[1]),
                        "foot_c": int(contacts[2]),
                    }
                    row.update({f"joint_{k:02d}_rad": float(v)
                                for k, v in enumerate(env.data.qpos[7:19])})
                    rows.append(row)
                    if writer is not None and (step % max(1, round(1.0 / (args.fps * env.dt))) == 0):
                        renderer.update_scene(env.data, camera=camera)
                        writer.append_data(renderer.render())
                    if terminated or truncated:
                        break
            finally:
                if writer is not None:
                    writer.close()
                if renderer is not None:
                    renderer.close()

            csv_path = out / f"episode_{ep:02d}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            if ep == 0:
                plot_gait(rows, out / "gait_and_trajectory.png")

            foot_counts = np.array([[r[f"foot_{tag}"] for tag in "abc"]
                                    for r in rows]).sum(axis=1)
            duration = len(rows) * env.dt
            net = info["net_distance_m"]
            stat = {
                "episode": ep,
                "seconds": duration,
                "net_displacement_m": net,
                "net_speed_mps": net / duration,
                "path_length_m": distance_walked,
                "path_efficiency_0to1": net / max(distance_walked, 1e-8),
                "mean_joint_speed_radps": float(np.mean([r["mean_abs_joint_speed_radps"] for r in rows])),
                "mean_servo_target_change_rad_per_step": float(np.mean(
                    [r["servo_target_change_rad"] for r in rows])),
                "any_nonfoot_ground_contact_fraction": float(np.mean(
                    [r["nonfoot_ground_contacts"] > 0 for r in rows])),
                "episode_reward": total_reward,
                "fell": bool(terminated),
                "foot_support_fraction_0_1_2_3": [float(np.mean(foot_counts == n)) for n in range(4)],
            }
            episode_summaries.append(stat)
            print(json.dumps(stat, ensure_ascii=False, indent=2))
        result = {
            "policy": args.model or args.segb_model or ("random baseline" if args.random else "neutral baseline"),
            "episodes": episode_summaries,
            "mean_net_speed_mps": float(np.mean([s["net_speed_mps"] for s in episode_summaries])),
            "mean_survival_seconds": float(np.mean([s["seconds"] for s in episode_summaries])),
            "survival_rate": float(np.mean([not s["fell"] for s in episode_summaries])),
        }
        (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved evaluation outputs to {out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
