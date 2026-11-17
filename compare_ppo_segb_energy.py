"""Compare held-out 5M PPO/SEGB under the SAME energy-weighted MuJoCo task.

Primary speed is fixed 20-second horizon displacement / 20s, never the
inflated distance / lifespan estimate for quick-falling policies.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

FIELDS = ["method", "episodes", "mean_fixed20_speed_mps", "mean_active_speed_mps",
          "mean_survival_seconds", "survival_rate", "mean_motor_work_j",
          "mean_motor_power_w_while_alive", "mean_work_j_per_meter", "mean_episode_reward"]


def summarize_evaluation(path: Path, method: str) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    episodes = raw.get("episodes", [])
    if not episodes:
        raise ValueError(f"No evaluation episodes in {path}")
    n = len(episodes)
    for i, e in enumerate(episodes):
        if int(e["episode"]) != i:
            raise ValueError(f"Evaluation episodes are not ordered in {path}")
        for key in ("net_displacement_m", "seconds", "motor_work_j", "fell"):
            if key not in e:
                raise ValueError(f"Old/non-energy evaluation missing {key}: {path}")
    norm = np.asarray([e["net_displacement_m"] / 20.0 for e in episodes], float)
    alive_speed = np.asarray([e["net_displacement_m"] / e["seconds"] for e in episodes], float)
    durations = np.asarray([e["seconds"] for e in episodes], float)
    work = np.asarray([e["motor_work_j"] for e in episodes], float)
    if not np.isfinite([*norm, *alive_speed, *durations, *work]).all() or np.any(work < 0):
        raise ValueError("Nonfinite/negative evaluation metrics")
    meters = np.asarray([e["net_displacement_m"] for e in episodes], float)
    return {
        "method": method, "episodes": n,
        "mean_fixed20_speed_mps": float(norm.mean()),
        "mean_active_speed_mps": float(alive_speed.mean()),
        "mean_survival_seconds": float(durations.mean()),
        "survival_rate": float(np.mean([not e["fell"] for e in episodes])),
        "mean_motor_work_j": float(work.mean()),
        "mean_motor_power_w_while_alive": float(np.mean(work / durations)),
        "mean_work_j_per_meter": float(work.sum() / max(meters.sum(), 1e-8)),
        "mean_episode_reward": float(np.mean([e["episode_reward"] for e in episodes])),
    }


def compare(ppo: dict, segb: dict) -> list[dict]:
    if ppo["episodes"] != segb["episodes"]:
        raise ValueError("PPO/SEGB evaluation episode counts differ")
    return [ppo, segb]


def make_bar(rows: list[dict], field: str, ylabel: str, file: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    vals = [r[field] for r in rows]
    bars = ax.bar([r["method"] for r in rows], vals)
    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:.3f}", (bar.get_x()+bar.get_width()/2, val),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(vals), 1e-6)*1.24)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(file, dpi=155)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ppo", type=Path, required=True)
    p.add_argument("--segb", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results_energy_matched_5m_report"))
    a=p.parse_args()
    ppo_raw=json.loads(a.ppo.read_text())
    segb_raw=json.loads(a.segb.read_text())
    if (float(ppo_raw.get("energy_weight",0.0)) != float(segb_raw.get("energy_weight",-1.0))
            or float(ppo_raw.get("energy_power_ref_w",0.0)) != float(segb_raw.get("energy_power_ref_w",-1.0))):
        raise ValueError("PPO/SEGB evaluation energy parameters are not identical")
    rows=compare(summarize_evaluation(a.ppo,"PPO energy 5M"),
                 summarize_evaluation(a.segb,"SEGB H2K6 energy 5M"))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (a.out_dir/"comparison.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    make_bar(rows,"mean_fixed20_speed_mps","Fixed 20s horizon displacement / 20s (m/s)",a.out_dir/"fixed20_speed.png")
    make_bar(rows,"survival_rate","Full 20s survival rate",a.out_dir/"survival_rate.png")
    make_bar(rows,"mean_motor_power_w_while_alive","Mean motor mechanical power while alive (W)",a.out_dir/"motor_power.png")
    print("Energy-matched results (same 10 held-out reset seeds):")
    for r in rows:
        print(f"{r['method']}: fixed20_speed={r['mean_fixed20_speed_mps']:.4f} m/s, "
              f"full-life={r['survival_rate']:.1%}, "
              f"mean_work={r['mean_motor_work_j']:.1f} J, "
              f"mean_power={r['mean_motor_power_w_while_alive']:.2f} W")
    print("Saved",a.out_dir)


if __name__=="__main__":
    main()
