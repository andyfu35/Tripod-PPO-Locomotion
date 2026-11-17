"""Held-out, identical-seed comparison for H2-K6 5M -> 10M -> ... PPO.

Inputs are the summary.json files exported by evaluate.py.  Comparison is
post-training and does not consume the evolution budget.  The same 10 reset
seeds must be used by the runner that produces the files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_eval(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", [])
    if not episodes:
        raise ValueError(f"No held-out episodes found: {path}")
    ids = [int(e["episode"]) for e in episodes]
    if ids != list(range(len(ids))):
        raise ValueError(f"Episodes must be ordered and contiguous from zero: {path}")
    return data


def compare(before: dict, after: dict, ppo: dict | None = None) -> list[dict]:
    sources = [("SEGB_before", before), ("SEGB_after", after)]
    if ppo is not None:
        sources.append(("PPO", ppo))
    sizes = {len(x["episodes"]) for _, x in sources}
    if len(sizes) != 1:
        raise ValueError("All methods must use the same held-out episode count")
    results = []
    for name, data in sources:
        episodes = data["episodes"]
        speeds = np.asarray([float(x["net_speed_mps"]) for x in episodes])
        duration = np.asarray([float(x["seconds"]) for x in episodes])
        survival = np.asarray([not bool(x["fell"]) for x in episodes])
        results.append({"method": name, "episodes": len(episodes),
                        "mean_speed_mps": float(speeds.mean()),
                        "median_speed_mps": float(np.median(speeds)),
                        "mean_survival_seconds": float(duration.mean()),
                        "survival_rate": float(survival.mean())})
    return results


def render_chart(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["method"].replace("_", " ") for row in rows]
    speeds = [row["mean_speed_mps"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, speeds)
    for bar, row in zip(bars, rows):
        ax.annotate(f'{row["mean_speed_mps"]:.3f} m/s\n'
                    f'{row["mean_survival_seconds"]:.1f} s survival',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                    textcoords="offset points")
    ax.set_ylim(0, max(speeds) * 1.3 if max(speeds) > 0 else 0.25)
    ax.set_ylabel("Held-out mean net speed (m/s)")
    ax.set_title("Same 10 reset seeds: SEGB continuation vs PPO")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--ppo", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    before = read_eval(args.before)
    after = read_eval(args.after)
    ppo = read_eval(args.ppo) if args.ppo else None
    rows = compare(before, after, ppo)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render_chart(rows, args.png)
    for row in rows:
        print(f'{row["method"]:<12} speed={row["mean_speed_mps"]:.4f} m/s | '
              f'survival={row["mean_survival_seconds"]:.2f} s | '
              f'full-life-rate={row["survival_rate"]:.1%}')
    difference = rows[1]["mean_speed_mps"] - rows[0]["mean_speed_mps"]
    print(f'Continuation minus baseline: {difference:+.4f} m/s (held-out 10 episodes)')
    print(f'Saved {args.csv} and {args.png}')


if __name__ == "__main__":
    main()
