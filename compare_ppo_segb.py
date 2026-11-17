"""Compare held-out PPO and SEGB evaluations (both produced by evaluate.py)."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

FIELDS = ("mean_net_speed_mps", "mean_survival_seconds", "survival_rate")


def comparison(ppo: dict, segb: dict) -> list[dict]:
    for name, payload in (("PPO", ppo), ("SEGB", segb)):
        if not isinstance(payload.get("episodes"), list) or not payload["episodes"]:
            raise ValueError(f"{name} evaluation summary has no episodes")
    if len(ppo["episodes"]) != len(segb["episodes"]):
        raise ValueError("Evaluation episode count must match")
    return [{"metric": k, "PPO": float(ppo[k]), "SEGB": float(segb[k]),
             "SEGB_minus_PPO": float(segb[k]) - float(ppo[k])} for k in FIELDS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ppo", type=Path, required=True)
    p.add_argument("--segb", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("comparison_ppo_segb.csv"))
    args = p.parse_args()
    a = json.loads(args.ppo.read_text(encoding="utf-8"))
    b = json.loads(args.segb.read_text(encoding="utf-8"))
    result = comparison(a, b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    print(f"Comparison of {len(a['episodes'])} held-out episodes per method:")
    for row in result:
        print(f" {row['metric']}: PPO={row['PPO']:.5f}, SEGB={row['SEGB']:.5f}, "
              f"SEGB-PPO={row['SEGB_minus_PPO']:+.5f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
