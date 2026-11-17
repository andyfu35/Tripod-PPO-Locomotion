"""Run SEGB tripod training using the same 5M env.step interaction budget as PPO.

Each of 5,000 candidates is given exactly 1,000 Gym control steps. Falling
robots are immediately reset; these continued steps count against the budget.
The last batch uses 8 worlds so there are NO uncounted training interactions.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from segb_core import (GENOME_LENGTH, Gene, ParetoGeneBank, batched_actions,
                       gene_from_archive, load_training_state, propose_gene,
                       record_improvement, save_training_state, write_policy)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tripod.xml"


def evaluation_batch(env, candidates: list[Gene], steps_per_gene: int, *,
                     base_seed: int) -> list[Gene]:
    """Shared fixed-size horizon, including episode restarts after a fall."""
    if steps_per_gene < 1 or not candidates:
        raise ValueError("Need a positive candidate horizon")
    env.seed(base_seed)
    obs = env.reset()
    n = len(candidates)
    params = np.stack([g.parameters for g in candidates]).astype(np.float32)
    if obs.shape != (n, 51) or params.shape != (n, GENOME_LENGTH):
        raise ValueError("Vector environment/policy dimension mismatch")
    progress = np.zeros(n, dtype=np.float64)
    reward = np.zeros(n, dtype=np.float64)
    life_lengths = np.zeros(n, dtype=np.int32)
    sum_squared_life = np.zeros(n, dtype=np.int64)
    falls = np.zeros(n, dtype=np.int32)
    for _ in range(steps_per_gene):
        actions = batched_actions(obs, params)
        obs, rews, dones, infos = env.step(actions)
        reward += rews
        life_lengths += 1
        for k, info in enumerate(infos):
            progress[k] += float(info["progress_speed_mps"])
            if bool(dones[k]):
                sum_squared_life[k] += int(life_lengths[k]) ** 2
                life_lengths[k] = 0
            if not bool(info["alive"]):
                falls[k] += 1
    sum_squared_life += life_lengths.astype(np.int64) ** 2
    # 0..1: a 1000-step uninterrupted life -> 1.0; 10 equal 100-step lives -> .1.
    longevity = sum_squared_life.astype(np.float64) / (steps_per_gene ** 2)
    for i, gene in enumerate(candidates):
        gene.progress_mps = float(progress[i] / steps_per_gene)
        gene.longevity = float(longevity[i])
        gene.mean_reward = float(reward[i] / steps_per_gene)
        gene.fall_count = int(falls[i])
    return candidates


def make_env(n_worlds: int, *, seed: int):
    """Defer MuJoCo and SB3 dependencies until actual training."""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from tripod_env import TripodEnv
    return make_vec_env(
        TripodEnv, n_envs=n_worlds, seed=seed,
        env_kwargs={"max_episode_steps": 1000, "alive_bonus": 0.2,
                    "action_tau": 0.10, "max_target_rate": 2.5},
        vec_env_cls=SubprocVecEnv if n_worlds > 1 else DummyVecEnv,
        vec_env_kwargs={"start_method": "spawn"} if n_worlds > 1 else None,
    )


def budget_plan(total_steps: int, steps_per_gene: int, worlds: int) -> list[int]:
    """Exact-budget planning. No ghost worlds or dropped residual steps."""
    if min(total_steps, steps_per_gene, worlds) < 1:
        raise ValueError("All budget parameters must be positive")
    if total_steps % steps_per_gene:
        raise ValueError("Total steps must be divisible by steps per candidate")
    count = total_steps // steps_per_gene
    return [min(worlds, count - start) for start in range(0, count, worlds)]


def output_summary(out: Path, *, bank: ParetoGeneBank, best: Gene | None,
                   steps: int, config: dict[str, Any], history: list[dict[str, Any]]):
    if not bank.genes:
        raise RuntimeError("Empty Gene Bank")
    champions = {
        "best_reward": best,
        "best_progress": gene_from_archive(bank, axis="progress"),
        "best_survival": gene_from_archive(bank, axis="survival"),
        "balanced": gene_from_archive(bank, axis="balanced"),
    }
    for name, gene in champions.items():
        if gene is None:
            raise RuntimeError("No best gene")
        write_policy(out / f"{name}.npz", gene)
    with (out / "candidate_history.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["id", "source", "parent", "generation", "progress_mps",
                   "longevity", "mean_reward", "fall_count", "admitted", "training_steps_total"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    result = {
        "algorithm": "SEGB tripod adaptation: Pareto Bank + PAES-style depth-5 crowding + Q2 parent selection",
        "interaction_steps": steps,
        "config": config,
        "bank_count": len(bank.genes),
        "archive_admissions": bank.admissions,
        "paes_density_evictions": bank.density_evictions,
        "dominated_rejections": bank.dominated_rejections,
        "generations_max": max(g.generation for g in bank.genes.values()),
        "champions": {name: {"checkpoint": f"{name}.npz", **gene.as_dict()}
                      for name, gene in champions.items()},
        "candidates": len(history),
        "de_novo_candidates": sum(row["source"] == "de_novo" for row in history),
        "inherited_candidates": sum(row["source"] == "inherited" for row in history),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("interaction_steps", "candidates", "bank_count",
                                            "archive_admissions", "generations_max")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--steps-per-gene", type=int, default=1000)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--archive-capacity", type=int, default=1024)
    parser.add_argument("--archive-depth", type=int, default=5)
    parser.add_argument("--de-novo", type=float, default=0.20)
    parser.add_argument("--save-every", type=int, default=250_000)
    parser.add_argument("--out-dir", type=Path, default=Path("runs_segb_5m_seed0"))
    parser.add_argument("--resume", action="store_true", help="Continue from --out-dir/checkpoint.npz")
    args = parser.parse_args()
    if not (0 <= args.de_novo <= 1) or args.archive_capacity < 3 or args.save_every < 1:
        parser.error("Invalid archive/de-novo/checkpoint settings")
    plan = budget_plan(args.total_steps, args.steps_per_gene, args.worlds)
    candidate_count = args.total_steps // args.steps_per_gene
    config = {
        "total_steps": args.total_steps,
        "steps_per_gene": args.steps_per_gene,
        "worlds": args.worlds,
        "seed": args.seed,
        "archive_capacity": args.archive_capacity,
        "archive_depth": args.archive_depth,
        "de_novo": args.de_novo,
        "robot_xml_sha256": sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "control_dt_s": 0.02,
        "reward": "signed_radial_progress_mps + 0.2*alive",
        "genome_architecture": f"MLP_51_64_64_12_{GENOME_LENGTH}_parameters",
        "objective_axes": ["mean_signed_radial_progress_mps", "longevity_episode_segment_squares"],
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.npz"
    if args.resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; missing {checkpoint}")
        bank, best, next_index, rng, history = load_training_state(checkpoint, config)
    else:
        if checkpoint.is_file() or (out / "summary.json").is_file():
            raise FileExistsError(f"Existing run in {out}; use --resume or a different --out-dir")
        bank = ParetoGeneBank(capacity=args.archive_capacity, depth=args.archive_depth)
        best = None
        next_index = 0
        rng = np.random.default_rng(args.seed)
        history: list[dict[str, Any]] = []
    print(f"SEGB {args.worlds} worlds, {candidate_count} candidates x {args.steps_per_gene} steps = "
          f"{args.total_steps:,} interactions, {len(plan)} batches")
    print(f"Continue from candidate {next_index}; archive={len(bank.genes)}")
    if next_index == candidate_count:
        output_summary(out, bank=bank, best=best, steps=args.total_steps,
                       config=config, history=history)
        return
    if next_index % args.worlds:
        raise ValueError("Resume checkpoint must be at a full batch boundary")
    step_counter = next_index * args.steps_per_gene
    start_steps = step_counter
    last_save_steps = step_counter
    env = None
    active_world_count = 0
    started = time.monotonic()
    try:
        for batch_index in range(next_index // args.worlds, len(plan)):
            count = plan[batch_index]
            if env is None or active_world_count != count:
                if env is not None:
                    env.close()
                env = make_env(count, seed=args.seed)
                active_world_count = count
            before = bank.genes.copy()
            candidates = [propose_gene(next_index + i, bank, rng, args.de_novo)
                          for i in range(count)]
            # All proposals are fixed before evaluation; archive update is atomic per batch.
            candidates = evaluation_batch(env, candidates, args.steps_per_gene,
                                          base_seed=args.seed + 100_000 + next_index)
            for candidate in candidates:
                record_improvement(candidate, before.get(candidate.parent_id))
                admitted = bank.accept(candidate, rng)
                if best is None or candidate.mean_reward > best.mean_reward:
                    best = candidate
                history.append({
                    "id": candidate.gene_id,
                    "source": "de_novo" if candidate.parent_id is None else "inherited",
                    "parent": "" if candidate.parent_id is None else candidate.parent_id,
                    "generation": candidate.generation,
                    "progress_mps": candidate.progress_mps,
                    "longevity": candidate.longevity,
                    "mean_reward": candidate.mean_reward,
                    "fall_count": candidate.fall_count,
                    "admitted": int(admitted),
                    "training_steps_total": (candidate.gene_id + 1) * args.steps_per_gene,
                })
            next_index += count
            step_counter += count * args.steps_per_gene
            if (batch_index + 1) % 10 == 0 or next_index == candidate_count:
                sec = max(1e-6, time.monotonic() - started)
                print(f"BATCH {batch_index+1}/{len(plan)} | Steps {step_counter:,}/"
                      f"{args.total_steps:,} | Bank {len(bank.genes)} | "
                      f"Best reward/step {best.mean_reward:.4f} | "
                      f"{(step_counter-start_steps)/sec:.0f} steps/s", flush=True)
            if step_counter - last_save_steps >= args.save_every or next_index == candidate_count:
                save_training_state(checkpoint, bank=bank, all_best=best,
                                    next_gene_index=next_index, rng=rng,
                                    configuration=config, history=history)
                last_save_steps = step_counter
    finally:
        if env is not None:
            env.close()
    if step_counter != args.total_steps:
        raise RuntimeError(f"Budget accounting error {step_counter} != {args.total_steps}")
    output_summary(out, bank=bank, best=best, steps=step_counter,
                   config=config, history=history)
    print(f"DONE. Archive, champions and history saved in {out}")


if __name__ == "__main__":
    main()
