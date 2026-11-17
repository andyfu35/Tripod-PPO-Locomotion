"""Steady-state SEGB V2, **exactly 5M control interactions** on 16 MuJoCo worlds.

A candidate is scored for one uninterrupted life, never an aggregate of
restarts. When it falls, that world auto-resets and begins a NEW candidate on
the following control step. No PPO, imitation or prespecified leg gait.

A synchronous 1,000-tick round boundary allows exact checkpoint/restart without
saving all 16 MuJoCo engine/RNG states. Partial lives are censored, never
accidentally promoted as 20-second survivors.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from segb_core import (
    Gene, ParetoGeneBank, batched_actions, gene_from_archive,
    record_improvement, write_policy,
)
from segb_v2_core import VERSION, new_gene, score_single_life
from train_segb import budget_plan, make_env

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tripod.xml"


def _score_and_archive(gene: Gene, tracking: dict, *, bank: ParetoGeneBank,
                       rng: np.random.Generator, horizon: int,
                       history: list[dict], finished_steps: int,
                       best: Gene | None, censored: bool) -> Gene | None:
    """Call only at done (single life) or at a checkpoint boundary (censored)."""
    life_steps = tracking["life_steps"]
    if life_steps <= 0:
        raise ValueError("Cannot score a zero-step gene")
    fell = bool(tracking["fell"])
    score_single_life(gene, total_progress=tracking["progress"],
                      total_reward=tracking["reward"],
                      steps=life_steps, horizon=horizon, fell=fell)
    admitted = False
    if not censored:
        parent = bank.genes.get(gene.parent_id)
        record_improvement(gene, parent)
        admitted = bank.accept(gene, rng)
        if best is None or gene.mean_reward > best.mean_reward:
            best = gene
    history.append({
        "id": gene.gene_id,
        "source": "de_novo" if gene.parent_id is None else "inherited",
        "parent": "" if gene.parent_id is None else gene.parent_id,
        "generation": gene.generation,
        "life_steps": life_steps,
        "progress_mps": gene.progress_mps,
        "longevity": gene.longevity,
        "mean_reward": gene.mean_reward,
        "fall_count": gene.fall_count,
        "censored": int(censored),
        "admitted": int(admitted),
        "training_steps_total": finished_steps,
    })
    return best


def run_round(env, *, bank: ParetoGeneBank, rng: np.random.Generator,
              gene_index: int, max_steps: int, horizon: int, seed: int,
              de_novo_fraction: float, history: list[dict],
              base_steps: int, best: Gene | None) -> tuple[int, Gene | None]:
    """A round is `worlds * max_steps` interactions, independent of fall times.

    SB3 VecEnv auto-resets a terminated world and returns its new observation
    and the terminal info. We assign a new gene to that slot for the next step.
    """
    n = env.num_envs if hasattr(env, "num_envs") else env.worlds
    env.seed(seed)
    obs = env.reset()
    if obs.shape != (n, 51):
        raise ValueError(f"Incorrect observation shape: {obs.shape}")
    active = [new_gene(gene_index + k, bank, rng, de_novo_fraction)
              for k in range(n)]
    gene_index += n
    track = [dict(life_steps=0, progress=0.0, reward=0.0, fell=False)
             for _ in range(n)]
    for t in range(max_steps):
        action = batched_actions(obs, np.stack([g.parameters for g in active]))
        obs, rewards, dones, infos = env.step(action)
        for w in range(n):
            tr = track[w]
            tr["life_steps"] += 1
            tr["progress"] += float(infos[w]["progress_speed_mps"])
            tr["reward"] += float(rewards[w])
            tr["fell"] = bool(not infos[w]["alive"])
            elapsed = base_steps + (t + 1) * n
            complete = bool(dones[w]) or tr["life_steps"] >= horizon
            if complete:
                best = _score_and_archive(
                    active[w], tr, bank=bank, rng=rng, horizon=horizon,
                    history=history, finished_steps=elapsed, best=best,
                    censored=False)
                if t + 1 < max_steps:
                    # A done VecEnv world has already reset; its new observation
                    # belongs to this new candidate, not to the previous gene.
                    active[w] = new_gene(gene_index, bank, rng, de_novo_fraction)
                    gene_index += 1
                    track[w] = dict(life_steps=0, progress=0.0, reward=0.0, fell=False)
                else:
                    track[w] = None
    # The remaining partial lives have reached a checkpoint barrier but NOT
    # an episode terminal. They count toward the physical budget, yet cannot
    # falsely enter the bank as full-horizon survivors.
    for w, tr in enumerate(track):
        if tr is not None and tr["life_steps"]:
            best = _score_and_archive(
                active[w], tr, bank=bank, rng=rng, horizon=horizon,
                history=history, finished_steps=base_steps + max_steps*n,
                best=best, censored=True)
    return gene_index, best


def save_checkpoint(path: Path, *, bank: ParetoGeneBank, best: Gene | None,
                    rng: np.random.Generator, history: list[dict],
                    config: dict[str, Any], steps: int, round_index: int,
                    gene_index: int) -> None:
    genes = bank.ordered()
    records = genes + ([best] if best is not None else [])
    params = np.stack([g.parameters for g in records]).astype(np.float32)
    meta = {
        "format_version": 2, "config": config, "steps": steps,
        "round_index": round_index, "gene_index": gene_index,
        "bank_size": len(genes), "genes": [g.as_dict() for g in records],
        "rng_state": rng.bit_generator.state, "history": history,
        "admissions": bank.admissions,
        "density_evictions": bank.density_evictions,
        "dominated_rejections": bank.dominated_rejections,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as out:
        np.savez_compressed(out, parameters=params, meta=json.dumps(meta))
    tmp.replace(path)


def load_checkpoint(path: Path, config: dict[str, Any]):
    with np.load(path, allow_pickle=False) as data:
        mat = np.asarray(data["parameters"], dtype=np.float32)
        meta = json.loads(str(data["meta"].item()))
    if meta.get("format_version") != 2 or meta.get("config") != config:
        raise ValueError("Checkpoint mismatch: use a fresh output dir for V2")
    bank = ParetoGeneBank(capacity=config["archive_capacity"],
                          depth=config["archive_depth"])
    for record, params in zip(meta["genes"][:meta["bank_size"]], mat[:meta["bank_size"]]):
        gene = Gene.from_dict(record, params)
        bank.genes[gene.gene_id] = gene
    bank.admissions = int(meta["admissions"])
    bank.density_evictions = int(meta["density_evictions"])
    bank.dominated_rejections = int(meta["dominated_rejections"])
    ibest = int(meta["bank_size"])
    best = Gene.from_dict(meta["genes"][ibest], mat[ibest]) if len(mat)>ibest else None
    rng = np.random.default_rng()
    rng.bit_generator.state = meta["rng_state"]
    return bank, best, rng, list(meta["history"]), int(meta["steps"]), int(meta["round_index"]), int(meta["gene_index"])


def output_summary(out: Path, *, bank: ParetoGeneBank, best: Gene | None,
                   steps: int, config: dict, history: list[dict]) -> None:
    if best is None or not bank.genes:
        raise RuntimeError("No completed genes; cannot export policy")
    champion = {
        "best_reward": best,
        "best_progress": gene_from_archive(bank, axis="progress"),
        "best_survival": gene_from_archive(bank, axis="survival"),
        "balanced": gene_from_archive(bank, axis="balanced"),
    }
    for name, gene in champion.items():
        write_policy(out/f"{name}.npz", gene)
    columns = ["id", "source", "parent", "generation", "life_steps", "progress_mps",
               "longevity", "mean_reward", "fall_count", "censored", "admitted", "training_steps_total"]
    with (out / "candidate_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    complete = [x for x in history if not x["censored"]]
    result = {
        "algorithm": VERSION, "interaction_steps": steps,
        "config": config, "gene_proposals": len(history),
        "completed_genes": len(complete),
        "censored_genes": len(history) - len(complete),
        "average_completed_life_steps": float(np.mean([x["life_steps"] for x in complete])),
        "median_completed_life_steps": float(np.median([x["life_steps"] for x in complete])),
        "archive_size": len(bank.genes), "archive_admissions": bank.admissions,
        "max_generation": max(g.generation for g in bank.genes.values()),
        "champions": {name: {"checkpoint": f"{name}.npz", **gene.as_dict()}
                      for name, gene in champion.items()},
    }
    (out/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("interaction_steps", "completed_genes",
                    "censored_genes", "median_completed_life_steps", "archive_size")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--steps-per-gene", type=int, default=1000,
                        help="Maximum uninterrupted life horizon, in env steps")
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--archive-capacity", type=int, default=1024)
    parser.add_argument("--archive-depth", type=int, default=5)
    parser.add_argument("--de-novo", type=float, default=0.20)
    parser.add_argument("--save-every", type=int, default=250_000)
    parser.add_argument("--out-dir", type=Path, default=Path("runs_segb_v2_5m_seed0"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.de_novo <= 1 or args.archive_capacity < 3 or args.save_every < 1:
        parser.error("Invalid settings")
    # The last partial batch uses exactly as many worlds as needed.
    plan = budget_plan(args.total_steps, args.steps_per_gene, args.worlds)
    config = {
        "algorithm": VERSION, "total_steps": args.total_steps,
        "steps_per_gene": args.steps_per_gene, "worlds": args.worlds,
        "seed": args.seed, "archive_capacity": args.archive_capacity,
        "archive_depth": args.archive_depth, "de_novo": args.de_novo,
        "robot_xml_sha256": sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "dt": 0.02, "reward": "signed_radial_progress + 0.2 * alive",
        "search": "single_life_early_replace_local_mutation",
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out/"checkpoint.npz"
    if args.resume:
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint {ckpt}")
        bank, best, rng, history, steps, index, gene_index = load_checkpoint(ckpt, config)
    else:
        if ckpt.exists() or (out / "summary.json").exists():
            raise FileExistsError("Existing V2 run; use --resume or another --out-dir")
        bank = ParetoGeneBank(capacity=args.archive_capacity, depth=args.archive_depth)
        best = None
        rng = np.random.default_rng(args.seed)
        history = []
        steps = index = gene_index = 0
    print(f"SEGB V2 {args.worlds} worlds, exact budget {args.total_steps:,}, "
          f"{len(plan)} rounds, currently {steps:,} steps", flush=True)
    if steps == args.total_steps:
        output_summary(out, bank=bank, best=best, steps=steps, config=config, history=history)
        return
    last_checkpoint_step = steps
    started = time.monotonic()
    env = None
    active_world_count = 0
    initial_steps = steps
    try:
        for ri in range(index, len(plan)):
            n_worlds = plan[ri]
            if env is None or n_worlds != active_world_count:
                if env is not None:
                    env.close()
                env = make_env(n_worlds, seed=args.seed)
                active_world_count = n_worlds
            gene_index, best = run_round(
                env, bank=bank, rng=rng, gene_index=gene_index,
                max_steps=args.steps_per_gene, horizon=args.steps_per_gene,
                seed=args.seed + 100000 + ri * 37,
                de_novo_fraction=args.de_novo, history=history,
                base_steps=steps, best=best,
            )
            steps += n_worlds * args.steps_per_gene
            if (ri+1) % 10 == 0 or steps == args.total_steps:
                elapsed = max(time.monotonic()-started, 1e-9)
                count_completed = sum(not h["censored"] for h in history)
                top_survival = max((g.longevity for g in bank.genes.values()), default=0.0)
                print(f"ROUND {ri+1}/{len(plan)} | steps {steps:,}/{args.total_steps:,} "
                      f"| completed {count_completed:,} "
                      f"| archive {len(bank.genes)} "
                      f"| best_survival {top_survival:.3f} "
                      f"| {(steps-initial_steps)/elapsed:.0f} steps/s", flush=True)
            if steps-last_checkpoint_step >= args.save_every or steps == args.total_steps:
                save_checkpoint(ckpt, bank=bank, best=best, rng=rng,
                                history=history, config=config, steps=steps,
                                round_index=ri+1, gene_index=gene_index)
                last_checkpoint_step = steps
    finally:
        if env is not None:
            env.close()
    if steps != args.total_steps:
        raise RuntimeError(f"Budget mismatch: {steps} != {args.total_steps}")
    output_summary(out, bank=bank, best=best, steps=steps, config=config, history=history)
    print(f"Saved policies & diagnostics to {out}")


if __name__ == "__main__":
    main()
