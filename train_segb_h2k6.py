"""Tripod SEGB H2-K6: private paired relative-history learning, exact step budget.

8 spawned CPU worker processes x 2 genuinely independent MuJoCo instances = 16
parallel simulated worlds. Pair parent/child at identical reset seeds. Their
actual env.step calls BOTH count toward the 5,000,000 interaction budget.

The original SEGB's lineage-local private psi, negative signed evidence,
Q^2 parent choice, 20% de-novo, 20% inner exploration, PAES 1024 depth 5,
and GRU architecture are retained. The H2-K6 4 Hopper axes are adapted to
(physical radial-progress speed, fraction of uninterrupted life).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from hashlib import sha256
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from h2k6_tripod_core import (
    ACT, HIDDEN, HORIZON, PARAM_COUNT, OBS,
    Evidence, Gene, GRU26Policy, PAESArchive, PrivatePsi,
    export_policy, mutate, propose_relative, q2_parent_probabilities,
    random_parameters,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tripod.xml"
PAIR_ENVS: tuple[Any, Any] | None = None


@dataclass(frozen=True)
class Config:
    total_steps: int = 5_000_000
    worlds: int = 16
    horizon: int = HORIZON
    seed: int = 0
    founders: int = 16
    de_novo_fraction: float = 0.20
    persistent_random_fraction: float = 0.20
    sigma_min: float = 0.01
    sigma_max: float = 30.0
    archive_capacity: int = 1024
    archive_depth: int = 5
    checkpoint_every: int = 250_000
    output: str = "runs_segb_h2k6_5m_seed0"

    def validate(self) -> None:
        if self.worlds != 16 or self.worlds % 2:
            raise ValueError("Formal comparison requires 16 worlds = 8 paired workers")
        if self.horizon != 1000 or self.founders != 16:
            raise ValueError("Invalid horizon/founders")
        if self.total_steps < 2 or self.checkpoint_every < 1:
            raise ValueError("Invalid interaction budget/checkpoint interval")
        if not (0 <= self.de_novo_fraction <= 1 and
                0 <= self.persistent_random_fraction <= 1):
            raise ValueError("Invalid random fractions")
        if not 0 < self.sigma_min < self.sigma_max or self.archive_capacity < 3:
            raise ValueError("Invalid mutation or archive bounds")
        if not 1 <= self.archive_depth <= 12:
            raise ValueError("Invalid archive grid depth")


def _make_worlds():
    # Delayed import: the pure-NumPy tests work without MuJoCo installed.
    from tripod_env import TripodEnv
    return TripodEnv(max_episode_steps=HORIZON), TripodEnv(max_episode_steps=HORIZON)


def eval_pair(task: dict, *, envs=None) -> dict:
    """One unbiased paired physical experiment, with an optional budget cap.

    Both worlds start from the SAME reset seed; never simulate a done world
    again merely to fill a rectangular rollout. A partial trial is CENSORED.
    """
    global PAIR_ENVS
    if envs is None:
        if PAIR_ENVS is None:
            PAIR_ENVS = _make_worlds()
        envs = PAIR_ENVS
    if len(envs) != 2:
        raise ValueError("Each worker requires two independent environments")
    limit = int(task["budget_cap"])
    horizon = int(task["horizon"])
    if not 1 <= limit <= 2*horizon:
        raise ValueError("Invalid paired budget cap")
    reset_seed = int(task["reset_seed"])
    policies = [GRU26Policy(np.asarray(v, np.float32)) for v in task["theta"]]
    observations = []
    for env, policy in zip(envs, policies):
        obs, _ = env.reset(seed=reset_seed)
        policy.reset(reset_seed)
        observations.append(obs)
    data = [dict(steps=0, reward=0.0, distance=0.0,
                 fell=False, completed=False, censored=False)
            for _ in range(2)]
    used = 0
    for _ in range(horizon):
        for w in range(2):
            if data[w]["completed"] or used >= limit:
                continue
            action, _ = policies[w].predict(observations[w])
            obs, reward, terminated, truncated, info = envs[w].step(action)
            observations[w] = obs
            used += 1
            d = data[w]
            d["steps"] += 1
            d["reward"] += float(reward)
            d["distance"] = float(info["net_distance_m"])
            d["fell"] = bool(terminated)
            d["completed"] = bool(terminated or truncated or d["steps"] >= horizon)
        if all(d["completed"] for d in data) or used >= limit:
            break
    for d in data:
        d["censored"] = not d["completed"]
    assert 1 <= used <= limit
    return {"used": used, "pair_seed": reset_seed, "results": data,
            "child_id": task["child_id"], "parent_id": task["parent_id"]}


def close_worker_envs():
    global PAIR_ENVS
    if PAIR_ENVS is not None:
        for e in PAIR_ENVS:
            e.close()
        PAIR_ENVS = None


def score_gene(g: Gene, r: dict, horizon: int, seed: int) -> None:
    if r["censored"]:
        raise ValueError("Censored gene must never enter the archive")
    g.progress = float(r["distance"]/(horizon*0.02))
    g.longevity = float(r["steps"] / horizon)
    g.reward = float(r["reward"] / horizon)
    g.steps = int(r["steps"])
    g.seed = int(seed)


def make_founder(gid: str, rng: np.random.Generator) -> Gene:
    state = PrivatePsi(owner=gid, lineage=gid)
    return Gene(gid, random_parameters(rng), state)


def prepare_batch(cfg: Config, bank: PAESArchive, rng: np.random.Generator,
                  gene_counter: int, batch_size: int, budget_remaining: int):
    """Every task has ≤2*horizon reserved steps; no hidden physics freebie."""
    tasks, entries = [], []
    pool = bank.ordered()
    q2 = q2_parent_probabilities(pool) if pool else None
    for k in range(batch_size):
        child_id = f"g{gene_counter + 2*k + 1:08d}"
        if not pool:
            parent_id = child_id
            child_id = f"g{gene_counter + 2*k + 1:08d}"
            parent = make_founder(parent_id, rng)
            child = make_founder(child_id, rng)
            seed = sig = None
            mode = "founder"
        elif rng.random() < cfg.de_novo_fraction:
            # Exactly one independent newborn in 20% of offspring slots;
            # replay a Q2-selected parent on the paired same-reset world.
            parent = pool[int(rng.choice(len(pool), p=q2))]
            parent_id = parent.id
            child = make_founder(child_id, rng)
            seed = sig = None
            mode = "de_novo"
        else:
            parent = pool[int(rng.choice(len(pool), p=q2))]
            seed, sig, mode = propose_relative(
                parent.psi, rng, sigma_min=cfg.sigma_min,
                sigma_max=cfg.sigma_max,
                random_fraction=cfg.persistent_random_fraction,
            )
            child = Gene(child_id, mutate(parent.theta, seed, sig),
                         PrivatePsi(owner=child_id, lineage=parent.psi.lineage,
                                    parent=parent.id),
                         parent_id=parent.id, generation=parent.generation+1)
            parent_id = parent.id
        cap = min(2*cfg.horizon, budget_remaining - 2*cfg.horizon*k)
        if cap <= 0:
            raise AssertionError("Overbooked physics interactions")
        reset_seed = int(rng.integers(1_000_000, 2**31-1))
        tasks.append({"parent_id": parent_id, "child_id": child_id,
                      "theta": (parent.theta, child.theta), "reset_seed": reset_seed,
                      "budget_cap": cap, "horizon": cfg.horizon})
        entries.append(dict(parent=parent, child=child, seed=seed,
                            signed_scale=sig, mode=mode, reset_seed=reset_seed))
    return tasks, entries


def commit_results(entries: list[dict], results: list[dict], *,
                   bank: PAESArchive, rng: np.random.Generator,
                   histories: list[dict], steps_used: int,
                   best_reward: Gene | None) -> Gene | None:
    """Update every parent's psi first, then atomic bank selection/inheritance.

    Thus a rejected child still teaches its parent; all accepted siblings inherit
    the *entire* updated private parent history, as in original SEGB Full.
    """
    proposals = []
    batch_history_start = len(histories)
    evidence_for_child = {}
    for e, result in zip(entries, results):
        parent, child = e["parent"], e["child"]
        parent_result, child_result = result["results"]
        if e["mode"] == "founder":
            # Before a bank exists, both paired worlds test NEW founders.
            for g, r in zip((parent, child), (parent_result, child_result)):
                if r["completed"]:
                    score_gene(g, r, HORIZON, result["pair_seed"])
                    proposals.append(g)
                    if best_reward is None or g.reward > best_reward.reward:
                        best_reward = g
                histories.append({"gene_id": g.id, "parent_id": "",
                                  "mode": "founder", "steps": r["steps"],
                                  "fell": int(r["fell"]), "censored": int(r["censored"]),
                                  "progress_mps": r["distance"]/(1000*.02),
                                  "life_fraction": r["steps"]/1000,
                                  "physics_steps_total": steps_used})
        elif e["mode"] == "de_novo":
            # The accompanying parent run is a charged baseline only: it is
            # NOT a new Gene. Preserve its stored bank fitness and private psi.
            r = child_result
            if r["completed"]:
                score_gene(child, r, HORIZON, result["pair_seed"])
                proposals.append(child)
                if best_reward is None or child.reward > best_reward.reward:
                    best_reward = child
            histories.append({"gene_id": child.id, "parent_id": "",
                              "mode": "de_novo", "steps": r["steps"],
                              "parent_steps": parent_result["steps"],
                              "fell": int(r["fell"]), "censored": int(r["censored"]),
                              "progress_mps": r["distance"]/(1000*.02),
                              "life_fraction": r["steps"]/1000,
                              "physics_steps_total": steps_used})
        else:
            if parent_result["completed"] and child_result["completed"]:
                score_gene(child, child_result, 1000, result["pair_seed"])
                # Reference H2-K6 uses capability-normalized signed differences
                # and paired physical survival counts in every evidence record.
                parent_progress = parent_result["distance"]/(1000*.02)
                parent_longevity = parent_result["steps"]/1000
                delta = (float(np.clip(child.progress-parent_progress, -1, 1)),
                         float(np.clip(child.longevity-parent_longevity, -1, 1)))
                evidence = Evidence(direction_seed=int(e["seed"]),
                                    signed_scale=float(e["signed_scale"]),
                                    delta=delta, parent_steps=parent_result["steps"],
                                    child_steps=child_result["steps"],
                                    survived=not child_result["fell"], admitted=False,
                                    reset_seed=result["pair_seed"])
                parent.psi.evidence.append(evidence)
                evidence_for_child[child.id] = (parent, evidence)
                proposals.append(child)
                if best_reward is None or child.reward > best_reward.reward:
                    best_reward = child
            histories.append({"gene_id": child.id, "parent_id": parent.id,
                              "mode": e["mode"], "steps": child_result["steps"],
                              "parent_steps": parent_result["steps"],
                              "fell": int(child_result["fell"]),
                              "censored": int(child_result["censored"] or parent_result["censored"]),
                              "progress_mps": child_result["distance"]/(1000*.02),
                              "life_fraction": child_result["steps"]/1000,
                              "physics_steps_total": steps_used})
    if proposals:
        admitted = bank.commit_batch(proposals, rng)
        for child_id, (parent, evidence) in evidence_for_child.items():
            evidence.admitted = child_id in admitted
            if child_id in admitted:
                child = bank.genes[child_id]
                child.psi = parent.psi.inherited_by(child.id, parent.id)
        for item in histories[batch_history_start:]:
            item["admitted"] = int(item["gene_id"] in admitted)
    return best_reward


def save_checkpoint(path: Path, *, cfg: Config, robot_hash: str,
                    bank: PAESArchive, rng: np.random.Generator,
                    steps: int, next_id: int, batch: int,
                    history: list[dict], best: Gene | None):
    records = bank.ordered()
    if best is not None and best.id not in bank.genes:
        records += [best]
    mat = np.stack([x.theta for x in records]) if records else np.zeros((0, PARAM_COUNT), np.float32)
    meta = {"version": "tripod_h2k6_private_psi_1", "config": asdict(cfg),
            "robot_hash": robot_hash, "steps": steps, "next_id": next_id,
            "batch": batch, "rng_state": rng.bit_generator.state,
            "bank_size": len(bank.genes), "records": [x.metadata() for x in records],
            "best_id": best.id if best is not None else None,
            "history": history, "admitted": bank.admitted, "evictions": bank.evictions}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, params=mat, meta=np.asarray(json.dumps(meta)))
    tmp.replace(path)


def load_checkpoint(path: Path, *, cfg: Config, robot_hash: str):
    with np.load(path, allow_pickle=False) as f:
        matrix = np.asarray(f["params"], dtype=np.float32)
        meta = json.loads(str(f["meta"].item()))
    if (meta.get("version") != "tripod_h2k6_private_psi_1"
            or meta["config"] != asdict(cfg) or meta["robot_hash"] != robot_hash):
        raise ValueError("H2-K6 checkpoint/config or robot XML mismatch")
    bank = PAESArchive(cfg.archive_capacity, cfg.archive_depth)
    all_genes = [Gene.restore(d, p) for d,p in zip(meta["records"],matrix)]
    for g in all_genes[:meta["bank_size"]]:
        bank.genes[g.id] = g
    bank.admitted = meta["admitted"]
    bank.evictions = meta["evictions"]
    best = next((g for g in all_genes if g.id == meta["best_id"]), None)
    rng = np.random.default_rng()
    rng.bit_generator.state = meta["rng_state"]
    return bank, rng, meta["steps"], meta["next_id"], meta["batch"], list(meta["history"]), best


def summarize(output: Path, cfg: Config, bank: PAESArchive,
              history: list[dict], total_steps: int, best: Gene | None):
    if not bank.genes:
        raise RuntimeError("Archive empty; no completed candidates")
    genes = bank.ordered()
    champs = {"best_survival": max(genes,key=lambda x:(x.longevity,x.progress)),
              "best_progress": max(genes,key=lambda x:(x.progress,x.longevity)),
              "balanced": genes[int(np.argmax(q2_parent_probabilities(genes)))],
              "best_reward": best}
    # The stable-distance champion is the fastest among full-horizon survivors.
    full = [g for g in genes if g.longevity >= 1-1e-9]
    champs["best_stable_progress"] = (max(full, key=lambda x:x.progress) if full
                                       else max(genes, key=lambda x:(x.longevity,x.progress)))
    for label, g in champs.items():
        export_policy(output/f"{label}.npz", g)
    fields = ["gene_id","parent_id","mode","steps","parent_steps","fell",
              "censored","progress_mps","life_fraction","physics_steps_total","admitted"]
    with (output/"candidate_history.csv").open("w", encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(history)
    result = {
        "algorithm": "SEGB-H2K6-tripod-private-history-GRU26",
        "interaction_steps": total_steps, "config": asdict(cfg),
        "bank_size": len(bank.genes), "admitted": bank.admitted,
        "max_generation": max(g.generation for g in genes),
        "history_event_count": sum(len(g.psi.evidence) for g in genes),
        "pair_candidates": sum(bool(x.get("parent_id")) for x in history),
        "survival_of_all_full_lives": len([h for h in history if not h.get("fell",False) and not h.get("censored",False)]),
        "champions": {n:{"id":g.id, "progress_mps":g.progress,
                        "longevity":g.longevity, "generation":g.generation,
                        "history_size":len(g.psi.evidence)} for n,g in champs.items()}
    }
    (output/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("interaction_steps","bank_size",
                    "max_generation","history_event_count")},indent=2))


def train(cfg: Config, *, pool=None, verbose: bool = True):
    cfg.validate()
    out=Path(cfg.output)
    out.mkdir(parents=True,exist_ok=True)
    ckpt=out/"checkpoint.npz"
    robot_hash=sha256(MODEL_PATH.read_bytes()).hexdigest()
    if ckpt.exists():
        bank,rng,steps,next_id,batch,history,best=load_checkpoint(
            ckpt,cfg=cfg,robot_hash=robot_hash)
        if verbose:print(f"Resuming H2-K6 from {steps:,} steps")
    else:
        if (out/"summary.json").exists():
            raise FileExistsError("Output run exists without checkpoint; use new directory")
        bank=PAESArchive(cfg.archive_capacity,cfg.archive_depth)
        rng=np.random.default_rng(cfg.seed)
        steps=next_id=batch=0
        history=[]
        best=None
    owns_pool = pool is None
    if owns_pool:
        pool=mp.get_context("spawn").Pool(processes=cfg.worlds//2)
    begun=time.monotonic()
    previous_log_steps=steps
    previous_checkpoint=steps
    try:
        while steps<cfg.total_steps:
            remain=cfg.total_steps-steps
            count=min(cfg.worlds//2,max(1,remain//(2*cfg.horizon)))
            # Near the exact 5M boundary a final pair may be budget-censored.
            if remain<2*cfg.horizon:
                count=1
            tasks,entries=prepare_batch(cfg,bank,rng,next_id,count,remain)
            results=pool.map(eval_pair,tasks)
            consumed=sum(x["used"] for x in results)
            if not 0<consumed<=remain:
                raise RuntimeError("Invalid physical-interaction accounting")
            steps+=consumed
            best=commit_results(entries,results,bank=bank,rng=rng,
                                histories=history,steps_used=steps,best_reward=best)
            next_id+=2*count  # Ensure uniqueness with possible de-novo paired founders.
            batch+=1
            if verbose and (batch%50==0 or steps==cfg.total_steps):
                top=max((g.longevity for g in bank.genes.values()),default=0.0)
                movement=max((g.progress for g in bank.genes.values()),default=0.0)
                elapsed=time.monotonic()-begun
                fps=(steps-previous_log_steps)/max(elapsed,1e-8)
                print(f"PAIR_BATCH {batch} | steps {steps:,}/{cfg.total_steps:,} | "
                      f"archive {len(bank.genes)} | max-life {top:.3f} "
                      f"| max-progress {movement:.4f} m/s | {fps:.0f} recent steps/s",
                      flush=True)
                begun=time.monotonic()
                previous_log_steps=steps
            if steps-previous_checkpoint>=cfg.checkpoint_every or steps==cfg.total_steps:
                save_checkpoint(ckpt,cfg=cfg,robot_hash=robot_hash,bank=bank,
                                rng=rng,steps=steps,next_id=next_id,batch=batch,
                                history=history,best=best)
                previous_checkpoint=steps
    finally:
        if owns_pool:
            pool.close();pool.join()
    if steps!=cfg.total_steps:
        raise AssertionError("Inexact H2-K6 interaction ledger")
    summarize(out,cfg,bank,history,steps,best)
    print(f"H2-K6 training complete: {out}")
    return steps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-steps",type=int,default=5_000_000)
    p.add_argument("--worlds",type=int,default=16)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--out-dir",default="runs_segb_h2k6_5m_seed0")
    p.add_argument("--checkpoint-every",type=int,default=250_000)
    args=p.parse_args()
    cfg=Config(total_steps=args.total_steps,worlds=args.worlds,seed=args.seed,
               output=args.out_dir,checkpoint_every=args.checkpoint_every)
    train(cfg)


if __name__=="__main__":
    main()
