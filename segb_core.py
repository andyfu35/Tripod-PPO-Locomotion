"""Tripod adaptation of Self-Evolving Gene Bank, with a strict 5M-step ledger.

Algorithm design follows the earlier SEGB benchmark: Q^2 parent sampling,
80% inherited mutation / 20% de novo, nondominated archive, and PAES-style
adaptive grid thinning at capacity 1024, depth 5. This is a tripod-specific
MLP genotype, not a byte-for-byte port of the Hopper recurrent genome.

No MuJoCo or Stable-Baselines3 dependency in this module; it can be tested offline.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

OBS_DIM = 51
HIDDEN_DIM = 64
ACT_DIM = 12
GENOME_LENGTH = OBS_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * ACT_DIM + ACT_DIM


def decode_parameters(parameters: np.ndarray) -> tuple[np.ndarray, ...]:
    p = np.asarray(parameters, dtype=np.float32)
    if p.shape[-1] != GENOME_LENGTH:
        raise ValueError(f"Genome requires {GENOME_LENGTH} parameters; got {p.shape}")
    dims = ((OBS_DIM, HIDDEN_DIM), (HIDDEN_DIM,), (HIDDEN_DIM, HIDDEN_DIM),
            (HIDDEN_DIM,), (HIDDEN_DIM, ACT_DIM), (ACT_DIM,))
    result: list[np.ndarray] = []
    offset = 0
    for shape in dims:
        size = int(np.prod(shape))
        result.append(p[..., offset:offset + size].reshape(*p.shape[:-1], *shape))
        offset += size
    return tuple(result)


def batched_actions(observations: np.ndarray, genomes: np.ndarray) -> np.ndarray:
    """One deterministic policy per world, sharing one network architecture."""
    obs = np.asarray(observations, dtype=np.float32)
    p = np.asarray(genomes, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM or p.shape != (len(obs), GENOME_LENGTH):
        raise ValueError(f"Expected (worlds, {OBS_DIM}) and (worlds, {GENOME_LENGTH})")
    w1, b1, w2, b2, w3, b3 = decode_parameters(p)
    h1 = np.tanh(np.einsum("bi,bij->bj", obs, w1, optimize=False) + b1)
    h2 = np.tanh(np.einsum("bi,bij->bj", h1, w2, optimize=False) + b2)
    out = np.tanh(np.einsum("bi,bij->bj", h2, w3, optimize=False) + b3)
    return out.astype(np.float32, copy=False)


def random_parameters(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 0.10, size=GENOME_LENGTH).astype(np.float32)


@dataclass
class Gene:
    gene_id: int
    parameters: np.ndarray
    parent_id: int | None = None
    lineage_id: int | None = None
    generation: int = 0
    mutation_seed: int | None = None
    mutation_sigma: float | None = None
    guide_seed: int | None = None
    guide_sigma: float | None = None
    progress_mps: float = -float("inf")
    longevity: float = -float("inf")
    mean_reward: float = -float("inf")
    fall_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {k: (int(v) if isinstance(v, np.integer) else v) for k, v in vars(self).items() if k != "parameters"}

    @classmethod
    def from_dict(cls, meta: dict[str, Any], parameters: np.ndarray) -> "Gene":
        return cls(**{**meta, "parameters": np.asarray(parameters, dtype=np.float32).copy()})

    @property
    def objectives(self) -> tuple[float, float]:
        return (self.progress_mps, self.longevity)


def dominates(a: Gene, b: Gene, atol: float = 1e-10) -> bool:
    x = np.asarray(a.objectives, dtype=float)
    y = np.asarray(b.objectives, dtype=float)
    return bool(np.all(x >= y - atol) and np.any(x > y + atol))


def q2_probabilities(genes: list[Gene]) -> np.ndarray:
    """Per-objective empirical midranks + canonical (0.10 + sum(axis))**2."""
    if not genes:
        raise ValueError("Cannot choose from an empty gene bank")
    if len(genes) == 1:
        return np.ones(1, dtype=float)
    values = np.asarray([g.objectives for g in genes], dtype=np.float64)
    scores = np.empty_like(values)
    for axis in range(2):
        sorted_values = np.sort(values[:, axis])
        lo = np.searchsorted(sorted_values, values[:, axis], side="left")
        hi = np.searchsorted(sorted_values, values[:, axis], side="right")
        scores[:, axis] = ((lo + hi - 1) * 0.5) / (len(genes) - 1)
    q = (0.10 + scores.sum(axis=1)) ** 2
    return q / q.sum()


def adaptive_cells(genes: list[Gene], depth: int = 5) -> list[tuple[int, int]]:
    if not genes:
        return []
    value = np.asarray([g.objectives for g in genes], dtype=np.float64)
    low = np.min(value, axis=0)
    span = np.max(value, axis=0) - low
    scaled = np.divide(value - low, span, out=np.zeros_like(value), where=span > 1e-12)
    grid = np.minimum(np.floor(scaled * (1 << depth)).astype(int), (1 << depth) - 1)
    return [tuple(map(int, pair)) for pair in grid]


class ParetoGeneBank:
    def __init__(self, capacity: int = 1024, depth: int = 5) -> None:
        if capacity < 3 or not (1 <= depth <= 12):
            raise ValueError("Invalid PAES-style archive settings")
        self.capacity = capacity
        self.depth = depth
        self.genes: dict[int, Gene] = {}
        self.admissions = 0
        self.density_evictions = 0
        self.dominated_rejections = 0

    def ordered(self) -> list[Gene]:
        return [self.genes[k] for k in sorted(self.genes)]

    def accept(self, candidate: Gene, rng: np.random.Generator) -> bool:
        values = candidate.objectives
        if not all(np.isfinite(x) for x in values):
            raise ValueError("Nonfinite Gene capabilities")
        old = self.ordered()
        for gene in old:
            if dominates(gene, candidate):
                self.dominated_rejections += 1
                return False
            if np.allclose(gene.objectives, candidate.objectives, atol=1e-10, rtol=0):
                if gene.mean_reward >= candidate.mean_reward:
                    self.dominated_rejections += 1
                    return False
        keep = {gene.gene_id: gene for gene in old if not dominates(candidate, gene)
                and not np.allclose(gene.objectives, candidate.objectives, atol=1e-10, rtol=0)}
        keep[candidate.gene_id] = candidate
        if len(keep) > self.capacity:
            active = sorted(keep.values(), key=lambda g: g.gene_id)
            coordinates = adaptive_cells(active, depth=self.depth)
            occupancy: dict[tuple[int, int], list[int]] = {}
            for gene, cell in zip(active, coordinates):
                occupancy.setdefault(cell, []).append(gene.gene_id)
            while len(keep) > self.capacity:
                maximum = max(map(len, occupancy.values()))
                densest = sorted(cell for cell, ids in occupancy.items() if len(ids) == maximum)
                cell = densest[int(rng.integers(0, len(densest)))]
                members = occupancy[cell]
                victim = members.pop(int(rng.integers(0, len(members))))
                del keep[victim]
                if not members:
                    del occupancy[cell]
                self.density_evictions += 1
        admitted = candidate.gene_id in keep
        if admitted:
            self.admissions += 1
        self.genes = keep
        return admitted


def propose_gene(index: int, bank: ParetoGeneBank, rng: np.random.Generator,
                 de_novo_fraction: float = 0.20) -> Gene:
    if not 0 <= de_novo_fraction <= 1:
        raise ValueError("Invalid de novo fraction")
    if not bank.genes or rng.random() < de_novo_fraction:
        return Gene(gene_id=index, parameters=random_parameters(rng), lineage_id=index)
    pool = bank.ordered()
    parent = pool[int(rng.choice(len(pool), p=q2_probabilities(pool)))]
    guide = parent.guide_seed is not None and rng.random() < 0.65
    if guide:
        direction_seed = int(parent.guide_seed)
        sigma = float(parent.guide_sigma) * float(rng.uniform(0.50, 1.50))
    else:
        direction_seed = int(rng.integers(1, 2**31 - 1))
        sigma = float(np.exp(rng.uniform(math.log(0.008), math.log(0.16))))
    direction = np.random.default_rng(direction_seed).standard_normal(GENOME_LENGTH)
    delta = direction * sigma
    params = np.clip(parent.parameters + delta, -3.0, 3.0).astype(np.float32)
    return Gene(gene_id=index, parameters=params, parent_id=parent.gene_id,
                lineage_id=parent.lineage_id, generation=parent.generation + 1,
                mutation_seed=direction_seed, mutation_sigma=sigma,
                guide_seed=parent.guide_seed, guide_sigma=parent.guide_sigma)


def record_improvement(child: Gene, parent: Gene | None) -> None:
    """Store lineage-local evidence if *either* capability improves, survival not required."""
    if parent is None or child.mutation_seed is None:
        return
    if any(x > y + 1e-8 for x, y in zip(child.objectives, parent.objectives)):
        child.guide_seed = child.mutation_seed
        child.guide_sigma = child.mutation_sigma


def gene_from_archive(bank: ParetoGeneBank, *, axis: str) -> Gene:
    pool = bank.ordered()
    if not pool:
        raise ValueError("Bank is empty")
    if axis == "progress":
        return max(pool, key=lambda g: (g.progress_mps, g.longevity, -g.gene_id))
    if axis == "survival":
        return max(pool, key=lambda g: (g.longevity, g.progress_mps, -g.gene_id))
    if axis == "balanced":
        prob = q2_probabilities(pool)
        return pool[int(np.argmax(prob))]
    raise ValueError(axis)


def write_policy(path: Path, gene: Gene) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(f, parameters=gene.parameters, meta=np.asarray(json.dumps(gene.as_dict())))


class SEGBPredictor:
    """Minimal .predict() API matching the existing evaluate.py's PPO path."""
    def __init__(self, parameters: np.ndarray):
        self.parameters = np.asarray(parameters, dtype=np.float32).reshape(GENOME_LENGTH)
        # Intentionally avoid importing gymnasium in the SEGB algorithm core.
        self.observation_space = type("ObservationShape", (), {"shape": (OBS_DIM,)})()

    @classmethod
    def load(cls, filename: str | Path) -> "SEGBPredictor":
        with np.load(filename, allow_pickle=False) as data:
            return cls(data["parameters"])

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        obs = np.asarray(observation, dtype=np.float32).reshape(1, OBS_DIM)
        return batched_actions(obs, self.parameters.reshape(1, -1))[0], None


def save_training_state(path: Path, *, bank: ParetoGeneBank, all_best: Gene | None,
                        next_gene_index: int, rng: np.random.Generator,
                        configuration: dict[str, Any], history: list[dict[str, Any]]) -> None:
    pool = bank.ordered()
    best = [all_best] if all_best is not None else []
    n = pool + best
    matrix = np.stack([g.parameters for g in n], axis=0) if n else np.zeros((0, GENOME_LENGTH), np.float32)
    meta = {
        "format_version": 1,
        "bank_size": len(pool),
        "genes": [g.as_dict() for g in n],
        "next_gene_index": next_gene_index,
        "rng_state": rng.bit_generator.state,
        "configuration": configuration,
        "history": history,
        "admissions": bank.admissions,
        "density_evictions": bank.density_evictions,
        "dominated_rejections": bank.dominated_rejections,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, parameters=matrix, meta=np.asarray(json.dumps(meta)))
    tmp.replace(path)


def load_training_state(path: Path, configuration: dict[str, Any]) -> tuple[ParetoGeneBank, Gene | None, int, np.random.Generator, list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
        matrix = np.asarray(data["parameters"], dtype=np.float32)
    if meta["format_version"] != 1 or meta["configuration"] != configuration:
        raise ValueError("Checkpoint configuration/robot-model differs; cannot resume")
    bank = ParetoGeneBank(capacity=configuration["archive_capacity"], depth=configuration["archive_depth"])
    n = meta["bank_size"]
    for item, p in zip(meta["genes"][:n], matrix[:n]):
        g = Gene.from_dict(item, p)
        bank.genes[g.gene_id] = g
    best = Gene.from_dict(meta["genes"][n], matrix[n]) if len(meta["genes"]) > n else None
    bank.admissions = meta["admissions"]
    bank.density_evictions = meta["density_evictions"]
    bank.dominated_rejections = meta["dominated_rejections"]
    rng = np.random.default_rng()
    rng.bit_generator.state = meta["rng_state"]
    return bank, best, int(meta["next_gene_index"]), rng, list(meta["history"])
