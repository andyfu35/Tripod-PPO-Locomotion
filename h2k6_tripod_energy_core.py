"""Three-axis energy experiment: tripod SEGB H2-K6 with private relative psi.

Reference semantics: Self-Evolving-Gene-Bank, sebg_full.py and
scripts/h2k6_relative_memory.py on experiment/segb-paes-adaptive-archive-20261017.
Differences: 3 axes (radial progress, uninterrupted life, energy-aware progress),
and CPU MuJoCo with GRU26. No PPO and no gait targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np

OBS = 51
HIDDEN = 26
ACT = 12
HORIZON = 1000
PARAM_COUNT = (OBS*HIDDEN + HIDDEN + 2*3*HIDDEN*HIDDEN +
               2*3*HIDDEN + HIDDEN*HIDDEN + HIDDEN + HIDDEN*ACT + ACT)
SIGMA_MIN = 0.01
SIGMA_MAX = 30.0
GRID_DEPTH = 5
ARCHIVE_CAPACITY = 1024


def sigmoid(x):
    x = np.clip(x, -45.0, 45.0)
    return 1.0/(1.0+np.exp(-x))


def silu(x):
    return x*sigmoid(x)


def unpack_parameters(parameters: np.ndarray) -> dict[str, np.ndarray]:
    """Same serialization and weight orientation as original RecurrentPolicyGenome(gru)."""
    p = np.asarray(parameters, dtype=np.float32)
    if p.shape != (PARAM_COUNT,) or not np.isfinite(p).all():
        raise ValueError(f"GRU26 policy expects {PARAM_COUNT} finite parameters")
    cursor = 0
    result = {}
    for key, shape in (
        ("input_weight", (HIDDEN, OBS)), ("input_bias", (HIDDEN,)),
        ("gru_weight_ih", (3*HIDDEN,HIDDEN)),
        ("gru_weight_hh", (3*HIDDEN,HIDDEN)),
        ("gru_bias_ih", (3*HIDDEN,)), ("gru_bias_hh", (3*HIDDEN,)),
        ("hidden_weight", (HIDDEN,HIDDEN)), ("hidden_bias", (HIDDEN,)),
        ("action_weight", (ACT,HIDDEN)), ("action_bias", (ACT,)),
    ):
        n = math.prod(shape)
        result[key] = p[cursor:cursor+n].reshape(shape)
        cursor += n
    assert cursor == PARAM_COUNT
    return result


class GRU26Policy:
    """NumPy port of the original CUDA GRUCell (r,z,n gate order)."""
    def __init__(self, parameters: np.ndarray):
        self.t = unpack_parameters(parameters)
        self.hidden = np.zeros(HIDDEN, dtype=np.float32)
        self.observation_space = type("ObservationSpace", (), {"shape": (OBS,)})()

    def reset(self, seed: int | None = None) -> None:
        self.hidden = np.zeros(HIDDEN, dtype=np.float32)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        x = np.asarray(obs, dtype=np.float32)
        if x.shape != (OBS,):
            raise ValueError(f"Observation shape {x.shape}, need ({OBS},)")
        t = self.t
        encoded = silu(t["input_weight"] @ x + t["input_bias"])
        ig = t["gru_weight_ih"] @ encoded + t["gru_bias_ih"]
        hg = t["gru_weight_hh"] @ self.hidden + t["gru_bias_hh"]
        ir, iz, inn = np.split(ig, 3)
        hr, hz, hn = np.split(hg, 3)
        r = sigmoid(ir+hr)
        z = sigmoid(iz+hz)
        n = np.tanh(inn + r * hn)
        self.hidden = ((1-z)*n + z*self.hidden).astype(np.float32)
        features = silu(t["hidden_weight"] @ self.hidden + t["hidden_bias"])
        action = np.tanh(t["action_weight"] @ features + t["action_bias"])
        return action.astype(np.float32), None

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            if str(f["architecture"].item()) != "gru26_h2k6_tripod":
                raise ValueError("Not an H2K6 Tripod GRU26 policy")
            return cls(f["parameters"])


def random_parameters(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 0.05, PARAM_COUNT).astype(np.float32)


def seeded_direction(seed: int) -> np.ndarray:
    """Exact original SEGB full-policy signed Rademacher direction / sqrt(N)."""
    rng = np.random.default_rng(int(seed))
    signs = rng.integers(0, 2, size=(PARAM_COUNT,), dtype=np.int8).astype(np.float32)
    return (signs*2.0-1.0)/np.float32(np.sqrt(float(PARAM_COUNT)))


def mutate(parent: np.ndarray, direction_seed: int, signed_scale: float) -> np.ndarray:
    if not np.isfinite(signed_scale) or signed_scale == 0:
        raise ValueError("Invalid signed mutation scale")
    return (np.asarray(parent, np.float32) +
            np.float32(signed_scale)*seeded_direction(direction_seed)).astype(np.float32)


@dataclass
class Evidence:
    direction_seed: int
    signed_scale: float
    delta: tuple[float, float, float]
    parent_steps: int
    child_steps: int
    survived: bool
    admitted: bool
    reset_seed: int

    def __post_init__(self):
        if not 1 <= self.parent_steps <= HORIZON or not 1 <= self.child_steps <= HORIZON:
            raise ValueError("Invalid H2-K6 paired life steps")
        if not math.isfinite(self.signed_scale) or self.signed_scale == 0:
            raise ValueError("Invalid signed mutation")
        if len(self.delta) != 3 or not all(-1 <= x <= 1 and math.isfinite(x) for x in self.delta):
            raise ValueError("Invalid clipped two-axis capability change")

    def useful_axes(self) -> tuple[bool, bool, bool]:
        stable = self.child_steps >= self.parent_steps
        # Lifetime requires at least two *physical* control steps improvement.
        life = self.child_steps >= self.parent_steps + 2
        progress = stable and self.delta[0] > 1e-4
        # A lower-power mutation must NOT be counted as useful if it falls sooner.
        energy_aware_progress = stable and self.delta[2] > 1e-4
        return progress, life, energy_aware_progress

    def to_dict(self):
        return {"direction_seed": self.direction_seed,
                "signed_scale": self.signed_scale, "delta": list(self.delta),
                "parent_steps": self.parent_steps, "child_steps": self.child_steps,
                "survived": self.survived, "admitted": self.admitted,
                "reset_seed": self.reset_seed}

    @classmethod
    def from_dict(cls, d):
        return cls(**{**d, "delta": tuple(d["delta"])})


@dataclass
class PrivatePsi:
    owner: str
    lineage: str
    parent: str | None = None
    evidence: list[Evidence] = field(default_factory=list)

    def inherited_by(self, child_id: str, parent_id: str) -> "PrivatePsi":
        if parent_id != self.owner:
            raise ValueError("Attempted cross-owner private memory inheritance")
        return PrivatePsi(owner=child_id, lineage=self.lineage, parent=parent_id,
                          evidence=[Evidence.from_dict(e.to_dict()) for e in self.evidence])

    def to_dict(self):
        return {"owner": self.owner, "lineage": self.lineage, "parent": self.parent,
                "evidence": [e.to_dict() for e in self.evidence]}

    @classmethod
    def from_dict(cls, d):
        return cls(owner=d["owner"], lineage=d["lineage"], parent=d["parent"],
                   evidence=[Evidence.from_dict(e) for e in d["evidence"]])


def propose_relative(state: PrivatePsi, rng: np.random.Generator, *,
                     sigma_min: float = SIGMA_MIN, sigma_max: float = SIGMA_MAX,
                     random_fraction: float = 0.20) -> tuple[int, float, str]:
    """Adapt original H2-K6: keep signed negative history; choose axis locally.

    No global direction learner; no history sharing across lineages.
    The positive-axis rule becomes [progress, life, energy-aware progress].
    """
    positives = [e for e in state.evidence if any(e.useful_axes())]
    used = {(e.direction_seed, e.signed_scale) for e in state.evidence}
    used_seed = {e.direction_seed for e in state.evidence}
    if not positives or rng.random() < random_fraction:
        for _ in range(256):
            seed = int(rng.integers(1, 2**31-1))
            if seed not in used_seed:
                mag = float(np.exp(rng.uniform(np.log(sigma_min), np.log(sigma_max))))
                sign = 1.0 if rng.random() < 0.5 else -1.0
                return seed, sign*mag, "bootstrap_random" if not positives else "persistent_random"
        raise RuntimeError("No unused discovery direction")
    active_axes = [j for j in range(3) if any(e.useful_axes()[j] for e in positives)]
    j = int(rng.choice(active_axes))
    eligible = [e for e in positives if e.useful_axes()[j]]
    failures: dict[tuple[int, float], int] = {}
    for e in state.evidence:
        if not e.useful_axes()[j]:
            k = (e.direction_seed, math.copysign(1., e.signed_scale))
            failures[k] = failures.get(k, 0) + 1
    weights = np.asarray([1./(1+failures.get((e.direction_seed,
                             math.copysign(1., e.signed_scale)), 0))
                          for e in eligible], dtype=np.float64)
    weights /= weights.sum()
    for _ in range(256):
        basis = eligible[int(rng.choice(len(eligible), p=weights))]
        factor = float(np.exp(rng.uniform(np.log(.5), np.log(2.0))))
        magnitude = float(np.clip(abs(basis.signed_scale)*factor, sigma_min, sigma_max))
        signed = math.copysign(magnitude, basis.signed_scale)
        if (basis.direction_seed, signed) not in used:
            return basis.direction_seed, signed, "relative_guided"
    # Original algorithm raises after exhausting duplicate attempts; preserve exploration instead.
    return propose_relative(PrivatePsi(state.owner, state.lineage), rng,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            random_fraction=1.0)


@dataclass
class Gene:
    id: str
    theta: np.ndarray
    psi: PrivatePsi
    parent_id: str | None = None
    generation: int = 0
    progress: float = -np.inf
    longevity: float = -np.inf
    reward: float = -np.inf
    energy_aware_progress: float = -np.inf
    motor_energy_j: float = 0.0
    steps: int = 0
    seed: int = 0

    def metadata(self) -> dict[str, Any]:
        return {"id": self.id, "psi": self.psi.to_dict(),
                "parent_id": self.parent_id, "generation": self.generation,
                "progress": self.progress, "longevity": self.longevity,
                "reward": self.reward, "energy_aware_progress": self.energy_aware_progress,
                "motor_energy_j": self.motor_energy_j, "steps": self.steps, "seed": self.seed}

    @classmethod
    def restore(cls, data: dict[str, Any], params: np.ndarray) -> "Gene":
        return cls(id=data["id"], psi=PrivatePsi.from_dict(data["psi"]),
                   theta=np.asarray(params, dtype=np.float32).copy(),
                   parent_id=data["parent_id"], generation=int(data["generation"]),
                   progress=float(data["progress"]), longevity=float(data["longevity"]),
                   reward=float(data["reward"]),
                   energy_aware_progress=float(data["energy_aware_progress"]),
                   motor_energy_j=float(data["motor_energy_j"]),
                   steps=int(data["steps"]), seed=int(data["seed"]))

    @property
    def capability(self):
        return (self.progress, self.longevity, self.energy_aware_progress)


def q2_parent_probabilities(genes: list[Gene]) -> np.ndarray:
    if not genes:
        raise ValueError("No parents")
    if len(genes) == 1:
        return np.ones(1, dtype=np.float64)
    matrix = np.array([g.capability for g in genes], dtype=np.float64)
    scores = np.empty_like(matrix)
    for j in range(matrix.shape[1]):
        v = matrix[:, j]
        sorted_v = np.sort(v)
        left = np.searchsorted(sorted_v, v, side="left")
        right = np.searchsorted(sorted_v, v, side="right")
        scores[:, j] = (left + right - 1)*.5/(len(genes)-1)
    # Original SEGB qweighted_probabilities uses (sum scores)^Q, zero weights allowed.
    weights = scores.sum(axis=1)**2
    if weights.sum() <= 0:
        return np.full(len(genes), 1./len(genes))
    return weights / weights.sum()


def dominates(a: Gene, b: Gene) -> bool:
    aa, bb = np.asarray(a.capability), np.asarray(b.capability)
    return bool(np.all(aa >= bb) and np.any(aa > bb))


class PAESArchive:
    """Nondominated archive + adaptive dense-cell thinning, same SEGB PAES principle."""
    def __init__(self, capacity: int = ARCHIVE_CAPACITY, depth: int = GRID_DEPTH):
        self.capacity = capacity
        self.depth = depth
        self.genes: dict[str, Gene] = {}
        self.admitted = 0
        self.evictions = 0

    def ordered(self) -> list[Gene]:
        return [self.genes[k] for k in sorted(self.genes)]

    def commit_batch(self, proposed: list[Gene], rng: np.random.Generator) -> set[str]:
        old_ids = set(self.genes)
        combined = self.ordered() + list(proposed)
        front = []
        for g in combined:
            if not np.all(np.isfinite(g.capability)):
                raise ValueError("Nonfinite capability in archive")
            if not any(dominates(other, g) for other in combined if other.id != g.id):
                front.append(g)
        seen = set()
        front_unique = []
        for g in sorted(front, key=lambda x: (x.id not in old_ids, x.id)):
            if g.capability not in seen:
                seen.add(g.capability)
                front_unique.append(g)
        if len(front_unique) > self.capacity:
            m = np.array([g.capability for g in front_unique], float)
            low, high = m.min(0), m.max(0)
            normalized = np.divide(m-low, high-low,
                                   out=np.zeros_like(m), where=(high-low)>1e-12)
            cells = np.floor(normalized*(1 << self.depth)).astype(int)
            cells = np.minimum(cells, (1 << self.depth)-1)
            by_cell: dict[tuple[int, ...], list[int]] = {}
            for i, c in enumerate(cells):
                by_cell.setdefault(tuple(c), []).append(i)
            keep = set(range(len(front_unique)))
            while len(keep) > self.capacity:
                largest = max(len(v) for v in by_cell.values())
                densest = sorted(k for k, v in by_cell.items() if len(v) == largest)
                victim_cell = densest[int(rng.integers(len(densest)))]
                pos = int(rng.integers(len(by_cell[victim_cell])))
                victim = by_cell[victim_cell].pop(pos)
                if not by_cell[victim_cell]:
                    del by_cell[victim_cell]
                keep.remove(victim)
                self.evictions += 1
            front_unique = [g for i,g in enumerate(front_unique) if i in keep]
        self.genes = {g.id: g for g in front_unique}
        accepted = {g.id for g in proposed} & set(self.genes)
        self.admitted += len(accepted)
        return accepted


def export_policy(path, gene: Gene) -> None:
    with open(path, "wb") as f:
        np.savez_compressed(f, parameters=gene.theta,
                            architecture=np.asarray("gru26_h2k6_tripod"),
                            meta=np.asarray(__import__("json").dumps(gene.metadata())))
