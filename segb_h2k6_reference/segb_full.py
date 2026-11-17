from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DirectionEvidence:
    direction_seed: int
    signed_scale: float
    delta: tuple[float, ...]
    survived: bool
    admitted: bool
    # Optional paired rollout lengths for isolated H2-K6 50M. Historical
    # DirectionEvidence payloads remain valid with both fields absent.
    parent_steps: int | None = None
    child_steps: int | None = None

    def validate(self, objective_count: int) -> None:
        if int(self.direction_seed) < 0:
            raise ValueError("direction_seed must be non-negative")
        scale = float(self.signed_scale)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("signed_scale must be finite and non-zero")
        values = np.asarray(self.delta, dtype=np.float64)
        if values.shape != (int(objective_count),):
            raise ValueError("delta dimension mismatch")
        if not np.all(np.isfinite(values)):
            raise ValueError("delta must be finite")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("normalized delta must lie in [-1,1]")
        if (self.parent_steps is None) != (self.child_steps is None):
            raise ValueError("paired parent/child steps must both be present or absent")
        if self.parent_steps is not None:
            if (type(self.parent_steps) is not int or type(self.child_steps) is not int
                    or not 1 <= self.parent_steps <= 1000
                    or not 1 <= self.child_steps <= 1000):
                raise ValueError("paired parent/child control steps outside [1,1000]")


@dataclass
class LineageEvolutionState:
    lineage_id: str
    owner_gene_id: str
    parent_gene_id: str | None = None
    random_sigma_min: float = 0.01
    random_sigma_max: float = 30.0
    guided_jitter_min: float = 0.50
    guided_jitter_max: float = 2.00
    evidence: list[DirectionEvidence] = field(default_factory=list)

    def validate(self, objective_count: int) -> None:
        if not self.lineage_id or not self.owner_gene_id:
            raise ValueError("lineage and owner IDs must be non-empty")
        if not (
            0.0 < float(self.random_sigma_min)
            < float(self.random_sigma_max)
        ):
            raise ValueError("invalid random sigma range")
        if not (
            0.0 < float(self.guided_jitter_min)
            <= float(self.guided_jitter_max)
        ):
            raise ValueError("invalid guided jitter range")
        seen: set[tuple[int, float]] = set()
        for item in self.evidence:
            item.validate(objective_count)
            key = (int(item.direction_seed), float(item.signed_scale))
            if key in seen:
                raise ValueError("duplicate direction evidence key")
            seen.add(key)

    def positive_for_axis(
        self,
        axis_index: int,
        *,
        epsilon: float = 1.0e-6,
    ) -> tuple[DirectionEvidence, ...]:
        """Return capability-local positive mutation evidence.

        Episode survival is retained on DirectionEvidence as diagnostic context,
        but it is not a prerequisite for learning a mutation direction.  A
        mutation that improves the scheduled capability axis is useful local
        evidence even when the child later terminates before the episode
        horizon.
        """
        index = int(axis_index)
        return tuple(
            item
            for item in self.evidence
            if float(item.delta[index]) > float(epsilon)
        )

    def with_observation(
        self,
        evidence: DirectionEvidence,
        *,
        objective_count: int,
    ) -> "LineageEvolutionState":
        self.validate(objective_count)
        evidence.validate(objective_count)
        key = (int(evidence.direction_seed), float(evidence.signed_scale))
        if any(
            (int(item.direction_seed), float(item.signed_scale)) == key
            for item in self.evidence
        ):
            raise ValueError("direction evidence key already exists")
        result = LineageEvolutionState(
            lineage_id=self.lineage_id,
            owner_gene_id=self.owner_gene_id,
            parent_gene_id=self.parent_gene_id,
            random_sigma_min=self.random_sigma_min,
            random_sigma_max=self.random_sigma_max,
            guided_jitter_min=self.guided_jitter_min,
            guided_jitter_max=self.guided_jitter_max,
            evidence=[*self.evidence, evidence],
        )
        result.validate(objective_count)
        return result

    def inherited_by(
        self,
        *,
        child_gene_id: str,
        parent_gene_id: str,
        objective_count: int,
    ) -> "LineageEvolutionState":
        self.validate(objective_count)
        result = LineageEvolutionState(
            lineage_id=self.lineage_id,
            owner_gene_id=str(child_gene_id),
            parent_gene_id=str(parent_gene_id),
            random_sigma_min=self.random_sigma_min,
            random_sigma_max=self.random_sigma_max,
            guided_jitter_min=self.guided_jitter_min,
            guided_jitter_max=self.guided_jitter_max,
            evidence=list(self.evidence),
        )
        result.validate(objective_count)
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "owner_gene_id": self.owner_gene_id,
            "parent_gene_id": self.parent_gene_id,
            "random_sigma_min": float(self.random_sigma_min),
            "random_sigma_max": float(self.random_sigma_max),
            "guided_jitter_min": float(self.guided_jitter_min),
            "guided_jitter_max": float(self.guided_jitter_max),
            "evidence": [
                {
                    "direction_seed": int(item.direction_seed),
                    "signed_scale": float(item.signed_scale),
                    "delta": [float(x) for x in item.delta],
                    "survived": bool(item.survived),
                    "admitted": bool(item.admitted),
                    **(
                        {"parent_steps": int(item.parent_steps),
                         "child_steps": int(item.child_steps)}
                        if item.parent_steps is not None else {}
                    ),
                }
                for item in self.evidence
            ],
        }


@dataclass(frozen=True)
class EvolutionProposal:
    source: str
    direction_seed: int
    signed_scale: float
    basis_direction_seed: int | None = None
    basis_signed_scale: float | None = None


def _log_uniform(
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
) -> float:
    lo, hi = float(minimum), float(maximum)
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def propose_random_mutation(
    state: LineageEvolutionState,
    *,
    scheduler_seed: int,
    objective_count: int,
) -> EvolutionProposal:
    """Create a fresh mutation direction regardless of existing guidance.

    This is reserved for the scheduler's persistent Random Discovery lane.
    It deliberately ignores positive capability evidence so that guided
    exploitation can never eliminate discovery of new parameter directions.
    """
    state.validate(objective_count)
    rng = np.random.default_rng(int(scheduler_seed))
    used_seeds = {int(item.direction_seed) for item in state.evidence}

    for _ in range(256):
        direction_seed = int(rng.integers(1, 2**31 - 1))
        if direction_seed in used_seeds:
            continue
        magnitude = _log_uniform(
            rng,
            state.random_sigma_min,
            state.random_sigma_max,
        )
        sign = 1.0 if bool(rng.integers(0, 2)) else -1.0
        return EvolutionProposal(
            source="persistent_random_discovery",
            direction_seed=direction_seed,
            signed_scale=float(sign * magnitude),
        )
    raise RuntimeError("unable to create unique persistent random proposal")


def propose_global_mutation(
    state: LineageEvolutionState,
    *,
    scheduler_seed: int,
    objective_count: int,
    persistent_random_fraction: float = 0.20,
    epsilon: float = 1.0e-6,
) -> EvolutionProposal:
    """Global no-lane mutation proposal.

    Every world uses the same reproduction pool. Mutation guidance is selected
    from any lineage-private direction that improved at least one capability.
    A fixed per-proposal persistent-random probability preserves discovery
    without reserving a dedicated world lane.
    """
    state.validate(objective_count)
    fraction = float(persistent_random_fraction)
    if not np.isfinite(fraction) or not (0.0 <= fraction <= 1.0):
        raise ValueError("persistent_random_fraction must lie in [0,1]")

    rng = np.random.default_rng(int(scheduler_seed))
    positive = tuple(
        item
        for item in state.evidence
        if np.any(np.asarray(item.delta, dtype=np.float64) > float(epsilon))
    )

    if not positive or float(rng.random()) < fraction:
        random_seed = int(rng.integers(1, 2**31 - 1))
        proposal = propose_random_mutation(
            state,
            scheduler_seed=random_seed,
            objective_count=objective_count,
        )
        if positive:
            return proposal
        return EvolutionProposal(
            source="bootstrap_random_no_positive_evidence",
            direction_seed=proposal.direction_seed,
            signed_scale=proposal.signed_scale,
        )

    used_keys = {
        (int(item.direction_seed), float(item.signed_scale))
        for item in state.evidence
    }
    for _ in range(256):
        basis = positive[int(rng.integers(0, len(positive)))]
        magnitude = abs(float(basis.signed_scale)) * _log_uniform(
            rng,
            state.guided_jitter_min,
            state.guided_jitter_max,
        )
        magnitude = min(
            state.random_sigma_max,
            max(state.random_sigma_min, magnitude),
        )
        signed = math.copysign(magnitude, float(basis.signed_scale))
        key = (int(basis.direction_seed), float(signed))
        if key in used_keys:
            continue
        return EvolutionProposal(
            source="global_lineage_guided",
            direction_seed=int(basis.direction_seed),
            signed_scale=float(signed),
            basis_direction_seed=int(basis.direction_seed),
            basis_signed_scale=float(basis.signed_scale),
        )
    raise RuntimeError("unable to create unique global guided proposal")


def propose_axis_mutation(
    state: LineageEvolutionState,
    *,
    axis_index: int,
    scheduler_seed: int,
    objective_count: int,
) -> EvolutionProposal:
    state.validate(objective_count)
    rng = np.random.default_rng(int(scheduler_seed))
    positive = state.positive_for_axis(axis_index)
    used_seeds = {int(item.direction_seed) for item in state.evidence}
    used_keys = {
        (int(item.direction_seed), float(item.signed_scale))
        for item in state.evidence
    }

    if positive:
        for _ in range(256):
            basis = positive[int(rng.integers(0, len(positive)))]
            magnitude = abs(float(basis.signed_scale)) * _log_uniform(
                rng,
                state.guided_jitter_min,
                state.guided_jitter_max,
            )
            magnitude = min(
                state.random_sigma_max,
                max(state.random_sigma_min, magnitude),
            )
            signed = math.copysign(magnitude, float(basis.signed_scale))
            key = (int(basis.direction_seed), float(signed))
            if key in used_keys:
                continue
            return EvolutionProposal(
                source="lineage_guided",
                direction_seed=int(basis.direction_seed),
                signed_scale=float(signed),
                basis_direction_seed=int(basis.direction_seed),
                basis_signed_scale=float(basis.signed_scale),
            )
        raise RuntimeError("unable to create unique guided proposal")

    for _ in range(256):
        direction_seed = int(rng.integers(1, 2**31 - 1))
        if direction_seed in used_seeds:
            continue
        magnitude = _log_uniform(
            rng,
            state.random_sigma_min,
            state.random_sigma_max,
        )
        sign = 1.0 if bool(rng.integers(0, 2)) else -1.0
        return EvolutionProposal(
            source="bootstrap_random_no_axis_evidence",
            direction_seed=direction_seed,
            signed_scale=float(sign * magnitude),
        )
    raise RuntimeError("unable to create unique random proposal")


@dataclass(frozen=True)
class RecurrentPolicyGenome:
    observation_dim: int
    hidden_dim: int
    action_dim: int
    parameters: np.ndarray
    # Historical Stage-1 default; MLP/LSTM only in isolated architecture trials.
    architecture: str = "gru"

    @staticmethod
    def parameter_count(
        observation_dim: int,
        hidden_dim: int,
        action_dim: int,
        *,
        architecture: str = "gru",
    ) -> int:
        obs, hidden, action = map(int, (observation_dim, hidden_dim, action_dim))
        if min(obs, hidden, action) <= 0:
            raise ValueError("policy dimensions must be positive")
        if architecture not in {"gru", "mlp", "lstm", "tqc_actor"}:
            raise ValueError("unsupported Policy architecture")
        common = obs * hidden + hidden + hidden * action + action
        hidden_layer = hidden * hidden + hidden
        if architecture == "mlp":
            return common + hidden_layer
        if architecture == "tqc_actor":
            # Same ReLU 256x256 Gaussian Actor heads as SB3-Contrib TQC,
            # mean AND log_std, but NO TQC gradient-trained Critics.
            return common + hidden_layer + (hidden * action + action)
        gates = 3 if architecture == "gru" else 4
        return common + 2 * gates * hidden * hidden + 2 * gates * hidden + hidden_layer

    @property
    def architecture_id(self) -> str:
        if self.architecture == "gru":
            return (
                f"benchmark_segb_full_mlp_gru_"
                f"obs{self.observation_dim}_h{self.hidden_dim}_a{self.action_dim}"
            )
        return (
            f"benchmark_segb_full_{self.architecture}_"
            f"obs{self.observation_dim}_h{self.hidden_dim}_a{self.action_dim}"
        )

    @classmethod
    def random(
        cls,
        *,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        seed: int,
        architecture: str = "gru",
    ) -> "RecurrentPolicyGenome":
        count = cls.parameter_count(
            observation_dim, hidden_dim, action_dim, architecture=architecture
        )
        rng = np.random.default_rng(int(seed))
        if architecture == "tqc_actor":
            # PyTorch nn.Linear reset_parameters() equivalent: uniform
            # [-1/sqrt(fan_in), +1/sqrt(fan_in)] for weights and biases.
            # Keep original deterministic parameter serialization order.
            sizes = (
                (int(hidden_dim), int(observation_dim)),
                (int(hidden_dim), int(hidden_dim)),
                (int(action_dim), int(hidden_dim)),
                (int(action_dim), int(hidden_dim)),
            )
            pieces = []
            for out_dim, in_dim in sizes:
                bound = 1.0 / math.sqrt(float(in_dim))
                pieces.append(rng.uniform(
                    -bound, bound, size=out_dim * in_dim,
                ).astype(np.float32))
                pieces.append(rng.uniform(
                    -bound, bound, size=out_dim,
                ).astype(np.float32))
            parameters = np.concatenate(pieces)
            if parameters.size != count:
                raise RuntimeError("TQC Actor initialization parameter count drift")
        else:
            parameters = rng.normal(
                0.0, 0.05, size=count
            ).astype(np.float32)
        result = cls(
            observation_dim=int(observation_dim),
            hidden_dim=int(hidden_dim),
            action_dim=int(action_dim),
            parameters=parameters,
            architecture=str(architecture),
        )
        result.validate()
        return result

    def validate(self) -> None:
        expected = self.parameter_count(
            self.observation_dim, self.hidden_dim, self.action_dim,
            architecture=self.architecture,
        )
        values = np.asarray(self.parameters)
        if values.shape != (expected,):
            raise ValueError(
                f"parameter shape mismatch {values.shape} != {(expected,)}"
            )
        if values.dtype.kind != "f" or not np.all(np.isfinite(values)):
            raise ValueError("policy parameters must be finite floats")

    @property
    def sha256(self) -> str:
        self.validate()
        digest = hashlib.sha256()
        digest.update(self.architecture_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            np.asarray(self.parameters, dtype="<f4").tobytes()
        )
        return digest.hexdigest()

    def unpack(self) -> dict[str, np.ndarray]:
        self.validate()
        p = np.asarray(self.parameters, dtype=np.float32)
        obs, hidden, action = (
            self.observation_dim,
            self.hidden_dim,
            self.action_dim,
        )
        cursor = 0

        def take(count: int, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal cursor
            value = p[cursor : cursor + count].reshape(shape)
            cursor += count
            return value

        tensors = {
            "input_weight": take(obs * hidden, (hidden, obs)),
            "input_bias": take(hidden, (hidden,)),
        }
        if self.architecture in {"gru", "lstm"}:
            gates = 3 if self.architecture == "gru" else 4
            name = self.architecture
            tensors[f"{name}_weight_ih"] = take(
                gates * hidden * hidden, (gates * hidden, hidden)
            )
            tensors[f"{name}_weight_hh"] = take(
                gates * hidden * hidden, (gates * hidden, hidden)
            )
            tensors[f"{name}_bias_ih"] = take(gates * hidden, (gates * hidden,))
            tensors[f"{name}_bias_hh"] = take(gates * hidden, (gates * hidden,))
        tensors.update({
            "hidden_weight": take(hidden * hidden, (hidden, hidden)),
            "hidden_bias": take(hidden, (hidden,)),
            "action_weight": take(hidden * action, (action, hidden)),
            "action_bias": take(action, (action,)),
        })
        if self.architecture == "tqc_actor":
            tensors.update({
                "log_std_weight": take(hidden * action, (action, hidden)),
                "log_std_bias": take(action, (action,)),
            })
        if cursor != p.size:
            raise RuntimeError("recurrent policy unpack cursor mismatch")
        return tensors

    def seeded_direction(self, seed: int) -> np.ndarray:
        self.validate()
        rng = np.random.default_rng(int(seed))
        signs = rng.integers(
            0, 2, size=self.parameters.shape, dtype=np.int8
        ).astype(np.float32)
        signs = signs * 2.0 - 1.0
        return signs / np.float32(
            math.sqrt(float(self.parameters.size))
        )

    def uniform_crossover(
        self,
        other: "RecurrentPolicyGenome",
        *,
        seed: int,
    ) -> "RecurrentPolicyGenome":
        """Uniform per-parameter 50/50 two-parent crossover; deterministic.

        Only neural Policy theta is recombined. Lineage-private psi is NOT
        mixed and must be constructed separately for the child.
        """
        self.validate()
        other.validate()
        if self.architecture_id != other.architecture_id:
            raise ValueError("crossover requires identical Policy architectures")
        rng = np.random.default_rng(int(seed))
        mask = rng.integers(
            0, 2, size=self.parameters.size, dtype=np.int8
        ).astype(bool)
        values = np.where(mask, self.parameters, other.parameters).astype(
            np.float32
        )
        result = RecurrentPolicyGenome(
            observation_dim=self.observation_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            parameters=values,
            architecture=self.architecture,
        )
        result.validate()
        return result

    def mutate(
        self,
        *,
        direction_seed: int,
        signed_scale: float,
    ) -> "RecurrentPolicyGenome":
        scale = float(signed_scale)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("signed_scale must be finite and non-zero")
        direction = self.seeded_direction(int(direction_seed))
        child = np.asarray(self.parameters, dtype=np.float32) + np.float32(
            scale
        ) * direction
        result = RecurrentPolicyGenome(
            observation_dim=self.observation_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            parameters=np.asarray(child, dtype=np.float32),
            architecture=self.architecture,
        )
        result.validate()
        if result.sha256 == self.sha256:
            raise RuntimeError("mutation produced unchanged policy")
        return result


def qweighted_probabilities(
    normalized_by_gene: Mapping[str, Sequence[float]],
    *,
    eligible_ids: Sequence[str] | None = None,
    axis_index: int | None = None,
    exponent: float = 2.0,
) -> dict[str, float]:
    """Return Q-weighted parent probabilities.

    axis_index=None uses total normalized quality:
        Q(g) = (sum_j s_j(g))^exponent

    axis_index=j preserves the optional historical axis-conditioned form:
        Q_j(g) = s_j(g)^exponent

    When exponent > 0, a Gene with zero selected quality has zero weight;
    if all weights vanish, use a uniform fallback.
    The annealed-exploration special case exponent == 0 is explicitly
    uniform over ALL eligible Genes, including zero-quality ones. This
    avoids ambiguous 0**0 semantics and applies per Gene, not per founder.
    """
    ids = tuple(
        sorted(
            normalized_by_gene
            if eligible_ids is None
            else (str(x) for x in eligible_ids)
        )
    )
    if not ids:
        raise ValueError("no eligible parents")
    selected_axis = None if axis_index is None else int(axis_index)
    power = float(exponent)
    if not np.isfinite(power) or power < 0.0:
        raise ValueError("exponent must be finite and nonnegative")
    weights: dict[str, float] = {}
    for gene_id in ids:
        values = np.asarray(
            normalized_by_gene[gene_id], dtype=np.float64
        )
        if values.ndim != 1 or values.size == 0:
            raise ValueError("normalized capability must be a vector")
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("normalized capability must lie in [0,1]")
        if selected_axis is None:
            quality = float(np.sum(values))
        else:
            if selected_axis < 0 or selected_axis >= int(values.size):
                raise ValueError("axis_index outside normalized capability")
            quality = float(values[selected_axis])
        weights[gene_id] = 1.0 if power == 0.0 else float(quality ** power)
    total = float(sum(weights.values()))
    if total <= 0.0:
        uniform = 1.0 / float(len(ids))
        return {gene_id: uniform for gene_id in ids}
    return {
        gene_id: float(weights[gene_id] / total)
        for gene_id in ids
    }


def sample_distinct_second_parent(
    probabilities: Mapping[str, float],
    *,
    primary_gene_id: str,
    seed: int,
) -> str | None:
    """Independently sample B from the frozen Q^2 distribution excluding A.

    If the excluded candidate takes all positive Q mass, only then use a
    uniform fallback over remaining Bank Genes. Return None for a 1-Gene Bank.
    """
    ids = tuple(sorted(
        gene_id for gene_id in probabilities
        if str(gene_id) != str(primary_gene_id)
    ))
    if not ids:
        return None
    weights = np.asarray(
        [float(probabilities[gene_id]) for gene_id in ids],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("invalid Q-weight second parent probabilities")
    total = float(weights.sum())
    weights = weights / total if total > 0.0 else np.full(
        len(ids), 1.0 / float(len(ids)), dtype=np.float64
    )
    return str(np.random.default_rng(int(seed)).choice(
        np.asarray(ids, dtype=object), p=weights
    ))


def sample_one_parent(
    probabilities: Mapping[str, float],
    *,
    seed: int,
) -> str:
    ids = tuple(sorted(probabilities))
    p = np.asarray(
        [float(probabilities[x]) for x in ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    return str(
        rng.choice(np.asarray(ids, dtype=object), p=p)
    )


__all__ = [
    "DirectionEvidence",
    "EvolutionProposal",
    "LineageEvolutionState",
    "RecurrentPolicyGenome",
    "propose_axis_mutation",
    "propose_global_mutation",
    "propose_random_mutation",
    "qweighted_probabilities",
    "sample_one_parent",
    "sample_distinct_second_parent",
]
