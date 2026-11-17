"""H2-K6 private relative-improvement memory; evolution only, never PPO.

The archive still judges the original four independent Hopper capability axes.
This isolated proposer learns only from ONE gene's inherited, same-reset paired
parent/child episodes. An early death does NOT erase positive relative evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .segb_full import (
    EvolutionProposal, LineageEvolutionState, propose_random_mutation,
)

LIFETIME_GAIN_FLOOR = 2  # control ticks; absolute, not % or full-survival
AXIS_GAIN_FLOOR = 1.0e-4  # original normalized capability-delta units
RANDOM_FRACTION = 0.20
HORIZON = 1000


@dataclass(frozen=True)
class RelativeEvidence:
    direction_seed: int
    signed_scale: float
    delta: tuple[float, float, float, float]  # ORIGINAL four benchmark axes
    parent_steps: int
    child_steps: int
    fully_survived: bool
    admitted: bool

    def __post_init__(self):
        if (not 1 <= self.parent_steps <= HORIZON or
            not 1 <= self.child_steps <= HORIZON):
            raise ValueError("Invalid actual paired parent/child rollout lengths")
        if self.direction_seed < 0 or not math.isfinite(self.signed_scale) or self.signed_scale == 0:
            raise ValueError("Invalid historical mutation key")
        if len(self.delta) != 4 or not all(math.isfinite(v) and -1 <= v <= 1 for v in self.delta):
            raise ValueError("Invalid signed four-axis normalized delta")

    @property
    def delta_life(self) -> float:
        return (self.child_steps-self.parent_steps)/float(HORIZON)

    @property
    def gain_axes(self) -> tuple[bool, bool, bool, bool, bool]:
        """Lifetime, Velocity, Jump, Energy, Healthy positive flags.

        Velocity/Jump/Healthy improvement is usable only if child doesn't die
        sooner. Energy requires at least same duration, to avoid short-life
        false gains. In all cases signed tradeoffs remain persisted.
        """
        stable = self.child_steps >= self.parent_steps
        return (
            self.child_steps >= self.parent_steps + LIFETIME_GAIN_FLOOR,
            stable and self.delta[0] > AXIS_GAIN_FLOOR,
            stable and self.delta[1] > AXIS_GAIN_FLOOR,
            stable and self.delta[2] > AXIS_GAIN_FLOOR,
            stable and self.delta[3] > AXIS_GAIN_FLOOR,
        )

    @property
    def useful(self) -> bool:
        return any(self.gain_axes)

    def to_dict(self) -> dict:
        return {
            "direction_seed":int(self.direction_seed),
            "signed_scale":float(self.signed_scale),
            "delta":list(self.delta),
            "parent_steps":int(self.parent_steps),
            "child_steps":int(self.child_steps),
            "delta_life":float(self.delta_life),
            "fully_survived":bool(self.fully_survived),
            "admitted":bool(self.admitted),
            "gain_axes":list(self.gain_axes),
            "useful":bool(self.useful),
        }


@dataclass
class RelativePrivateState:
    lineage_id: str
    owner_gene_id: str
    parent_gene_id: str | None = None
    evidence: list[RelativeEvidence] = field(default_factory=list)

    def validate(self) -> None:
        if not self.lineage_id or not self.owner_gene_id:
            raise ValueError("Relative memory requires an owner and lineage")
        seen = set()
        for e in self.evidence:
            key=(e.direction_seed,e.signed_scale)
            if key in seen:
                raise ValueError("Duplicate relative mutation evidence")
            seen.add(key)

    def observed(self, item: RelativeEvidence) -> "RelativePrivateState":
        self.validate()
        result=RelativePrivateState(
            lineage_id=self.lineage_id,owner_gene_id=self.owner_gene_id,
            parent_gene_id=self.parent_gene_id,evidence=[*self.evidence,item])
        result.validate()
        return result

    def inherited_by(self, child: str, parent: str) -> "RelativePrivateState":
        self.validate()
        if not child or parent != self.owner_gene_id:
            raise ValueError("Attempted cross-owner or cross-lineage private memory copy")
        result=RelativePrivateState(
            lineage_id=self.lineage_id,owner_gene_id=child,
            parent_gene_id=parent,evidence=list(self.evidence))
        result.validate()
        return result

    @property
    def positives(self) -> tuple[RelativeEvidence, ...]:
        return tuple(e for e in self.evidence if e.useful)

    def to_dict(self) -> dict:
        return {
            "schema":"h2k6_relative_private_psi_v1",
            "lineage_id":self.lineage_id,
            "owner_gene_id":self.owner_gene_id,
            "parent_gene_id":self.parent_gene_id,
            "positive_count":len(self.positives),
            "negative_or_tradeoff_count":sum(not e.useful for e in self.evidence),
            "evidence":[e.to_dict() for e in self.evidence],
        }


def propose_relative_mutation(
    *,
    mode: str,
    state: LineageEvolutionState,
    memory: RelativePrivateState,
    scheduler_seed: int,
) -> EvolutionProposal:
    """One mutation from an inherited private relative-evidence landscape.

    Until H2 has at least one relative-positive record, R and H2 produce
    IDENTICAL random proposals for identical parent thetas/seeds/states.
    Signed negative evidence is kept; repeated poor outcomes for a direction
    family downweight, never silently purge, that direction's later sampling.
    """
    if mode not in ("R","H2"):
        raise ValueError("Unknown H2-K6 arm")
    state.validate(4)
    memory.validate()
    if (memory.lineage_id!=state.lineage_id or
        memory.owner_gene_id!=state.owner_gene_id):
        raise ValueError("Private psi lineage/owner isolation violated")

    positive=memory.positives if mode=="H2" else ()
    if not positive:
        p=propose_random_mutation(state,scheduler_seed=scheduler_seed,
                                  objective_count=4)
        return EvolutionProposal(
            source="matched_random_before_relative_guidance" if mode=="H2"
                   else "random_baseline",
            direction_seed=p.direction_seed,signed_scale=p.signed_scale)

    rng=np.random.default_rng(int(scheduler_seed))
    if float(rng.random()) < RANDOM_FRACTION:
        p=propose_random_mutation(
            state,scheduler_seed=int(rng.integers(1,2**31-1)),
            objective_count=4)
        return EvolutionProposal(source="relative_persistent_random",
                                 direction_seed=p.direction_seed,
                                 signed_scale=p.signed_scale)

    # One randomly chosen capability target among axes with actual positive
    # private evidence; no scalar reward and no fixed preference weighting.
    active_axes=[i for i in range(5)
                 if any(e.gain_axes[i] for e in positive)]
    axis=active_axes[int(rng.integers(0,len(active_axes)))]
    eligible=[e for e in positive if e.gain_axes[axis]]
    # Direction-family setbacks on the SELECTED axis discourage retries.
    # A record helping lifespan but harming jump is negative jump evidence.
    # Nothing is erased and no other Gene/lineage's history is consulted.
    bad={}
    for row in memory.evidence:
        if not row.gain_axes[axis]:
            k=(row.direction_seed,math.copysign(1.0,row.signed_scale))
            bad[k]=bad.get(k,0)+1
    weights=np.asarray([
        1.0/(1.0+bad.get((e.direction_seed,math.copysign(1.0,e.signed_scale)),0))
        for e in eligible],dtype=np.float64)
    weights/=weights.sum()
    basis=eligible[int(rng.choice(len(eligible),p=weights))]
    used={(e.direction_seed,float(e.signed_scale)) for e in state.evidence}
    for _ in range(256):
        factor=math.exp(float(rng.uniform(
            math.log(state.guided_jitter_min),
            math.log(state.guided_jitter_max))))
        magnitude=max(state.random_sigma_min,
                      min(state.random_sigma_max,abs(basis.signed_scale)*factor))
        signed=math.copysign(magnitude,basis.signed_scale)
        if (basis.direction_seed,signed) not in used:
            return EvolutionProposal(
                source="relative_guided",
                direction_seed=basis.direction_seed,
                signed_scale=signed,
                basis_direction_seed=basis.direction_seed,
                basis_signed_scale=basis.signed_scale)
    raise RuntimeError("Cannot generate unused relative-guided mutation")



def from_direction_evidence_state(state: LineageEvolutionState) -> RelativePrivateState:
    """Read H2-K6 B50M private psi from durable native DirectionEvidence.

    Unlike the mature-founder short-run adapter, B50M journals persist only
    active Gene psi. Paired step lengths live in each DirectionEvidence so
    uninterrupted and resumed 50M runs use exactly the same local knowledge.
    This conversion is read-only and never uses another Gene's evidence.
    """
    state.validate(4)
    observations = []
    for row in state.evidence:
        if row.parent_steps is None or row.child_steps is None:
            raise ValueError(
                "H2-K6 50M requires paired physical step data in every "
                "inherited DirectionEvidence; cannot silently use proxy history"
            )
        observations.append(RelativeEvidence(
            direction_seed=int(row.direction_seed),
            signed_scale=float(row.signed_scale),
            delta=tuple(float(x) for x in row.delta),
            parent_steps=int(row.parent_steps),
            child_steps=int(row.child_steps),
            fully_survived=bool(row.survived),
            admitted=bool(row.admitted),
        ))
    memory = RelativePrivateState(
        lineage_id=state.lineage_id,
        owner_gene_id=state.owner_gene_id,
        parent_gene_id=state.parent_gene_id,
        evidence=observations,
    )
    memory.validate()
    return memory


def propose_from_durable_relative_psi(
    mode: str, state: LineageEvolutionState, *, scheduler_seed: int
) -> EvolutionProposal:
    """Opt-in from-scratch 50M H2/Random with exact K6 evidence thresholds."""
    if mode not in ("h2", "random"):
        raise ValueError("Unknown H2-K6 B50M mode")
    memory = from_direction_evidence_state(state)
    return propose_relative_mutation(
        mode="H2" if mode == "h2" else "R",
        state=state,
        memory=memory,
        scheduler_seed=int(scheduler_seed),
    )
