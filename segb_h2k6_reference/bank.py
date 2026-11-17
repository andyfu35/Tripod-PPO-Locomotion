from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneRecord:
    gene_id: str
    genome_sha256: str
    capability: tuple[float, ...]
    parent_gene_id: str | None = None
    lineage_id: str | None = None
    generation: int = 0

    def validate(self, objective_count: int | None = None) -> None:
        if not self.gene_id:
            raise ValueError("gene_id must be non-empty")
        if not self.genome_sha256:
            raise ValueError("genome_sha256 must be non-empty")
        values = np.asarray(self.capability, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("capability must be a non-empty 1-D vector")
        if not np.all(np.isfinite(values)):
            raise ValueError("capability values must be finite")
        if objective_count is not None and values.size != int(objective_count):
            raise ValueError("capability dimension mismatch")
        if int(self.generation) < 0:
            raise ValueError("generation must be non-negative")


@dataclass(frozen=True)
class ParetoUpdate:
    admitted: bool
    removed_gene_ids: tuple[str, ...]
    active_gene_ids: tuple[str, ...]


@dataclass(frozen=True)
class ParetoBatchUpdate:
    admitted_gene_ids: tuple[str, ...]
    rejected_gene_ids: tuple[str, ...]
    removed_gene_ids: tuple[str, ...]
    active_gene_ids: tuple[str, ...]


def pareto_dominates(
    lhs: Iterable[float],
    rhs: Iterable[float],
    *,
    atol: float = 1.0e-12,
) -> bool:
    a = np.asarray(tuple(lhs), dtype=np.float64)
    b = np.asarray(tuple(rhs), dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("Pareto vectors must have equal non-empty 1-D shape")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Pareto vectors must be finite")
    tolerance = float(atol)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("atol must be finite and non-negative")
    no_worse = np.all(a >= b - tolerance)
    strictly_better = np.any(a > b + tolerance)
    return bool(no_worse and strictly_better)


def nondominated_ids(
    records: dict[str, GeneRecord],
    *,
    atol: float = 1.0e-12,
    block_size: int = 256,
) -> tuple[str, ...]:
    """Return the exact Pareto front using block-vectorized comparisons.

    Semantics match :func:`pareto_dominates`, but capability vectors are
    materialized once and dominance checks run in NumPy blocks rather than
    Python pairwise loops. This is important for Grid publication where many
    children can occupy the same cell in one frozen generation.
    """
    ids = tuple(sorted(records))
    if not ids:
        return ()

    matrix = np.asarray(
        [records[gene_id].capability for gene_id in ids],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[1] <= 0:
        raise ValueError("Pareto capability matrix must be non-empty 2-D")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Pareto capability vectors must be finite")

    tolerance = float(atol)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("atol must be finite and non-negative")
    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")

    dominated = np.zeros(len(ids), dtype=bool)
    for start in range(0, len(ids), block):
        stop = min(start + block, len(ids))
        targets = matrix[start:stop]
        no_worse = np.all(
            matrix[:, None, :] >= targets[None, :, :] - tolerance,
            axis=2,
        )
        strictly_better = np.any(
            matrix[:, None, :] > targets[None, :, :] + tolerance,
            axis=2,
        )
        dominated[start:stop] = np.any(
            no_worse & strictly_better,
            axis=0,
        )

    return tuple(
        gene_id
        for index, gene_id in enumerate(ids)
        if not bool(dominated[index])
    )


def _crowding_distance(records: list[GeneRecord]) -> dict[str, float]:
    if not records:
        return {}
    objective_count = len(records[0].capability)
    for record in records:
        record.validate(objective_count)
    distance = {record.gene_id: 0.0 for record in records}
    if len(records) <= 2:
        return {record.gene_id: float("inf") for record in records}

    for objective in range(objective_count):
        ordered = sorted(
            records,
            key=lambda item: (float(item.capability[objective]), item.gene_id),
        )
        low = float(ordered[0].capability[objective])
        high = float(ordered[-1].capability[objective])
        distance[ordered[0].gene_id] = float("inf")
        distance[ordered[-1].gene_id] = float("inf")
        span = high - low
        if span <= 1.0e-15:
            continue
        for index in range(1, len(ordered) - 1):
            gene_id = ordered[index].gene_id
            if np.isinf(distance[gene_id]):
                continue
            previous_value = float(ordered[index - 1].capability[objective])
            next_value = float(ordered[index + 1].capability[objective])
            distance[gene_id] += (next_value - previous_value) / span
    return distance


def _rank_percentile_scores(
    records: dict[str, GeneRecord],
) -> dict[str, tuple[float, ...]]:
    """Return per-axis empirical mid-ranks in [0,1].

    The score uses only ordering inside the current nondominated front, so no
    capability-specific scale, threshold, exponent, or calibration range is
    introduced. The worst unique value maps to 0, the best unique value maps
    to 1, and exact ties share their average rank. A single-Gene front maps to
    1 on every axis.
    """
    ids = tuple(sorted(records))
    if not ids:
        return {}
    matrix = np.asarray(
        [records[gene_id].capability for gene_id in ids],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("retention capability matrix must be finite 2-D")
    count = int(matrix.shape[0])
    if count == 1:
        return {
            ids[0]: tuple(1.0 for _ in range(int(matrix.shape[1])))
        }

    scores = np.empty_like(matrix, dtype=np.float64)
    denominator = float(count - 1)
    for axis in range(matrix.shape[1]):
        ordered = np.sort(matrix[:, axis])
        values = matrix[:, axis]
        left = np.searchsorted(ordered, values, side="left")
        right = np.searchsorted(ordered, values, side="right")
        average_zero_based_rank = 0.5 * (
            left.astype(np.float64)
            + right.astype(np.float64)
            - 1.0
        )
        scores[:, axis] = average_zero_based_rank / denominator

    return {
        gene_id: tuple(float(x) for x in scores[index].tolist())
        for index, gene_id in enumerate(ids)
    }


def _truncate_by_crowding(
    records: dict[str, GeneRecord], *, capacity: int | None
) -> tuple[str, ...]:
    if capacity is None:
        return tuple(sorted(records))
    if len(records) <= capacity:
        return tuple(sorted(records))
    distance = _crowding_distance(list(records.values()))
    ranked = sorted(
        records,
        key=lambda gene_id: (-float(distance[gene_id]), gene_id),
    )
    return tuple(sorted(ranked[:capacity]))


def _truncate_by_linear_rank_proportional(
    records: dict[str, GeneRecord],
    *,
    capacity: int | None,
    seed: int | None,
) -> tuple[str, ...]:
    """Bound an oversized archive without nonlinear capability shaping.

    Protect one champion per axis and the Gene with the highest product of
    per-axis percentile ranks. Fill remaining slots by weighted sampling
    without replacement with weight equal to the linear sum of percentile
    ranks. Zero-weight candidates are allowed; if every remaining candidate
    has zero weight, the fallback sampling distribution is uniform.
    """
    if capacity is None:
        return tuple(sorted(records))
    if len(records) <= capacity:
        return tuple(sorted(records))
    if seed is None:
        raise ValueError(
            "linear_rank_proportional retention requires a deterministic seed"
        )

    ids = tuple(sorted(records))
    objective_count = len(records[ids[0]].capability)
    if int(capacity) < objective_count + 1:
        raise ValueError(
            "capacity is too small to protect per-axis champions plus HV-best"
        )

    scores = _rank_percentile_scores(records)
    protected: set[str] = set()
    for axis in range(objective_count):
        champion = min(
            ids,
            key=lambda gene_id: (
                -float(records[gene_id].capability[axis]),
                gene_id,
            ),
        )
        protected.add(champion)

    hv_best = min(
        ids,
        key=lambda gene_id: (
            -float(np.prod(np.asarray(scores[gene_id], dtype=np.float64))),
            gene_id,
        ),
    )
    protected.add(hv_best)

    remaining_slots = int(capacity) - len(protected)
    remaining_ids = tuple(
        gene_id for gene_id in ids if gene_id not in protected
    )
    if remaining_slots <= 0:
        return tuple(sorted(protected))
    if len(remaining_ids) <= remaining_slots:
        return tuple(sorted(protected | set(remaining_ids)))

    weights = np.asarray(
        [
            float(np.sum(np.asarray(scores[gene_id], dtype=np.float64)))
            for gene_id in remaining_ids
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise RuntimeError(
            "linear retention weights must be finite and non-negative"
        )
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        probabilities = np.full(
            len(remaining_ids),
            1.0 / float(len(remaining_ids)),
            dtype=np.float64,
        )
    else:
        probabilities = weights / weight_sum
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(
        np.asarray(remaining_ids, dtype=object),
        size=remaining_slots,
        replace=False,
        p=probabilities,
    )
    return tuple(sorted(protected | {str(x) for x in chosen.tolist()}))




def _adaptive_grid_cells(
    records: dict[str, GeneRecord], *, depth: int
) -> dict[str, tuple[int, ...]]:
    """Place full-precision objective vectors in PAES-style adaptive cells.

    Each objective range is recomputed from this publication's nondominated
    candidates, not fixed physical score widths.  Sparse integer coordinates
    avoid a dense (2**depth)**objective_count allocation.
    """
    ids = tuple(sorted(records))
    if not ids:
        return {}
    matrix = np.asarray(
        [records[gene_id].capability for gene_id in ids], dtype=np.float64
    )
    divisions = 1 << int(depth)
    low = np.min(matrix, axis=0)
    span = np.max(matrix, axis=0) - low
    scaled = np.zeros_like(matrix)
    varying = span > 1.0e-12
    if np.any(varying):
        scaled[:, varying] = (
            (matrix[:, varying] - low[varying]) / span[varying]
        )
    coordinates = np.minimum(
        np.floor(scaled * float(divisions)).astype(np.int64),
        divisions - 1,
    )
    return {
        gene_id: tuple(int(v) for v in coordinates[i])
        for i, gene_id in enumerate(ids)
    }


def _truncate_by_paes_adaptive_density(
    records: dict[str, GeneRecord],
    *,
    capacity: int,
    depth: int,
    seed: int,
) -> tuple[tuple[str, ...], int, int, int]:
    """Bound a nondominated batch using adaptive-grid crowding.

    This is the PAES *archive* density principle, adapted to SEGB's atomic
    many-offspring generation.  It is NOT a faithful (1+1)-PAES optimizer:
    when a full batch exceeds capacity, cells are computed once and densest
    cells are thinned with seeded, deterministic tie-breaking. Q-based
    parent choice and lineage-local mutation remain SEGB mechanisms.
    """
    cells = _adaptive_grid_cells(records, depth=depth)
    by_cell: dict[tuple[int, ...], list[str]] = {}
    for gene_id, cell in cells.items():
        by_cell.setdefault(cell, []).append(gene_id)
    for occupants in by_cell.values():
        occupants.sort()
    original_cell_count = len(by_cell)
    max_occupancy = max(map(len, by_cell.values()), default=0)
    overflow = max(0, len(records) - int(capacity))
    if overflow == 0:
        return tuple(sorted(records)), original_cell_count, max_occupancy, 0
    rng = np.random.default_rng(int(seed))
    for _ in range(overflow):
        highest_density = max(len(v) for v in by_cell.values())
        densest = sorted(
            cell for cell, occupants in by_cell.items()
            if len(occupants) == highest_density
        )
        chosen_cell = densest[int(rng.integers(len(densest)))]
        occupants = by_cell[chosen_cell]
        # Prefer retaining sparse regions; within equal-density cells,
        # randomize reproducibly rather than introducing an extra quality
        # scalar or a lineage priority.
        victim_index = int(rng.integers(len(occupants)))
        occupants.pop(victim_index)
        if not occupants:
            del by_cell[chosen_cell]
    kept = tuple(sorted(
        gene_id
        for occupants in by_cell.values()
        for gene_id in occupants
    ))
    if len(kept) != capacity:
        raise RuntimeError("PAES-style adaptive archive capacity invariant failed")
    return kept, original_cell_count, max_occupancy, overflow


class ParetoBank:
    """Archive with optional capability-grid resolution and bounded retention.

    Without capability_resolution, historical Pareto semantics are preserved:
    Pareto dominance is applied first, then an optional capacity truncates the
    nondominated front.

    With capability_resolution, the grid itself defines archive niches:
    exactly one representative is kept per occupied capability cell and
    different cells do NOT Pareto-dominate one another. An optional
    capability_origin shifts the fixed grid without changing stored raw
    capabilities. Capacity is only a safety fallback when the number of
    occupied cells exceeds the limit.
    """

    def __init__(
        self,
        *,
        objective_count: int,
        capacity: int | None = None,
        retention_mode: str = "crowding",
        adaptive_grid_depth: int = 5,
        capability_resolution: tuple[float, ...] | None = None,
        capability_origin: tuple[float, ...] | None = None,
    ) -> None:
        self.objective_count = int(objective_count)
        self.capacity = None if capacity is None else int(capacity)
        self.retention_mode = str(retention_mode)
        self.adaptive_grid_depth = int(adaptive_grid_depth)
        self.capability_resolution = (
            None
            if capability_resolution is None
            else tuple(float(x) for x in capability_resolution)
        )
        self.capability_origin = (
            None
            if capability_origin is None
            else tuple(float(x) for x in capability_origin)
        )
        if self.objective_count <= 0:
            raise ValueError("objective_count must be positive")
        if self.capacity is not None and self.capacity <= 0:
            raise ValueError("capacity must be positive when specified")
        if self.retention_mode not in {
            "crowding",
            "linear_rank_proportional",
            "paes_adaptive_grid",
        }:
            raise ValueError(
                "retention_mode must be crowding, linear_rank_proportional or paes_adaptive_grid"
            )
        if not 1 <= self.adaptive_grid_depth <= 12:
            raise ValueError("adaptive_grid_depth must be within [1,12]")
        if self.retention_mode == "paes_adaptive_grid":
            if self.capacity is None or self.capability_resolution is not None:
                raise ValueError("PAES adaptive archive requires capacity and no fixed capability grid")
        if self.capability_resolution is not None:
            widths = np.asarray(
                self.capability_resolution,
                dtype=np.float64,
            )
            if widths.shape != (self.objective_count,):
                raise ValueError("capability_resolution dimension mismatch")
            if (
                not np.all(np.isfinite(widths))
                or np.any(widths <= 0.0)
            ):
                raise ValueError(
                    "capability_resolution must be finite and positive"
                )
            origins = (
                np.zeros(self.objective_count, dtype=np.float64)
                if self.capability_origin is None
                else np.asarray(self.capability_origin, dtype=np.float64)
            )
            if origins.shape != (self.objective_count,):
                raise ValueError("capability_origin dimension mismatch")
            if not np.all(np.isfinite(origins)):
                raise ValueError("capability_origin must be finite")
        elif self.capability_origin is not None:
            raise ValueError(
                "capability_origin requires capability_resolution"
            )
        self._records: dict[str, GeneRecord] = {}
        self.last_combined_record_count = 0
        self.last_unique_cell_count = 0
        self.last_pareto_front_size = 0
        self.last_archive_candidate_count = 0
        self.last_capacity_truncation_used = False
        self.last_max_cell_occupancy = 0
        self.last_adaptive_cell_count = 0
        self.last_paes_density_evictions = 0

    @property
    def records(self) -> tuple[GeneRecord, ...]:
        return tuple(
            self._records[gene_id] for gene_id in sorted(self._records)
        )

    @property
    def gene_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def get(self, gene_id: str) -> GeneRecord:
        return self._records[str(gene_id)]

    def capability_cell(
        self,
        capability: Iterable[float],
    ) -> tuple[int, ...] | None:
        if self.capability_resolution is None:
            return None
        values = np.asarray(tuple(capability), dtype=np.float64)
        widths = np.asarray(self.capability_resolution, dtype=np.float64)
        origins = (
            np.zeros(self.objective_count, dtype=np.float64)
            if self.capability_origin is None
            else np.asarray(self.capability_origin, dtype=np.float64)
        )
        if values.shape != widths.shape:
            raise ValueError("capability dimension mismatch")
        if not np.all(np.isfinite(values)):
            raise ValueError("capability values must be finite")
        return tuple(
            int(x)
            for x in np.floor(
                (values - origins) / widths
            ).astype(np.int64).tolist()
        )

    def _selection_records(
        self,
        combined: dict[str, GeneRecord],
        *,
        incumbent_ids: set[str],
    ) -> tuple[dict[str, GeneRecord], set[str]]:
        if self.capability_resolution is None:
            return dict(combined), set()

        by_cell: dict[tuple[int, ...], list[str]] = {}
        for gene_id in sorted(combined):
            cell = self.capability_cell(combined[gene_id].capability)
            assert cell is not None
            by_cell.setdefault(cell, []).append(gene_id)

        self.last_max_cell_occupancy = max(
            (len(ids) for ids in by_cell.values()),
            default=0,
        )

        selected: dict[str, GeneRecord] = {}
        duplicate_ids: set[str] = set()
        for cell, ids in by_cell.items():
            raw_cell_records = {
                gene_id: combined[gene_id]
                for gene_id in ids
            }
            raw_front = set(nondominated_ids(raw_cell_records))
            incumbent_front = sorted(
                gene_id
                for gene_id in raw_front
                if gene_id in incumbent_ids
            )
            representative = (
                incumbent_front[0]
                if incumbent_front
                else sorted(raw_front)[0]
            )
            source = combined[representative]
            selected[representative] = GeneRecord(
                gene_id=source.gene_id,
                genome_sha256=source.genome_sha256,
                capability=tuple(float(x) for x in cell),
                parent_gene_id=source.parent_gene_id,
                lineage_id=source.lineage_id,
                generation=source.generation,
            )
            duplicate_ids.update(
                gene_id for gene_id in ids if gene_id != representative
            )
        return selected, duplicate_ids

    def commit_batch(
        self,
        candidates: Iterable[GeneRecord],
        *,
        retention_seed: int | None = None,
    ) -> ParetoBatchUpdate:
        batch = tuple(candidates)
        if not batch:
            raise ValueError("candidate batch must not be empty")
        candidate_ids = [candidate.gene_id for candidate in batch]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate batch contains duplicate gene_id")
        overlap = set(candidate_ids) & set(self._records)
        if overlap:
            raise ValueError(
                f"candidate gene_id already active: {sorted(overlap)[0]}"
            )
        for candidate in batch:
            candidate.validate(self.objective_count)

        before = set(self._records)
        combined = dict(self._records)
        combined.update({candidate.gene_id: candidate for candidate in batch})
        selection_records, duplicate_ids = self._selection_records(
            combined,
            incumbent_ids=before,
        )

        if self.capability_resolution is None:
            front_ids = nondominated_ids(selection_records)
            if self.retention_mode == "paes_adaptive_grid":
                # Collapse exactly equal raw capability vectors while preferring
                # the old archive representative; equal vectors do not express
                # a new capability breakthrough.
                seen_vectors: set[tuple[float, ...]] = set()
                unique_front_ids: list[str] = []
                for gene_id in sorted(front_ids, key=lambda x: (x not in before, x)):
                    raw_vector = tuple(selection_records[gene_id].capability)
                    if raw_vector in seen_vectors:
                        continue
                    seen_vectors.add(raw_vector)
                    unique_front_ids.append(gene_id)
                front_ids = tuple(unique_front_ids)
            archive_candidates = {
                gene_id: selection_records[gene_id]
                for gene_id in front_ids
            }
            self.last_pareto_front_size = len(archive_candidates)
        else:
            # Grid mode: every occupied capability cell is a valid niche.
            # Do NOT apply Pareto dominance across different cells.
            archive_candidates = dict(selection_records)
            self.last_pareto_front_size = 0

        capacity_truncation_used = (
            self.capacity is not None
            and len(archive_candidates) > int(self.capacity)
        )
        self.last_adaptive_cell_count = 0
        self.last_paes_density_evictions = 0
        if self.retention_mode == "paes_adaptive_grid":
            kept_ids, n_cells, max_occupancy, n_evicted = (
                _truncate_by_paes_adaptive_density(
                    archive_candidates,
                    capacity=int(self.capacity),
                    depth=self.adaptive_grid_depth,
                    seed=0 if retention_seed is None else int(retention_seed),
                )
            )
            self.last_adaptive_cell_count = n_cells
            self.last_max_cell_occupancy = max_occupancy
            self.last_paes_density_evictions = n_evicted
        elif self.retention_mode == "linear_rank_proportional":
            kept_ids = _truncate_by_linear_rank_proportional(
                archive_candidates,
                capacity=self.capacity,
                seed=retention_seed,
            )
        else:
            kept_ids = _truncate_by_crowding(
                archive_candidates,
                capacity=self.capacity,
            )
        self._records = {
            gene_id: combined[gene_id]
            for gene_id in kept_ids
        }

        self.last_combined_record_count = len(combined)
        self.last_unique_cell_count = len(selection_records)
        self.last_archive_candidate_count = len(archive_candidates)
        self.last_capacity_truncation_used = bool(
            capacity_truncation_used
        )

        after = set(self._records)
        candidate_set = set(candidate_ids)
        return ParetoBatchUpdate(
            admitted_gene_ids=tuple(sorted(candidate_set & after)),
            rejected_gene_ids=tuple(
                sorted((candidate_set - after) | (candidate_set & duplicate_ids))
            ),
            removed_gene_ids=tuple(sorted(before - after)),
            active_gene_ids=tuple(sorted(after)),
        )

    def update(
        self,
        candidate: GeneRecord,
        *,
        retention_seed: int | None = None,
    ) -> ParetoUpdate:
        result = self.commit_batch(
            (candidate,),
            retention_seed=retention_seed,
        )
        return ParetoUpdate(
            admitted=candidate.gene_id in result.admitted_gene_ids,
            removed_gene_ids=result.removed_gene_ids,
            active_gene_ids=result.active_gene_ids,
        )


__all__ = [
    "GeneRecord",
    "ParetoBank",
    "ParetoBatchUpdate",
    "ParetoUpdate",
    "_rank_percentile_scores",
    "nondominated_ids",
    "pareto_dominates",
]
