"""SEGB tripod V2: score uninterrupted lives and use local, structured mutations.

Same robot, inputs, action range, environment reward, and global Q2/Pareto archive.
Changes concern *search efficiency*, not a hand-coded gait or reward shaping.
"""
from __future__ import annotations

import math
import numpy as np

from segb_core import (
    ACT_DIM, GENOME_LENGTH, HIDDEN_DIM, Gene, ParetoGeneBank, q2_probabilities,
)

VERSION = "segb-tripod-single-life-sparse-v2"


def local_mutation_direction(seed: int) -> np.ndarray:
    """Reproducible mutation in a small parameter subset.

    Reusing a successful direction seed reproduces its exact affected weights.
    The operator never embeds a gait trajectory or ground-contact sequence.
    """
    rng = np.random.default_rng(int(seed))
    op = float(rng.random())
    direction = np.zeros(GENOME_LENGTH, dtype=np.float32)
    if op < 0.40:             # alter only the 12 joint-output biases
        direction[-ACT_DIM:] = rng.standard_normal(ACT_DIM).astype(np.float32)
    elif op < 0.70:           # jointly adjust the final output layer
        n = (HIDDEN_DIM + 1) * ACT_DIM
        direction[-n:] = rng.standard_normal(n).astype(np.float32)
    else:                    # explore about 8% of all parameters
        k = max(1, round(GENOME_LENGTH * 0.08))
        idx = rng.choice(GENOME_LENGTH, k, replace=False)
        direction[idx] = rng.standard_normal(k).astype(np.float32)
    return direction


def new_gene(index: int, bank: ParetoGeneBank, rng: np.random.Generator,
             de_novo_fraction: float = 0.20) -> Gene:
    if not 0 <= float(de_novo_fraction) <= 1:
        raise ValueError("invalid de_novo_fraction")
    if not bank.genes or rng.random() < de_novo_fraction:
        # A modest random scale gives near-neutral *initial* joint commands;
        # all 12 joint targets remain free for discovery by evolution.
        weights = rng.normal(0.0, 0.05, GENOME_LENGTH).astype(np.float32)
        return Gene(gene_id=index, parameters=weights, lineage_id=index)

    pool = bank.ordered()
    parent = pool[int(rng.choice(len(pool), p=q2_probabilities(pool)))]
    if parent.guide_seed is not None and rng.random() < 0.50:
        mutation_seed = int(parent.guide_seed)
        sigma = float(parent.guide_sigma) * float(rng.uniform(0.5, 1.5))
        sigma = float(np.clip(sigma, 0.002, 0.05))
    else:
        mutation_seed = int(rng.integers(1, 2**31 - 1))
        sigma = float(math.exp(rng.uniform(math.log(0.003), math.log(0.04))))
    delta = local_mutation_direction(mutation_seed) * sigma
    parameters = np.clip(parent.parameters + delta, -3.0, 3.0).astype(np.float32)
    return Gene(gene_id=index, parameters=parameters,
                parent_id=parent.gene_id, lineage_id=parent.lineage_id,
                generation=parent.generation + 1,
                mutation_seed=mutation_seed, mutation_sigma=sigma,
                guide_seed=parent.guide_seed, guide_sigma=parent.guide_sigma)


def score_single_life(gene: Gene, *, total_progress: float, total_reward: float,
                      steps: int, horizon: int, fell: bool) -> Gene:
    """An early fall receives no reset-derived second chance on the *same gene*.

    Normalize ALL capabilities by the full evaluation horizon (not by life length).
    This avoids an immediate high-speed fall having a high mean-reward-per-alive-step.
    It does not add or alter any reward inside TripodEnv.
    """
    if horizon < 1 or not 1 <= steps <= horizon:
        raise ValueError("invalid single-life horizon")
    gene.progress_mps = float(total_progress / horizon)
    gene.mean_reward = float(total_reward / horizon)
    gene.longevity = float(steps / horizon)
    gene.fall_count = int(fell)
    return gene
