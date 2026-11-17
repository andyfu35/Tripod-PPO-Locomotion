# SEGB H2-K6 for the 12-DOF tripod: faithful history-mechanism port

**Experiment branch:** `experiment/segb-tripod-h2k6-private-history-20261117`.

This port restores *per-gene private lineage history* (`psi`), including **all positive, negative, and trade-off directions**, **matched-reset physical parent/child evaluations**, a *signed* direction family, relative improvement by capability, physical lifetime comparison, parent-history updates even for **rejected** children, and full history inheritance by accepted offspring. This replaces the previous single `guide_seed` approximation; it does **not** change the PPO baseline, robot, physics, 51-dimensional observations, 12 servos, motor command filter, or the original speed-plus-alive reward.

## Precise origin / differences

Reference code from the original research repo `andyfu35/Self-Evolving-Gene-Bank` at branch `experiment/segb-paes-adaptive-archive-20261017`:

- `benchmarks/segb_multiobjective_benchmark/segb_benchmark/segb_full.py`: `DirectionEvidence`, `LineageEvolutionState`, `RecurrentPolicyGenome`, signed Rademacher `seeded_direction`, Q-squared parent weights, `propose_global_mutation`.
- `benchmarks/segb_multiobjective_benchmark/scripts/h2k6_relative_memory.py`: H2 matched parent/child private-relative history, per-axis positive checks, signed negative-family downweighting, non-duplicated guided proposals, 20% direction random exploration.
- `benchmarks/segb_multiobjective_benchmark/segb_benchmark/segb_full_torch.py`: GRUCell `r,z,n` gate order and `silu` nonlinearities.
- `benchmarks/segb_multiobjective_benchmark/segb_benchmark/bank.py`: Pareto archive with PAES adaptive density at cap=1024, depth=5.

The tripod version is a **two-axis CPU adaptation**, not a claim of running the unchanged Hopper GRU/CUDA/4-axis pipeline. It reproduces the behavior of these components in self-contained `h2k6_tripod_core.py`, with no dependency on the older repository. The chosen GRU has hidden size **26**, is initialized with Normal(0,0.05), and uses the original signed `±1/sqrt(parameters)` perturbation with log-uniform magnitude `[0.01,30]`.

## Same research environment and budget

| Item | PPO Experiment C | SEGB H2-K6 tripod |
|---|---:|---:|
| MuJoCo XML and robot | same | same |
| 51 observations / 12 position targets | same | same |
| Reward | signed outward distance speed + 0.2 alive | **unchanged** |
| Physical control interval | 0.02 s | 0.02 s |
| CPU MuJoCo worlds | 16 | 16 (8 paired workers) |
| Requested control interactions | 5,000,000 | **exactly 5,000,000** |
| Real PPO control steps | 5,013,504 due rollout rounding | 5,000,000 |
| Policy | MLP 256×256, gradient-trained | GRU26, independent per-gene psi |
| Optimization | PPO | H2-K6 private relative history + SEGB |

**A control interaction = one `TripodEnv.step()`**, NOT a single low-level MuJoCo integration step. Both parent and child evaluations consume this budget, and early-fall evaluations terminate without repeated restarts or padded no-op physics. A final incomplete trial may be *censored* and excluded from archive updates; its control steps are still charged. Reset settling steps are excluded consistently with Gym/PPO episodes.

Training uses **eight parallel matched parent/child pairs** (16 simultaneous independent MuJoCo environments). Two worlds in a pair are reset using the identical seed. The paired runs terminate at their own fall or the 1,000-step horizon.

## The real H2-K6 private memory

For each Gene `i`, the private history is:

`Psi_i = [(direction_seed, signed_scale, delta_progress, delta_life, parent_steps, child_steps, survived, admitted, matched_seed), ...]`

All proposed mutations, **including non-admitted or harmful children**, are recorded in their selected parent's private history when both matched evaluations complete. An accepted child inherits an independent copy of the parent's updated history. Failed directions are retained to reduce repetition, not erased. No globally trained mutation model or hidden sharing between different lineages.

Fitness capabilities for archive selection (not an extra Gym reward) are:

```
F_progress = final_net_distance_m / (1000 * 0.02)  # m/s over full horizon
F_life     = first_uninterrupted_life_steps / 1000
```

The relative evidence vector is the clipped signed difference of the **same-seed** child and parent capabilities. A movement mutation is considered relatively useful only when the child did **not** die sooner and its progress improved. Lifetime gains require **at least 2 control steps**. Both beneficial and detrimental outcomes stay in `Psi_i`.

When private positive evidence exists, H2 samples a beneficial capability axis and a direction family. The score of a direction family is penalized by its axis-specific historical setbacks `1/(1+N_bad)`. A signed direction is reused with log-uniform multiplicative jitter `[0.5,2]` without repeating an identical historical `(seed, signed_scale)` pair. With probability 0.20 it still explores a brand-new direction. **20% of all offspring are independent de-novo genes** and 80% are inherited/mutated. Parent genes are sampled from a frozen Pareto bank with empirical per-axis midranks and `Q=(sum_j score_j)^2`. PAES archive cap 1024, grid depth 5.

The SEGB GRU network uses genuine internal dynamic state reset at each episode, which makes sustained movement *possible* even under static sensor inputs. It is **not guaranteed** to discover a walking gait; real simulator evidence is needed.

## Mac execution

```bash
cd ~/PycharmProjects/Tripod-PPO-Locomotion
git fetch origin
git switch --track origin/experiment/segb-tripod-h2k6-private-history-20261117
source .venv/bin/activate
bash run_segb_h2k6_5m.sh
```

A separate checkpoint directory `runs_segb_h2k6_5m_seed0/` is used. If interrupted, run the script again: a matching checkpoint resumes automatically. Checkpoint contents include the 8k-ish GRU parameter vectors, **all active genes' private histories**, RNG state, budget accounting, lineage IDs and the best reward candidate. Checkpoints are written atomically.

After training:

```bash
open results_segb_h2k6_video/tripod_episode_00.mp4
cat runs_segb_h2k6_5m_seed0/summary.json
```

Also inspect alternate policies with:

```bash
python3 evaluate.py --h2k6-model runs_segb_h2k6_5m_seed0/best_progress.npz --episodes 1 --video --out-dir results_h2k6_raw_progress
python3 evaluate.py --h2k6-model runs_segb_h2k6_5m_seed0/best_survival.npz --episodes 1 --video --out-dir results_h2k6_survival
```

`best_stable_progress` prefers the largest progress among fully surviving training candidates; if none has survived all 1,000 steps, it falls back to the longest-lived archive candidate. This presentation preference affects evaluation only and **does not alter the archive fitness or training reward**.

For a matched **held-out** comparison to PPO:

```bash
python3 evaluate.py --model runs_c_16x1024_seed0/tripod_ppo_final.zip --episodes 10 --seed 100 --out-dir results_ppo_h2k6_compare
python3 evaluate.py --h2k6-model runs_segb_h2k6_5m_seed0/best_stable_progress.npz --episodes 10 --seed 100 --out-dir results_segb_h2k6_5m_seed0
python3 compare_ppo_segb.py --ppo results_ppo_h2k6_compare/summary.json --segb results_segb_h2k6_5m_seed0/summary.json
```

**Verification:** NumPy tests cover paired same-seed rollouts, private evidence, Q2, PAES, budget counts, checkpoint resume, GRU action shape. The publishing environment does not have `mujoco` installed; **actual MuJoCo 5M training has not been run here**, and no gait quality claim is made. Multiple seeds will be necessary for a paper-grade conclusion.
