# SEGB tripod V2 — single-life evaluation and failure analysis

This is a **new experiment, not a hotfix of the completed V1 run**. It keeps PPO
Experiment C, `TripodEnv`, motor constraints, and reward (`signed radial progress
speed + 0.2 alive`) unchanged. SEGB still uses 16 CPU MuJoCo worlds, the
Q²/Pareto archive, PAES-style density with capacity 1024 and depth 5,
80% inheritance and 20% de novo, and no prescribed gait.

## Why V1 produced falls in under one second

1. V1 evaluated a gene for exactly 1,000 steps **even if the robot fell every
   ~50 steps**, resetting and reusing the same gene. All but 20 of 1,000 steps
   could receive the `+0.2 alive` term, so a gene falling every second earns
   about 0.196 survival reward/step, versus 0.200 for a perfect survivor.
2. Selection and `best_reward.npz` thus promoted some fast-falling strategies.
   The separate longevity objective limited but did not fix this reward bias.
3. Reusing a failed gene across resets spent most of the 5M-step budget without
   evaluating new offspring (only 5,000 genotypes in the entire V1 run).
4. Full 8,268-parameter Gaussian mutations at sigma up to 0.16 can destroy
   useful inherited controllers before meaningful refinement.

These are verified **source-code design defects**; whether they fully explain
any one user's observed run needs its `candidate_history.csv` and `summary.json`.

## V2 method (same 5M interactions, no extra gait reward)

- Each candidate gets **one uninterrupted trial**, up to 1,000 control steps.
- On fall, score and archive immediately, then assign **a new candidate** to
  that world from the next step, keeping every interaction in the budget.
- The two archive abilities are signed radial progress normalized by **1,000**
  and first-life length normalized by **1,000**. Raw episode reward is also
  normalized by **1,000**, avoiding a faller's misleading conditional average.
- Mutations perturb either output biases, the output layer, or a sparse 8%
  subset of the 8,268 weights, with smaller magnitudes (0.003–0.04). A seed
  preserves the exact direction for inherited positive mutation evidence.
- At each synchronization/checkpoint boundary (1,000 vector steps), any
  incomplete trial is **censored** and excluded from archive selection. These
  used steps remain counted against the 5M-step training budget.
- 16 worlds × 1,000 steps per full round, with the final partial 8-world round:
  **exactly 5,000,000** Gym `env.step()` interactions. A physical control
  step is 0.02 s, itself comprising 10 MuJoCo physics substeps.
- All 5M are counted including failed lives and censored partial trials.
  Number of completed genes is data-dependent, *not* fixed at 5,000.
- V2 checkpoint has an incompatible version and uses a **fresh** output
  directory, `runs_segb_v2_5m_seed0`. Do not point V2 at a V1 checkpoint.

## Mac: read existing run diagnostics before retraining

```bash
source .venv/bin/activate
python3 diagnose_segb_v1.py --run-dir runs_segb_5m_seed0
```

## Mac: train and evaluate V2

```bash
git fetch origin
git switch experiment/segb-tripod-single-life-fix-20261117
source .venv/bin/activate
bash run_segb_v2_5m.sh
```

Video:

```bash
open results_segb_v2_video/tripod_episode_00.mp4
```

Compare held-out results with PPO (evaluate at seeds 100..109, separate from
training), with the same simulator/reward and fixed training interaction budget.
Note that the SEGB policy search distribution was also changed; **V1→V2 is
not a clean single-factor ablation**.

Verification in this environment: numpy-only offline unit and simulated
vector-environment tests; MuJoCo could not be installed in the publishing
container. **No full physics-training outcome is claimed.**
