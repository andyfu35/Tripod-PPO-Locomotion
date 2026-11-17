# SEGB tripod locomotion: exactly 5,000,000 control interactions

This is an **SEGB adaptation** built alongside, NOT in place of, the successful PPO Experiment C. The original PPO training script and PPO policy are preserved. SEGB has **no PPO gradients**.

## One-command training (Mac)

From the `Tripod-PPO-Locomotion` repository directory:

```bash
source .venv/bin/activate
bash run_segb_5m.sh
```

This runs dependency/robot/offline algorithm checks, exactly 5M training interactions, 10 held-out evaluations, and one video. If interrupted after a checkpoint, running the script again uses `--resume`. Checkpoints and videos remain local; do not delete your PPO files.

Run the trainer by itself, without overwriting a previous experiment:

```bash
python3 train_segb.py --total-steps 5000000 --steps-per-gene 1000 \
  --worlds 16 --seed 0 --out-dir runs_segb_5m_seed0
```

## Exact interaction accounting

- **Same robot**: generated `models/tripod.xml` (three legs / 12 actuators / 120 degrees)
- **Same 51 observations, 12 actions, joint servo smoothing**: `TripodEnv` unchanged
- **Same MuJoCo physics**: 0.002 s physics timestep and 10 simulation substeps per action; 0.02 s control timestep
- **Same two training reward terms**: signed radial progress speed + 0.2 alive; no learned gait prior, gait bonus, energy penalty, or contact bonus
- **Same requested budget**: PPO Experiment C requests 5,000,000 steps and SB3 actually rounds up to 5,013,504 due to rollouts; SEGB is constrained to exactly 5,000,000 control interactions. This difference (0.27%) is reported, not hidden.
- **16 parallel environments**: 312 full batches of 16 genes and one last batch of 8 genes; each gene evaluated for precisely 1000 control steps, including automatic resets after falls
- 5000 completed gene evaluations; SEGB trains without optimizer/gradient update counts
- Training steps are simulated `env.step()` calls, not individual MuJoCo integration substeps. Validation and videos run after training and do not consume the 5M training budget (same principle as PPO post-training evaluation).

## Evolution rule

- Controller **gene**: a deterministic MLP `(51, 64, 64, 12)` of 8,268 float32 parameters with tanh activations, acting on the same normalized joint-target interface as PPO. This is a *tripod-specific MLP genotype*, not a claim of copying the original Hopper recurrent GRU code.
- Each 1000-step gene trial collects the same raw environment reward plus physical diagnostics.
- **Global Pareto Bank**: two capabilities, (1) mean signed radial progress speed and (2) episode longevity. Longevity is `sum(length_of_uninterrupted_life_segment**2) / (steps_per_gene**2)`; higher values mean less frequent falling; this is for archive selection, **not a new environment reward**.
- **Archive**: nondominated genes, with capacity 1,024 and PAES-style adaptive 2D grid depth=5 on overflow. It is a PAES-style **archive mechanism**, not a separate (1+1) PAES optimizer.
- **Parent selection**: empirical percentile-normalized scores `s_j` in `[0,1]`, then `Q=(0.10+sum(s_j))**2` and sample parents with probability proportional to Q. Retains nonzero probability for low-ranked Pareto genes.
- **New candidate mixture**: 80% inherited from the global bank with directional mutation, 20% entirely new random genes; no crossover. If archive is empty, candidates are all new genes.
- **Local guidance**: a mutation direction that improves either capability is preserved as lineage-local positive evidence and may be reused with perturbed magnitude; 35% of inherited proposals always use fresh Gaussian directions even when guidance exists. Direction evidence does not require episode survival.
- **Saved policies**: `best_reward.npz`, `best_progress.npz`, `best_survival.npz`, `balanced.npz`.

These are explicitly documented adaptations to the radial tripod environment; avoid calling this an exact reproduction of SEGB Hopper's recurrent-mutation/GPU implementation. No GPU/MuJoCo-Warp parallel physics is used here; as in the PPO Experiment C, MuJoCo physics runs in CPU subprocesses. CPU worker limits can make 16-world execution slower on an 8-core Mac.

## Results

```text
runs_segb_5m_seed0/
  checkpoint.npz             # atomic resumable archive/weights/PRNG state
  candidate_history.csv      # one row per evaluated Gene
  summary.json               # budget, bank, lineage and champion metrics
  best_reward.npz
  best_progress.npz
  best_survival.npz
  balanced.npz
results_segb_5m_seed0/
  summary.json               # 10 held-out episodes, same evaluator as PPO
  gait_and_trajectory.png
  episode_00.csv ...
results_segb_5m_video/
  tripod_episode_00.mp4
```

Video and 10 episodes:

```bash
python3 evaluate.py --segb-model runs_segb_5m_seed0/best_reward.npz \
  --episodes 10 --seed 100 --out-dir results_segb_5m_seed0
python3 evaluate.py --segb-model runs_segb_5m_seed0/best_reward.npz \
  --episodes 1 --seed 100 --video --camera fixed --out-dir results_segb_5m_video
open results_segb_5m_video/tripod_episode_00.mp4
```

To inspect the best **survival** gene instead of the highest raw reward gene:

```bash
python3 evaluate.py --segb-model runs_segb_5m_seed0/best_survival.npz \
  --episodes 1 --seed 100 --video --camera fixed --out-dir results_segb_survival_video
```

### PPO vs SEGB apples-to-apples held-out evaluation

Train both methods with the same robot and independently compare using the **same 10 evaluation reset seeds 100 through 109**:

```bash
python3 evaluate.py --model runs_c_16x1024_seed0/tripod_ppo_final.zip \
  --episodes 10 --seed 100 --out-dir results_ppo_c_10ep
python3 evaluate.py --segb-model runs_segb_5m_seed0/best_reward.npz \
  --episodes 10 --seed 100 --out-dir results_segb_5m_seed0
python3 compare_ppo_segb.py --ppo results_ppo_c_10ep/summary.json \
  --segb results_segb_5m_seed0/summary.json
```

A result after a single seed is only an exploratory comparison, not proof that one algorithm dominates the other. Compare 5-10 independent seeds for publishable conclusions.

## Reproducibility and caveats

- A checkpoint contains the gene bank, best sampled policy, RNG state, configuration/robot XML SHA and completed gene count. Run `--resume` with the exact same configuration.
- A partial batch is *not* committed before the next checkpoint. Upon interruption, re-evaluation after the last checkpoint restores the experiment's committed-step ledger; do not report total transient CPU steps as the formal 5M budget. Resume starts from the last committed full batch.
- This environment has not completed any full MuJoCo physics benchmark here. Runtime preflight and output inspection must be performed on your Mac, and **no locomotion score or SEGB success is claimed in advance**.
