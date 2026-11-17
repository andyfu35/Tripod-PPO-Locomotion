# SEGB H2-K6: continue from 5M to 10M (and optionally 20M)

This is a **continuation of the same evolutionary run**, not a fresh SEGB seed, reset of the Gene Bank, PPO fine-tuning, reward shaping, or a new capability-axis experiment.

## What remains unchanged

- Identical `TripodEnv` and MuJoCo XML (verified using the recorded SHA-256 hash).
- 51 observation values, 12 servo targets, the same 26-hidden-unit GRU.
- The exact original two capabilities: `progress = net_distance / 20s` and `longevity = uninterrupted_life_steps / 1000`.
- Original environment reward: signed outward progress speed + 0.2 while alive.
- 16 worlds, each parent and child evaluated from the same reset seed, Q² selection, private history `Psi`, signed positive/negative mutation evidence, PAES, 20% de novo and 20% persistent random exploration.
- Both parent and child `env.step()` calls count against the cumulative interaction budget.
- The existing 5M experiment's checkpoints, champions, recorded summary, and videos are untouched.

## One command to add 5M additional interactions

Run these commands from the project root on your Mac, where your existing 5M run output already exists:

```bash
cd ~/PycharmProjects/Tripod-PPO-Locomotion
git fetch origin
git switch --track origin/experiment/segb-h2k6-continuation-10m-20261117
source .venv/bin/activate
bash run_segb_h2k6_continue.sh
```

`bash run_segb_h2k6_continue.sh` restores `runs_segb_h2k6_5m_seed0/checkpoint.npz` and continues until **10,000,000 total environment interactions**, in the **new** directory `runs_segb_h2k6_10m_seed0`.

The checkpoint loader refuses continuation if any evolutionary setting is changed, the robot XML hash differs, or the requested cumulative budget is not greater than the saved interaction counter. Only `total_steps`, `output`, and `checkpoint_every` may change. A new `continuation_provenance.json` records the source checkpoint hash and executed step count. Restart the same command after an interruption: it will resume from the new destination checkpoint, not reload the 5M baseline.

**Important:** the 5M baseline checkpoint is a real evolutionary boundary. The 10M result is a genuine continuation from that completed state, not numerically identical to an uninterrupted single-call 10M run (whose final partial-batch boundary may occur differently).

## Outputs to review

```bash
cat runs_segb_h2k6_10m_seed0/summary.json
cat results_compare_runs_segb_h2k6_10m_seed0_report/heldout_comparison.csv
open results_compare_runs_segb_h2k6_10m_seed0_report/heldout_mean_speed.png
open results_compare_runs_segb_h2k6_10m_seed0_video/tripod_episode_00.mp4
```

The runner re-evaluates the **original 5M checkpoint**, **extended 10M checkpoint**, and (if locally available) **the PPO 5M model**, all on the identical 10 independent reset seeds `100..109`. The comparison prints mean net speed, survival time, and full-episode survival rate. This is a held-out exploratory comparison; repeat with additional train seeds and test seeds before publication.

**Training score is not held-out performance:** the original reported `progress_mps` ~0.196 m/s on its best training episode, not a guaranteed speed in these 10 independent tests. Improving the training best without increasing held-out speed does not demonstrate learning generalization.

## Extend farther only if the result justifies it

To go from 10M to 20M (another 10M interactions) on the same lineage and private history:

```bash
bash run_segb_h2k6_continue.sh \
  runs_segb_h2k6_10m_seed0 20000000 runs_segb_h2k6_20m_seed0
```

The script compares the new 20M policy against the 10M source and PPO, in fresh results directories. A further 50M stage is possible by the same mechanism but is **not** recommended automatically if the 10M result already shows no held-out improvement.

## Scientific stopping criterion

Check both **held-out mean net speed** and **survival rate**. If speed increases but survival collapses, the result is a trade-off rather than a clear win. If the 10M stage stalls, do not simply assume 50M will solve it: inspect bank diversity (`bank_size`), generations, and private history, then consider an explicit exploration/learning ablation.

## Verification

Zero-MuJoCo unit tests verify: extension from a completed run, source checkpoint integrity, immutable evolutionary parameters, model-hash guard, exact cumulative interactions, uninterrupted private `Psi`/RNG state, and repeated launch recovery. The full 10M physical training must run on the user's Mac; this source change is not a precomputed gait result.
