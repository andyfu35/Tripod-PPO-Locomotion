# Tripod-PPO-Locomotion

Independent MuJoCo + Stable-Baselines3 PPO experiment for a radial three-legged robot (3 legs x 4 DOF, 120-degree symmetry). This repository is standalone and does **not** require or modify any other robot or SEGB repository.

## Experiment C: fixed training configuration

| Setting | Value |
| --- | ---: |
| Independent MuJoCo worlds | 16 |
| Steps per world per PPO rollout | 1,024 |
| Collected transitions per PPO update | 16,384 |
| Total requested transitions | 5,000,000 |
| PPO rollout updates | 306 |
| Actual collected transitions | 5,013,504 |
| Mini-batch size | 256 |
| Epochs per rollout | 10 |
| Seed | 0 |

Each of the 16 worlds shares **one PPO network**, rather than training 16 independent policies.

## Reward and control

No prescribed gait. The reward has two terms only:

```text
reward = (distance_from_start_now - distance_from_start_previous) / dt + 0.2 * alive
```

The 12 desired joint position targets are low-pass filtered and rate-limited to reduce 20 ms target jumps. The agent observes 51 values. Old 49-observation V1 policy archives are not compatible. This design may still learn unstable locomotion; full physics training has not been validated in this packaging environment.

## Install on macOS

> If a previous installation failed with "No space left on device", free storage before creating a new virtual environment. Recommended free space: at least 5 GiB (larger is safer). This project uses its own `.venv` and does not modify the old project.

```bash
cd Tripod-PPO-Locomotion
df -h .
bash setup_mac.sh
source .venv/bin/activate
bash run_experiment_c.sh
```

The setup script uses `pip --no-cache-dir` and checks for incomplete installations. You can run `python3 preflight.py` at any time to check installed dependencies without launching training.

## Model evaluation and video

```bash
source .venv/bin/activate
python3 evaluate.py --model runs_c_16x1024_seed0/tripod_ppo_final.zip --episodes 10 --out-dir results_c_16x1024_seed0
python3 evaluate.py --model runs_c_16x1024_seed0/tripod_ppo_final.zip --episodes 1 --video --camera fixed --out-dir results_c_16x1024_video
```

Check duration, net displacement, and foot contacts. A high short-term reward is not evidence of stable walking. Output directories and `.venv` are ignored by Git.

## Publish to a NEW GitHub repository

1. Go to https://github.com/new and create an **empty private** repository named `Tripod-PPO-Locomotion` under your account. Leave README, .gitignore and license unchecked because they are already in this folder.
2. Run from the unzipped project root:

   ```bash
   bash publish_to_github.sh andyfu35 Tripod-PPO-Locomotion
   ```

3. Once pushed, your Mac can freshly clone it:

   ```bash
   git clone https://github.com/andyfu35/Tripod-PPO-Locomotion.git
   ```

**Note:** The connected GitHub tool used to prepare this project does not have a create-repository action, so the empty repository must be created by the account owner before pushing.

## Continue H2-K6 from 5M to 10M without discarding private history

From this experiment branch, use `bash run_segb_h2k6_continue.sh` to continue the already completed 5M checkpoint for another 5M interactions in a separate output directory. Both initial and extended policies are then evaluated on the same held-out seeds and compared with PPO when the PPO model is available. See [README_SEGB_CONTINUATION.md](README_SEGB_CONTINUATION.md) for commands and validation details.

## Fresh 5M PPO + SEGB H2-K6 with a matched motor-work penalty

Run `bash run_energy_matched_5m.sh` from the dedicated energy experiment branch to **train PPO and SEGB from scratch**, then compare the same 10 held-out reset seeds. Both use the same motor mechanical-work cost; SEGB keeps its per-Gene signed private history and adds a third energy-aware locomotion archive ability. No old PPO or SEGB results are overwritten. See [README_ENERGY_5M.md](README_ENERGY_5M.md) for equations, run instructions and caveats.
