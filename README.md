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

## Clone and run from GitHub

This is the independent repository. It does **not** modify Self-Evolving-Gene-Bank.

```bash
cd ~/PycharmProjects
git clone https://github.com/andyfu35/Tripod-PPO-Locomotion.git
cd Tripod-PPO-Locomotion

# Recover disk space first if previous pip install failed with Errno 28.
df -h .
bash setup_mac.sh
source .venv/bin/activate
bash run_experiment_c.sh
```

Model files, checkpoints and rendered videos remain local and are ignored by Git.

## Robot blueprint

Run `python3 draw_blueprint.py` after installing dependencies to regenerate the robot blueprint locally.

## Verification status

Structural and math tests passed when packaged; real MuJoCo physics and
5-million-step PPO training must be run and verified on the host computer.

## SEGB 5-million-step comparison

An additional SEGB tripod experiment is provided without modifying the successful PPO training setup. See [README_SEGB.md](README_SEGB.md) for the Gene Bank rules, identical environment/action/reward definitions and exact interaction ledger.

```bash
source .venv/bin/activate
bash run_segb_5m.sh
```

It evolves 5,000 candidate policies on 16 CPU MuJoCo worlds for exactly 5,000,000 training control steps, using PAES-style archive density, Q^2 parent sampling, 80% inheritance mutation / 20% de novo. PPO Experiment C remains intact. Compare both methods with the same held-out seeds using `compare_ppo_segb.py`.

## SEGB V2: early-fall learning fix (5M steps)

If the original SEGB policy repeatedly falls within one second, consult [README_SEGB_V2.md](README_SEGB_V2.md). The new experiment scores one uninterrupted life per gene, immediately replaces failed genes, and uses local mutations without changing the MuJoCo model or reward. Run `python3 diagnose_segb_v1.py` to inspect the original results and `bash run_segb_v2_5m.sh` for an independent V2 5-million-step run. PPO and SEGB V1 scripts and output directories are preserved.
