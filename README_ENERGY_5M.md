# Energy-matched 5M: PPO vs SEGB H2-K6 tripod

This is a **fresh experiment** (seed 0), not a continuation of either pre-existing PPO or SEGB 5M/10M. Both methods use the same 12-DOF MuJoCo robot, reset randomization, 51 observations, 12 servo position actions, 50 Hz controller, servo smoothing, and identical energy-aware `TripodEnv` reward. Neither controller receives a prescribed gait or contact schedule.

## Energy is a physical **mechanical-work proxy**, not battery consumption

The model has 12 geared-1 joint `position` servos. During each active 0.02 s controller action, the environment reads `data.actuator_force` (joint torque in N m) and the corresponding joint's `qvel` (rad/s) after **each** of ten 0.002 s MuJoCo steps. All 12 servo torques are counted.

```
P_s  = sum_j abs(tau_j,s * angular_velocity_j,s)   [W]
E_t  = sum_s P_s * 0.002                         [J per control step]
C_t  = energy_weight * E_t / (0.02 * power_ref_w)
R_t  = signed_radial_progress_speed + 0.2*alive - C_t
```

Locked initial setting in `run_energy_matched_5m.sh`: **`energy_weight=0.10` and `energy_power_ref_w=50.0` W**. At mean absolute motor work power 50 W, each control-step reward is reduced by 0.10. This numerical choice is a **starting experiment**, not a tuned optimum. It omits holding-current losses, motor efficiency, battery, and regeneration; therefore never label the resulting Joules as measured electrical consumption. Absolute rather than positive-only mechanical power intentionally charges both driving and resisting joint motion.

`energy_weight=0` still uses the **original unmodified MuJoCo stepping path** and original two-term reward, so the previous experiments remain valid.

## PPO

- PPO **from scratch**, 16 worlds, 1,024 rollout steps/world, same PPO MLP256x256, LR 3e-4, gamma .995, etc.
- Requested 5,000,000 transitions; **actual 5,013,504** (rollout rounding as in the original PPO comparison, +0.270%).
- Model and training configuration stored separately in `runs_ppo_energy_5m_seed0/`.
- A partial PPO directory is not silently restarted by the wrapper. Keep it, repair resume manually or use a separate experiment folder.

## SEGB H2-K6

- Independent de-novo GRU26 founders; 80% inherited / 20% independent, persistent random directions within inherited offspring, Q² parent sampling.
- **Exactly 5,000,000** parent+child physical `env.step()` calls, across 8 same-reset pairs of MuJoCo worlds.
- Gene Bank **three capabilities** (maximize all):
  1. `F_progress = net_distance_m / 20`, fixed-horizon radial progress (m/s)
  2. `F_survival = uninterrupted_life_steps / 1000`
  3. `F_energy_aware_progress = F_progress - energy_weight * motor_work_J / (20 * power_ref_w)`
- This third axis is an **energy-aware locomotion** ability, not a bare low-work axis. Standing still with zero work gets 0 on the third axis, not a positive energy reward. Standstill may still be retained by the longevity axis, which is intentional.
- H2-K6 `PrivatePsi` retains the original signed direction, 3-component positive/negative capability deltas, paired parent/child life counts, admitted status, and same-reset seed. The energy-aware direction is marked as useful only when the child lives at least as long as its parent. The PAES archive uses the 3D Pareto front and a three-dimensional depth=5 adaptive density grid.
- **No old private histories are imported**: all genes are newborns. Archive is capped at 1,024.
- The default reported SEGB policy is `best_stable_energy_aware` (highest energy-aware progress among complete training survivors); `best_stable_progress` and unrestricted `best_energy_aware` are also retained for diagnosis.
- The checkpoint version is unique to energy experiments and rejects conflicting settings/XML. Restarting the wrapper resumes an interrupted matching SEGB energy checkpoint.

## Mac: run both from scratch

```bash
cd ~/PycharmProjects/Tripod-PPO-Locomotion

git fetch origin
git switch --track origin/experiment/tripod-segb-ppo-energy-5m-20261117
source .venv/bin/activate

bash run_energy_matched_5m.sh
```

The wrapper trains **PPO first, SEGB second**, *not concurrently*, to avoid CPU contention on a MacBook Air. The training results and videos use independent directories and never overwrite the previous PPO or SEGB runs.

## Outputs

```text
runs_ppo_energy_5m_seed0/tripod_ppo_final.zip
runs_segb_h2k6_energy_5m_seed0/best_stable_energy_aware.npz
runs_segb_h2k6_energy_5m_seed0/best_energy_aware.npz
runs_segb_h2k6_energy_5m_seed0/checkpoint.npz
runs_segb_h2k6_energy_5m_seed0/summary.json
results_ppo_energy_5m_seed0/summary.json
results_segb_h2k6_energy_5m_seed0/summary.json
results_energy_matched_5m_report/comparison.csv
results_energy_matched_5m_report/fixed20_speed.png
results_energy_matched_5m_report/survival_rate.png
results_energy_matched_5m_report/motor_power.png
results_ppo_energy_5m_video/tripod_episode_00.mp4
results_segb_h2k6_energy_5m_video/tripod_episode_00.mp4
```

To view:

```bash
cat results_energy_matched_5m_report/comparison.csv
open results_energy_matched_5m_report/fixed20_speed.png
open results_energy_matched_5m_report/survival_rate.png
open results_energy_matched_5m_report/motor_power.png
open results_ppo_energy_5m_video/tripod_episode_00.mp4
open results_segb_h2k6_energy_5m_video/tripod_episode_00.mp4
```

Both evaluations use held-out reset seeds **100–109**. The **primary speed metric** is net displacement / 20 seconds, not net displacement / actual lifespan: dying early therefore cannot inflate the primary score. Report survival fraction, power *while alive*, episode Joules and travel distance alongside one another. Lower episode Joules by itself is **not** evidence of better energy efficiency when a controller falls early.

To inspect the energy-efficiency champion separately:

```bash
python3 evaluate.py --h2k6-model runs_segb_h2k6_energy_5m_seed0/best_energy_aware.npz \
  --episodes 10 --seed 100 --energy-weight .10 --energy-power-ref-w 50 \
  --out-dir results_segb_h2k6_energy_aware_10ep
```

The policy comparison is 10-episode exploratory evidence, not publishable statistical proof. For a paper, repeat several independent training seeds, and consider a parameter sweep over `energy_weight` rather than treating 0.10 as optimal.

## Verification

- `tests/test_energy_experiment.py`: mechanical units, third axis, private 3-axis signed evidence, PAES/Q², same-seed pairing, distinct founders, exact budget and checkpoint rejection.
- `tests/test_energy_report.py`: early-fall-safe metrics.
- `tests/test_energy_physics.py`: actual MuJoCo regression with weight=0 vs weight=.1 trajectories, the reward difference, and positive motor work; run by GitHub Actions `energy-physics-smoke.yml`.
- These checks are **not** a completed 5M MuJoCo benchmark. The Mac must do the physical training.
