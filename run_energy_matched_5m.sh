#!/usr/bin/env bash
# Fresh 5M energy-reward experiment for BOTH PPO and SEGB-H2K6, same robot.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PPO_DIR="runs_ppo_energy_5m_seed0"
SEGB_DIR="runs_segb_h2k6_energy_5m_seed0"
ENERGY_WEIGHT="0.10"
POWER_REF_W="50.0"

python3 preflight.py
python3 smoke_test.py
python3 -m unittest discover -s tests -p 'test_energy_experiment.py' -v

if [[ -f "$PPO_DIR/tripod_ppo_final.zip" ]]; then
  python3 - "$PPO_DIR/train_config.json" "$ENERGY_WEIGHT" "$POWER_REF_W" <<'PY'
import json,sys
kw=json.load(open(sys.argv[1]))['env_kwargs']
if abs(float(kw.get('energy_weight',-1))-float(sys.argv[2]))>1e-12 or abs(float(kw.get('energy_power_ref_w',-1))-float(sys.argv[3]))>1e-12:
    raise SystemExit('Existing PPO model was trained with different energy settings')
PY
  echo "PPO model already finished with matching energy settings; skipping retrain."
elif [[ -f "$PPO_DIR/train_config.json" ]]; then
  echo "ERROR: Partial PPO output exists: $PPO_DIR. Keep it safe; resume or choose a fresh directory." >&2
  exit 2
else
  echo "=== TRAIN PPO from scratch, energy-weighted 5M ==="
  python3 train.py --timesteps 5000000 --n-envs 16 --n-steps 1024 \
    --seed 0 --energy-weight "$ENERGY_WEIGHT" --energy-power-ref-w "$POWER_REF_W" \
    --out-dir "$PPO_DIR"
fi

echo "=== TRAIN SEGB H2-K6 from scratch, energy-weighted EXACT 5M ==="
# SEGB's checkpoint can resume ONLY the new energy experiment, not old 5M/10M.
python3 train_segb_h2k6_energy.py --total-steps 5000000 --worlds 16 \
  --seed 0 --energy-weight "$ENERGY_WEIGHT" --energy-power-ref-w "$POWER_REF_W" \
  --out-dir "$SEGB_DIR"

echo "=== EVALUATE PPO + SEGB with identical held-out reset seeds 100..109 ==="
python3 evaluate.py --model "$PPO_DIR/tripod_ppo_final.zip" --episodes 10 \
  --seed 100 --out-dir results_ppo_energy_5m_seed0
python3 evaluate.py --h2k6-model "$SEGB_DIR/best_stable_progress.npz" --episodes 10 \
  --seed 100 --energy-weight "$ENERGY_WEIGHT" --energy-power-ref-w "$POWER_REF_W" \
  --out-dir results_segb_h2k6_energy_5m_seed0
python3 compare_ppo_segb_energy.py \
  --ppo results_ppo_energy_5m_seed0/summary.json \
  --segb results_segb_h2k6_energy_5m_seed0/summary.json \
  --out-dir results_energy_matched_5m_report

python3 evaluate.py --model "$PPO_DIR/tripod_ppo_final.zip" \
  --episodes 1 --seed 100 --video --camera fixed \
  --out-dir results_ppo_energy_5m_video
python3 evaluate.py --h2k6-model "$SEGB_DIR/best_stable_progress.npz" \
  --episodes 1 --seed 100 --video --camera fixed \
  --energy-weight "$ENERGY_WEIGHT" --energy-power-ref-w "$POWER_REF_W" \
  --out-dir results_segb_h2k6_energy_5m_video

echo "Done: results_energy_matched_5m_report/comparison.csv"
echo "PPO video: results_ppo_energy_5m_video/tripod_episode_00.mp4"
echo "SEGB video: results_segb_h2k6_energy_5m_video/tripod_episode_00.mp4"
