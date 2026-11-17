#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python3 preflight.py
python3 smoke_test.py
python3 -m unittest discover -s tests -p 'test_h2k6_tripod.py' -v

# The H2-K6 trainer resumes automatically from a matching checkpoint.
# All 5M env.step() calls include BOTH matched parent and child rollouts.
python3 train_segb_h2k6.py \
  --worlds 16 --total-steps 5000000 --seed 0 \
  --out-dir runs_segb_h2k6_5m_seed0

python3 evaluate.py \
  --h2k6-model runs_segb_h2k6_5m_seed0/best_stable_progress.npz \
  --episodes 10 --seed 100 \
  --out-dir results_segb_h2k6_5m_seed0

python3 evaluate.py \
  --h2k6-model runs_segb_h2k6_5m_seed0/best_stable_progress.npz \
  --episodes 1 --seed 100 --video --camera fixed \
  --out-dir results_segb_h2k6_video
