#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python3 preflight.py
python3 smoke_test.py
python3 -m unittest discover -s tests -p 'test_segb*.py' -v

if [[ -f runs_segb_5m_seed0/checkpoint.npz ]]; then
  python3 train_segb.py --resume --total-steps 5000000 --steps-per-gene 1000 --worlds 16 --seed 0 --out-dir runs_segb_5m_seed0
else
  python3 train_segb.py --total-steps 5000000 --steps-per-gene 1000 --worlds 16 --seed 0 --out-dir runs_segb_5m_seed0
fi

python3 evaluate.py --segb-model runs_segb_5m_seed0/best_reward.npz --episodes 10 --seed 100 --out-dir results_segb_5m_seed0
python3 evaluate.py --segb-model runs_segb_5m_seed0/best_reward.npz --episodes 1 --seed 100 --video --camera fixed --out-dir results_segb_5m_video
