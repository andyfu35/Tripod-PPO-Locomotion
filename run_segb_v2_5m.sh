#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python3 preflight.py
python3 smoke_test.py
python3 -m unittest discover -s tests -p 'test_segb_v2*.py' -v

DIR="runs_segb_v2_5m_seed0"
if [[ -f "$DIR/checkpoint.npz" ]]; then
  python3 train_segb_v2.py --resume --total-steps 5000000 --steps-per-gene 1000 --worlds 16 --seed 0 --out-dir "$DIR"
else
  python3 train_segb_v2.py --total-steps 5000000 --steps-per-gene 1000 --worlds 16 --seed 0 --out-dir "$DIR"
fi

python3 evaluate.py --segb-model "$DIR/best_reward.npz" --episodes 10 --seed 100 --out-dir results_segb_v2_5m_seed0
python3 evaluate.py --segb-model "$DIR/best_survival.npz" --episodes 10 --seed 100 --out-dir results_segb_v2_survival_seed0
python3 evaluate.py --segb-model "$DIR/best_reward.npz" --episodes 1 --seed 100 --video --camera fixed --out-dir results_segb_v2_video
