#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python3 preflight.py
python3 smoke_test.py
python3 -m unittest discover -s tests -v
python3 train.py --n-envs 16 --n-steps 1024 --timesteps 5000000 --seed 0 --out-dir runs_c_16x1024_seed0
python3 evaluate.py --model runs_c_16x1024_seed0/tripod_ppo_final.zip --episodes 10 --out-dir results_c_16x1024_seed0
python3 evaluate.py --model runs_c_16x1024_seed0/tripod_ppo_final.zip --episodes 1 --video --camera fixed --out-dir results_c_16x1024_video
