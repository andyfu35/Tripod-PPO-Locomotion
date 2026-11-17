#!/usr/bin/env bash
# Continue 5M -> 10M, then optionally 10M -> 20M without discarding private psi.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SOURCE_DIR="${1:-runs_segb_h2k6_5m_seed0}"
TARGET_STEPS="${2:-10000000}"
DEST_DIR="${3:-runs_segb_h2k6_10m_seed0}"

if [[ ! -f "$SOURCE_DIR/checkpoint.npz" ]]; then
  echo "ERROR: Missing source H2-K6 checkpoint: $SOURCE_DIR/checkpoint.npz" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_DIR/best_stable_progress.npz" ]]; then
  echo "ERROR: Missing source best_stable_progress.npz; complete the source run first" >&2
  exit 2
fi
if ! [[ "$TARGET_STEPS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: TARGET_STEPS must be an integer" >&2
  exit 2
fi
if [[ "$SOURCE_DIR" == "$DEST_DIR" ]]; then
  echo "ERROR: New results must use a different destination directory" >&2
  exit 2
fi

python3 preflight.py
python3 -m unittest discover -s tests -p 'test_h2k6*.py' -v

# An existing destination checkpoint has priority for interruption recovery.
# The source 5M checkpoint is never changed.
python3 train_segb_h2k6.py \
  --total-steps "$TARGET_STEPS" \
  --worlds 16 --seed 0 \
  --resume-from "$SOURCE_DIR/checkpoint.npz" \
  --out-dir "$DEST_DIR"

SOURCE_LABEL="$(basename "$SOURCE_DIR")"
DEST_LABEL="$(basename "$DEST_DIR")"
BEFORE_RESULTS="results_compare_${SOURCE_LABEL}"
AFTER_RESULTS="results_compare_${DEST_LABEL}"
VIDEO_RESULTS="results_compare_${DEST_LABEL}_video"
REPORT_RESULTS="results_compare_${DEST_LABEL}_report"

# Same independent episodes, same seed 100..109, for BOTH policies.
python3 evaluate.py --h2k6-model "$SOURCE_DIR/best_stable_progress.npz" \
  --episodes 10 --seed 100 --out-dir "$BEFORE_RESULTS"
python3 evaluate.py --h2k6-model "$DEST_DIR/best_stable_progress.npz" \
  --episodes 10 --seed 100 --out-dir "$AFTER_RESULTS"
python3 evaluate.py --h2k6-model "$DEST_DIR/best_stable_progress.npz" \
  --episodes 1 --seed 100 --video --camera fixed --out-dir "$VIDEO_RESULTS"

PPO_MODEL="runs_c_16x1024_seed0/tripod_ppo_final.zip"
PPO_ARGS=()
if [[ -f "$PPO_MODEL" ]]; then
  python3 evaluate.py --model "$PPO_MODEL" --episodes 10 --seed 100 \
    --out-dir results_compare_ppo_5m
  PPO_ARGS=(--ppo results_compare_ppo_5m/summary.json)
else
  echo "NOTE: PPO model not found locally; comparison will include SEGB only"
fi
python3 compare_segb_h2k6_milestones.py \
  --before "$BEFORE_RESULTS/summary.json" \
  --after "$AFTER_RESULTS/summary.json" \
  "${PPO_ARGS[@]}" \
  --csv "$REPORT_RESULTS/heldout_comparison.csv" \
  --png "$REPORT_RESULTS/heldout_mean_speed.png"

echo "Video: $VIDEO_RESULTS/tripod_episode_00.mp4"
echo "Results: $REPORT_RESULTS/heldout_comparison.csv"
echo "Chart: $REPORT_RESULTS/heldout_mean_speed.png"
