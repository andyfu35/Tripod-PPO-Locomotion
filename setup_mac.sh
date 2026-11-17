#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo 'Checking disk space before installing PyTorch and MuJoCo...'
python3 - <<'CHECK'
import shutil
import sys
from pathlib import Path
free = shutil.disk_usage(Path.cwd()).free / 1024**3
print(f'Free space: {free:.2f} GiB')
if free < 3.0:
    print('ERROR: Less than 3 GiB free. Clear storage before installation.', file=sys.stderr)
    raise SystemExit(1)
if free < 5.0:
    print('WARNING: 5 GiB or more is recommended for initial install and logs.')
CHECK

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python3 -m pip install --no-cache-dir -r requirements.txt
python3 preflight.py
echo 'Environment ready. Run: source .venv/bin/activate && bash run_experiment_c.sh'
