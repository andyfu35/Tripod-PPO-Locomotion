"""Check installed libraries and disk headroom before launching PPO."""
import importlib
import shutil
import sys
from pathlib import Path

REQUIRED = {
    'numpy': 'numpy',
    'mujoco': 'mujoco',
    'gymnasium': 'gymnasium',
    'torch': 'torch',
    'stable-baselines3': 'stable_baselines3',
    'matplotlib': 'matplotlib',
    'imageio': 'imageio',
    'imageio-ffmpeg': 'imageio_ffmpeg',
}

def main():
    failed = []
    for label, module in REQUIRED.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            failed.append(f'{label}: {type(exc).__name__}: {exc}')
    if failed:
        print('ERROR: Python dependencies are incomplete:', file=sys.stderr)
        for problem in failed:
            print('  - ' + problem, file=sys.stderr)
        print('Free disk space and install with:', file=sys.stderr)
        print('  python3 -m pip install --no-cache-dir -r requirements.txt', file=sys.stderr)
        return 1
    free_gib = shutil.disk_usage(Path.cwd()).free / (1024 ** 3)
    print(f'PASS: All dependencies import correctly. Available: {free_gib:.2f} GiB.')
    if free_gib < 2:
        print('WARNING: Low free disk space; checkpoints/video could fail.', file=sys.stderr)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
