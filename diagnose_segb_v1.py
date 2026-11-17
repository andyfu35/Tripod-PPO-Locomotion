"""Read *existing* v1 SEGB results without modifying them or rerunning physics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=Path('runs_segb_5m_seed0'))
    p.add_argument('--evaluation', type=Path, default=Path('results_segb_5m_seed0/summary.json'))
    args = p.parse_args()
    path = args.run_dir / 'candidate_history.csv'
    if not path.is_file():
        p.error(f'Missing {path}; cannot diagnose actual training')
    with path.open(newline='', encoding='utf-8') as f:
        records = list(csv.DictReader(f))
    if not records:
        p.error('Empty candidate history')
    sizes = [int(r['fall_count']) for r in records]
    reward = [float(r['mean_reward']) for r in records]
    life = [float(r['longevity']) for r in records]
    sources = {'de_novo': sum(r['source']=='de_novo' for r in records),
               'inherited': sum(r['source']=='inherited' for r in records)}
    def stats(rs):
        return {'n': len(rs), 'median_falls_per_1000_steps': median(int(r['fall_count']) for r in rs),
                'best_reward_per_step': max(float(r['mean_reward']) for r in rs),
                'best_longevity': max(float(r['longevity']) for r in rs),
                'max_generation': max(int(r['generation']) for r in rs)}
    print('SEGB v1 diagnostic — no files changed')
    print(json.dumps({'total_genes': len(records), 'source_counts': sources,
                      'fraction_with_at_least_5_falls': sum(x>=5 for x in sizes)/len(sizes),
                      'mean_falls_per_1000_steps': mean(sizes),
                      'median_falls_per_1000_steps': median(sizes),
                      'best_reward_per_step': max(reward),
                      'best_longevity': max(life)}, indent=2, ensure_ascii=False))
    print('Trend by 1000-candidate window:')
    for a in range(0, len(records), 1000):
        print(f' {a:5d}..{min(a+1000, len(records))-1:5d}: '+json.dumps(stats(records[a:a+1000])))
    summary = args.run_dir / 'summary.json'
    if summary.exists():
        s = json.loads(summary.read_text())
        print('Saved champions:')
        for label, data in s.get('champions', {}).items():
            print(label, 'reward/step=', data.get('mean_reward'),
                  'longevity=', data.get('longevity'),
                  'falls/1000=', data.get('fall_count'))
    if args.evaluation.is_file():
        report = json.loads(args.evaluation.read_text())
        print('Held-out 10-episode evaluation:')
        print(json.dumps({key: report.get(key) for key in
                          ('mean_survival_seconds', 'survival_rate', 'mean_net_speed_mps')}, indent=2))


if __name__ == '__main__':
    main()
