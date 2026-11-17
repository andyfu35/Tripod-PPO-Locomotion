"""Regression tests for SEGB V1 early-fall bugs, V2 interaction ledger."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from segb_core import GENOME_LENGTH, Gene, ParetoGeneBank
from segb_v2_core import local_mutation_direction, new_gene, score_single_life
from train_segb_v2 import run_round, load_checkpoint, save_checkpoint
from train_segb import budget_plan
import train_segb_v2


class EarlyFallVecEnv:
    """Mimics SB3 auto-resets on done, with a fixed lifespan by slot."""
    def __init__(self, periods):
        self.num_envs = len(periods)
        self.periods = periods
        self.ages = [0] * self.num_envs
        self.calls = 0
        self.interactions = 0
        self.epoch_seed = None
        self.closed = False

    def seed(self, seed):
        self.epoch_seed = seed

    def reset(self):
        self.ages[:] = [0] * self.num_envs
        return np.zeros((self.num_envs, 51), dtype=np.float32)

    def step(self, actions):
        assert actions.shape == (self.num_envs, 12)
        self.calls += 1
        self.interactions += self.num_envs
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []
        for i, period in enumerate(self.periods):
            self.ages[i] += 1
            died = self.ages[i] >= period
            dones[i] = died
            # A faster but unsustainable runner; even the terminal step
            # can contain positive progress from the previous displacement.
            speed = 1.0 if period < 10 else 0.20
            rewards[i] = speed + (0.0 if died and period < 10 else 0.2)
            infos.append({"progress_speed_mps": speed, "alive": not (died and period < 10)})
            if died:
                self.ages[i] = 0  # VecEnv's automatic reset
        return np.zeros((self.num_envs, 51), dtype=np.float32), rewards, dones, infos

    def close(self):
        self.closed = True


class TestSEGBV2(unittest.TestCase):
    def test_exact_five_million_schedule(self):
        plan = budget_plan(5_000_000, 1000, 16)
        self.assertEqual((len(plan), plan[-1], plan.count(16)), (313, 8, 312))
        self.assertEqual(sum(w * 1000 for w in plan), 5_000_000)

    def test_original_survival_reward_reset_loophole(self):
        self.assertAlmostEqual(980 * .2 / 1000, 0.196)
        self.assertAlmostEqual(1000 * .2 / 1000, 0.2)

    def test_one_life_fixed_denominator_penalizes_fast_fall(self):
        fast = score_single_life(Gene(1, np.zeros(GENOME_LENGTH, np.float32)),
                                 total_progress=2*1., total_reward=2*1.,
                                 steps=2, horizon=10, fell=True)
        slow = score_single_life(Gene(2, np.zeros(GENOME_LENGTH, np.float32)),
                                 total_progress=10*.2, total_reward=10*.4,
                                 steps=10, horizon=10, fell=False)
        self.assertLess(fast.mean_reward, slow.mean_reward)
        self.assertEqual((fast.longevity, slow.longevity), (.2, 1.))
        self.assertEqual((fast.fall_count, slow.fall_count), (1, 0))
        with self.assertRaises(ValueError):
            score_single_life(slow, total_progress=0, total_reward=0, steps=11, horizon=10, fell=False)

    def test_mutation_sparse_and_repeatable_no_gait_prior(self):
        for seed in (1, 2, 20, 41, 101, 42):
            d1 = local_mutation_direction(seed)
            d2 = local_mutation_direction(seed)
            np.testing.assert_array_equal(d1, d2)
            self.assertEqual(d1.shape, (GENOME_LENGTH,))
            self.assertLess(np.count_nonzero(d1), GENOME_LENGTH // 2)
            self.assertGreater(np.count_nonzero(d1), 0)

    def test_early_fall_immediately_replaces_gene_same_budget(self):
        env = EarlyFallVecEnv([2, 10])
        bank = ParetoGeneBank(1024, 5)
        history = []
        idx, best = run_round(
            env, bank=bank, rng=np.random.default_rng(123), gene_index=0,
            max_steps=10, horizon=10, seed=88,
            de_novo_fraction=.2, history=history, base_steps=0, best=None,
        )
        self.assertEqual(env.interactions, 20)
        self.assertEqual(env.calls, 10)
        self.assertEqual(idx, 6)  # 5 failed life trials and 1 full survivor
        self.assertEqual(len(history), 6)
        self.assertEqual(sum(x['life_steps'] for x in history), 20)
        self.assertEqual(sum(x['censored'] for x in history), 0)
        self.assertEqual(sum(x['fall_count'] for x in history), 5)
        self.assertTrue(all(0 <= x['training_steps_total'] <= 20 for x in history))
        self.assertAlmostEqual(bank.genes[max(bank.genes)].longevity, 1.0)
        self.assertIsNotNone(best)

    def test_partial_life_censored_not_promoted(self):
        env = EarlyFallVecEnv([3, 10])
        bank = ParetoGeneBank()
        history = []
        idx, best = run_round(
            env, bank=bank, rng=np.random.default_rng(0), gene_index=0,
            max_steps=10, horizon=10, seed=3, de_novo_fraction=.2,
            history=history, base_steps=0, best=None,
        )
        self.assertEqual(env.interactions, 20)
        self.assertEqual(sum(x['life_steps'] for x in history), 20)
        self.assertEqual(sum(x['censored'] for x in history), 1)
        censored = next(x for x in history if x['censored'])
        self.assertEqual(censored['life_steps'], 1)
        self.assertEqual(censored['admitted'], 0)
        self.assertEqual(len(bank.genes) > 0, True)

    def test_checkpoint_and_exact_end_to_end_budget(self):
        made = []
        def maker(n_worlds, *, seed):
            env = EarlyFallVecEnv([2] * n_worlds)
            made.append(env)
            return env
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            argv = ['train_segb_v2.py', '--total-steps', '40', '--steps-per-gene', '5',
                    '--worlds', '4', '--save-every', '20', '--out-dir', str(out)]
            with patch.object(sys, 'argv', argv), patch.object(train_segb_v2, 'make_env', side_effect=maker):
                train_segb_v2.main()
            d = json.loads((out / 'summary.json').read_text())
            self.assertEqual(d['interaction_steps'], 40)
            self.assertGreater(d['completed_genes'], 8)  # more than the V1 fixed 8
            self.assertEqual(sum(x.interactions for x in made), 40)
            self.assertEqual(len(made), 1)  # reuse subprocesses across rounds
            self.assertTrue(made[0].closed)
            with patch.object(sys, 'argv', argv+['--resume']), patch.object(train_segb_v2, 'make_env', side_effect=maker):
                train_segb_v2.main()
            self.assertEqual(len(made), 1)
            self.assertTrue((out/'best_reward.npz').exists())
            self.assertTrue((out/'best_survival.npz').exists())
            self.assertTrue((out/'candidate_history.csv').exists())


if __name__ == '__main__':
    unittest.main()
