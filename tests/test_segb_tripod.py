"""Offline and zero-MuJoCo checks for exact-budget SEGB tripod training."""
import json
import tempfile
import unittest
import sys
from unittest.mock import patch
from pathlib import Path

import numpy as np

from segb_core import (ACT_DIM, GENOME_LENGTH, OBS_DIM, Gene, ParetoGeneBank,
                       SEGBPredictor, adaptive_cells, batched_actions, dominates,
                       gene_from_archive, load_training_state, propose_gene,
                       q2_probabilities, record_improvement, save_training_state,
                       write_policy)
from train_segb import budget_plan, evaluation_batch


class FakeVecEnv:
    def __init__(self, worlds):
        self.worlds = worlds
        self.steps = 0
        self.step_calls = 0
        self.last_seed = None

    def seed(self, seed):
        self.last_seed = seed

    def reset(self):
        return np.zeros((self.worlds, OBS_DIM), np.float32)

    def close(self):
        pass

    def step(self, actions):
        assert actions.shape == (self.worlds, ACT_DIM)
        self.step_calls += 1
        self.steps += self.worlds
        info = [{"progress_speed_mps": 1., "alive": True}
                for _ in range(self.worlds)]
        return self.reset(), np.full(self.worlds, 1.2, np.float32), np.zeros(self.worlds, bool), info


class TestSEGBTripod(unittest.TestCase):
    def make_gene(self, k, progress, alive):
        g = Gene(k, np.zeros(GENOME_LENGTH, np.float32))
        g.progress_mps = progress
        g.longevity = alive
        g.mean_reward = progress + 0.2 * alive
        return g

    def test_exact_five_million_no_ghost_worlds(self):
        plan = budget_plan(5_000_000, 1000, 16)
        self.assertEqual((len(plan), plan[-1]), (313, 8))
        self.assertEqual(sum(plan), 5000)
        self.assertEqual(sum(x * 1000 for x in plan), 5_000_000)
        self.assertEqual(plan.count(16), 312)
        with self.assertRaises(ValueError):
            budget_plan(5_000_001, 1000, 16)

    def test_mlp_actions_shape_and_predictor(self):
        rng = np.random.default_rng(2)
        p = rng.normal(0, 0.1, (3, GENOME_LENGTH)).astype(np.float32)
        x = rng.normal(size=(3, OBS_DIM)).astype(np.float32)
        outputs = batched_actions(x, p)
        self.assertEqual(outputs.shape, (3, ACT_DIM))
        self.assertTrue(np.isfinite(outputs).all())
        self.assertTrue(np.all(np.abs(outputs) <= 1))
        self.assertTrue(np.allclose(outputs[0], SEGBPredictor(p[0]).predict(x[0])[0], atol=1e-6))

    def test_pareto_front_and_q_weight(self):
        bank = ParetoGeneBank(capacity=4, depth=5)
        rng = np.random.default_rng(1)
        a = self.make_gene(0, 1., .2)
        b = self.make_gene(1, .4, .9)
        c = self.make_gene(2, .3, .1)
        self.assertTrue(bank.accept(a, rng))
        self.assertTrue(bank.accept(b, rng))
        self.assertFalse(bank.accept(c, rng))
        self.assertEqual(set(bank.genes), {0, 1})
        self.assertFalse(dominates(a, b))
        prob = q2_probabilities(bank.ordered())
        self.assertAlmostEqual(prob.sum(), 1)
        self.assertTrue(np.all(prob > 0))
        self.assertEqual(len(adaptive_cells(bank.ordered())), 2)

    def test_paes_archive_occupancy(self):
        rng = np.random.default_rng(15)
        bank = ParetoGeneBank(capacity=3, depth=5)
        for i in range(14):
            self.assertTrue(bank.accept(self.make_gene(i, float(i), float(14-i)), rng) or i not in bank.genes)
        self.assertEqual(len(bank.genes), 3)
        self.assertGreater(bank.density_evictions, 0)
        self.assertIn(gene_from_archive(bank, axis="progress").gene_id, bank.genes)

    def test_parent_and_direction_inheritance(self):
        bank = ParetoGeneBank()
        rng = np.random.default_rng(10)
        seed_gene = propose_gene(0, bank, rng)
        seed_gene.progress_mps, seed_gene.longevity, seed_gene.mean_reward = 0., .1, .1
        bank.accept(seed_gene, rng)
        child = propose_gene(1, bank, rng, de_novo_fraction=0)
        self.assertEqual((child.parent_id, child.generation), (0, 1))
        child.progress_mps, child.longevity, child.mean_reward = 0.1, .12, .12
        record_improvement(child, seed_gene)
        self.assertIsNotNone(child.guide_seed)

    def test_exact_env_interaction_counter(self):
        fake = FakeVecEnv(8)
        genes = [self.make_gene(i, 0, .0) for i in range(8)]
        evaluation_batch(fake, genes, 1000, base_seed=77)
        self.assertEqual(fake.steps, 8000)
        self.assertEqual(fake.step_calls, 1000)
        self.assertEqual(fake.last_seed, 77)
        self.assertAlmostEqual(genes[0].progress_mps, 1.0)
        self.assertAlmostEqual(genes[0].longevity, 1.0)
        self.assertAlmostEqual(genes[0].mean_reward, 1.2, places=5)


    def test_end_to_end_small_budget_no_physics(self):
        import train_segb
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = ["train_segb.py", "--total-steps", "40", "--steps-per-gene", "1",
                    "--worlds", "16", "--save-every", "16", "--out-dir", str(output)]
            made = []
            def fake_maker(n_worlds, *, seed):
                fake = FakeVecEnv(n_worlds)
                made.append(fake)
                return fake
            with patch.object(sys, "argv", args), patch.object(train_segb, "make_env", side_effect=fake_maker):
                train_segb.main()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["interaction_steps"], 40)
            self.assertEqual(summary["candidates"], 40)
            self.assertEqual([x.worlds for x in made], [16, 8])
            self.assertEqual(sum(x.steps for x in made), 40)
            self.assertTrue((output / "best_reward.npz").is_file())
            self.assertEqual(len((output / "candidate_history.csv").read_text().splitlines()), 41)
            with patch.object(sys, "argv", args + ["--resume"]), patch.object(train_segb, "make_env", side_effect=fake_maker):
                train_segb.main()
            self.assertEqual(len(made), 2)  # completed resume does not simulate again

    def test_checkpoint_reproducibility(self):
        bank = ParetoGeneBank(capacity=1024, depth=5)
        rng = np.random.default_rng(112)
        g = self.make_gene(0, 1., .7)
        bank.accept(g, rng)
        cfg = {"archive_capacity": 1024, "archive_depth": 5, "robot_xml_sha256": "unit_test"}
        before = json.dumps(rng.bit_generator.state, sort_keys=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ckpt.npz"
            save_training_state(path, bank=bank, all_best=g, next_gene_index=16,
                                rng=rng, configuration=cfg, history=[])
            restored, best, index, restored_rng, hist = load_training_state(path, cfg)
            self.assertEqual(index, 16)
            self.assertEqual(list(restored.genes), [0])
            self.assertEqual(best.gene_id, 0)
            self.assertEqual(before, json.dumps(restored_rng.bit_generator.state, sort_keys=True))
            model = Path(directory) / "best_reward.npz"
            write_policy(model, best)
            predictor = SEGBPredictor.load(model)
            self.assertEqual(predictor.predict(np.zeros(OBS_DIM, np.float32))[0].shape, (12,))


if __name__ == "__main__":
    unittest.main()
