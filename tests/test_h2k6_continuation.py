"""Regression checks for extending a FULL H2-K6 private-history run.

No MuJoCo is required. Fake physics counts every step, making budget and
checkpoint invariants testable on GitHub Actions / macOS independently.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from h2k6_tripod_core import Evidence, Gene, PrivatePsi, PAESArchive, random_parameters
from train_segb_h2k6 import (
    Config, MODEL_PATH, load_checkpoint, save_checkpoint, train,
)
from test_h2k6_tripod import FakePool


class TestH2K6Continuation(unittest.TestCase):
    def test_external_extension_preserves_source_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "runs_5m"
            continued = Path(tmp) / "runs_10m"
            cfg5 = Config(total_steps=101, checkpoint_every=25, output=str(original))
            fake5 = FakePool(die_on_step=3)
            self.assertEqual(train(cfg5, pool=fake5, verbose=False), 101)
            src = original / "checkpoint.npz"
            before = hashlib.sha256(src.read_bytes()).hexdigest()
            old_result = json.loads((original / "summary.json").read_text())
            cfg10 = replace(cfg5, total_steps=207, output=str(continued), checkpoint_every=17)
            fake10 = FakePool(die_on_step=3)
            self.assertEqual(train(cfg10, pool=fake10, verbose=False, resume_from=src), 207)
            self.assertEqual(fake10.steps, 106)
            self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), before)
            self.assertEqual(json.loads((original / "summary.json").read_text()), old_result)
            self.assertEqual(json.loads((continued / "summary.json").read_text())["interaction_steps"], 207)
            trace = json.loads((continued / "continuation_provenance.json").read_text())
            self.assertEqual(trace["source_steps"], 101)
            self.assertEqual(trace["source_sha256"], before)
            cfg_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
            _, _, _, _, _, original_history, _ = load_checkpoint(src, cfg=cfg5, robot_hash=cfg_hash)
            _, _, _, _, _, continued_history, _ = load_checkpoint(
                continued / "checkpoint.npz", cfg=cfg10, robot_hash=cfg_hash)
            self.assertEqual(continued_history[:len(original_history)], original_history)
            before_second = fake10.steps
            self.assertEqual(train(cfg10, pool=fake10, verbose=False, resume_from=src), 207)
            self.assertEqual(fake10.steps, before_second)

    def test_private_evidence_survives_extension_without_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg5 = Config(total_steps=100, output=str(Path(tmp) / "old"))
            cfg10 = replace(cfg5, total_steps=200, output=str(Path(tmp) / "new"))
            rng = np.random.default_rng(42)
            p = random_parameters(rng)
            state = PrivatePsi(owner="g1", lineage="root")
            state.evidence.append(Evidence(
                direction_seed=333, signed_scale=-2.0, delta=(0.4, -0.2),
                parent_steps=1000, child_steps=800, survived=False,
                admitted=False, reset_seed=77))
            g = Gene("g1", p, state)
            g.progress = 0.3
            g.longevity = 0.8
            bank = PAESArchive()
            bank.commit_batch([g], rng)
            src = Path(cfg5.output) / "checkpoint.npz"
            robot_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
            save_checkpoint(src, cfg=cfg5, robot_hash=robot_hash, bank=bank, rng=rng,
                            steps=100, next_id=74, batch=8, history=[{"gene_id":"g1"}], best=g)
            before = json.dumps(rng.bit_generator.state, sort_keys=True)
            restored, rng2, steps, next_id, batch, history, best = load_checkpoint(
                src, cfg=cfg10, robot_hash=robot_hash, allow_extension=True)
            self.assertEqual((steps, next_id, batch), (100, 74, 8))
            self.assertEqual(history, [{"gene_id":"g1"}])
            self.assertEqual(json.dumps(rng2.bit_generator.state, sort_keys=True), before)
            self.assertEqual(best.id, "g1")
            ev = restored.genes["g1"].psi.evidence
            self.assertEqual(len(ev), 1)
            self.assertEqual((ev[0].direction_seed, ev[0].signed_scale, ev[0].parent_steps,
                              ev[0].child_steps, ev[0].delta), (333, -2.0, 1000, 800, (0.4, -0.2)))
            with self.assertRaisesRegex(ValueError, "target must exceed"):
                load_checkpoint(src, cfg=replace(cfg10, total_steps=100),
                                robot_hash=robot_hash, allow_extension=True)

    def test_extension_rejects_parameter_and_xml_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(total_steps=100, output=str(Path(tmp) / "old"))
            rng = np.random.default_rng(1)
            p = random_parameters(rng)
            g = Gene("g0", p, PrivatePsi(owner="g0", lineage="g0"))
            g.progress = .2
            g.longevity = 1.
            bank = PAESArchive()
            bank.commit_batch([g], rng)
            src = Path(cfg.output) / "checkpoint.npz"
            robot_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
            save_checkpoint(src, cfg=cfg, robot_hash=robot_hash, bank=bank, rng=rng,
                            steps=100, next_id=10, batch=2, history=[], best=g)
            base = replace(cfg, total_steps=200, output=str(Path(tmp) / "new"))
            for changed in (replace(base, seed=100), replace(base, sigma_max=10.),
                            replace(base, de_novo_fraction=.4), replace(base, worlds=8)):
                with self.subTest(changed=changed):
                    with self.assertRaisesRegex(ValueError, "evolutionary settings"):
                        load_checkpoint(src, cfg=changed, robot_hash=robot_hash,
                                        allow_extension=True)
            with self.assertRaisesRegex(ValueError, "XML mismatch"):
                load_checkpoint(src, cfg=base, robot_hash="different_model_hash",
                                allow_extension=True)
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                load_checkpoint(src, cfg=base, robot_hash=robot_hash)


if __name__ == "__main__":
    unittest.main()
