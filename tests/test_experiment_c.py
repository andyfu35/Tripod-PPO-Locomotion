"""Static tests for exact Experiment C rollout parameters; no SB3 dependency."""
import ast
import math
from pathlib import Path
import unittest

TRAIN = Path(__file__).resolve().parents[1] / "train.py"


class TestExperimentC(unittest.TestCase):
    def test_c_rollout_updates_and_final_step_count(self):
        worlds, steps_per_world, requested = 16, 1024, 5_000_000
        samples = worlds * steps_per_world
        updates = math.ceil(requested / samples)
        self.assertEqual(samples, 16_384)
        self.assertEqual(updates, 306)
        self.assertEqual(updates * samples, 5_013_504)

    def test_defaults_and_training_code(self):
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        names = {}
        for node in calls:
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            names[node.args[0].value] = next((k.value.value for k in node.keywords
                                             if k.arg == "default" and isinstance(k.value, ast.Constant)), None)
        self.assertEqual(names["--n-envs"], 16)
        self.assertEqual(names["--n-steps"], 1024)
        self.assertEqual(names["--timesteps"], 5_000_000)
        self.assertEqual(names["--out-dir"], "runs_c_16x1024_seed0")


if __name__ == "__main__":
    unittest.main()
