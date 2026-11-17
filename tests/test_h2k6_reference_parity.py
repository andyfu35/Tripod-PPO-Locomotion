"""Cross-check tripod H2-K6 math against the original SEGB module copied verbatim.

The canonical research baseline is distributed as a read-only source reference;
tripod uses its own adapted 2-axis physics interface, not the Hopper CLI.
"""
import unittest
import numpy as np

from h2k6_tripod_core import PARAM_COUNT, seeded_direction, mutate

try:
    from segb_h2k6_reference.segb_full import RecurrentPolicyGenome
except ImportError:
    RecurrentPolicyGenome = None


@unittest.skipIf(RecurrentPolicyGenome is None, "Canonical reference not included in local standalone test snapshot")
class TestOriginalSEGBParity(unittest.TestCase):
    def test_original_gru26_weight_count_and_seeded_direction(self):
        self.assertEqual(PARAM_COUNT, RecurrentPolicyGenome.parameter_count(
            51,26,12,architecture="gru"))
        g=RecurrentPolicyGenome.random(observation_dim=51,action_dim=12,
                                       hidden_dim=26,seed=5,architecture="gru")
        for seed in [1,18,12345]:
            self.assertTrue(np.array_equal(seeded_direction(seed), g.seeded_direction(seed)))
            for signed in [-.5,3.0]:
                original=g.mutate(direction_seed=seed,signed_scale=signed).parameters
                adapted=mutate(g.parameters,seed,signed)
                self.assertTrue(np.array_equal(original,adapted))


if __name__=="__main__":
    unittest.main()
