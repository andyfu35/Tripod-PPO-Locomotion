"""Small report regression checks with deterministic fake evaluation records."""
from __future__ import annotations

import unittest
from compare_segb_h2k6_milestones import compare


def synthetic(speed, fell=False):
    return {"episodes": [{"episode": i, "net_speed_mps": speed,
                           "seconds": 20 if not fell else 2,
                           "fell": fell} for i in range(10)]}


class TestH2K6MilestoneReport(unittest.TestCase):
    def test_paired_heldout_summary(self):
        result = compare(synthetic(.12), synthetic(.19), synthetic(.26))
        self.assertEqual([x["method"] for x in result],
                         ["SEGB_before", "SEGB_after", "PPO"])
        self.assertAlmostEqual(result[1]["mean_speed_mps"] - result[0]["mean_speed_mps"], .07)
        self.assertEqual(result[1]["survival_rate"], 1.0)

    def test_mismatched_episode_counts_rejected(self):
        a = synthetic(.1)
        b = synthetic(.2)
        b["episodes"].pop()
        with self.assertRaisesRegex(ValueError, "episode count"):
            compare(a, b)


if __name__ == "__main__":
    unittest.main()
