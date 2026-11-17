"""Actual MuJoCo energy reward sanity checks (CI installs MuJoCo/gymnasium)."""
import unittest
import numpy as np

try:
    from tripod_env import TripodEnv
    HAVE_PHYSICS=True
except ImportError:
    HAVE_PHYSICS=False


@unittest.skipUnless(HAVE_PHYSICS, 'MuJoCo and gymnasium not installed')
class EnergyPhysicsTests(unittest.TestCase):
    def test_same_physics_and_exact_reward_difference(self):
        baseline = TripodEnv(energy_weight=0.0)
        energy = TripodEnv(energy_weight=0.1, energy_power_ref_w=50.)
        try:
            x,_ = baseline.reset(seed=123)
            y,_ = energy.reset(seed=123)
            self.assertTrue(np.array_equal(x,y))
            summed=0.
            for i in range(20):
                command = np.full(12, float(0.5*np.sin(i*.4)), np.float32)
                ox,ra,ta,ua,ia = baseline.step(command)
                oy,rb,tb,ub,ib = energy.step(command)
                self.assertTrue(np.allclose(ox,oy,atol=1e-5))
                self.assertEqual((ta,ua),(tb,ub))
                self.assertAlmostEqual(ra-rb,ib['energy_penalty'],places=5)
                self.assertGreaterEqual(ib['actuator_energy_j'],0.)
                self.assertTrue(np.isfinite(ib['actuator_power_w']))
                summed+=ib['actuator_energy_j']
                self.assertAlmostEqual(ib['episode_actuator_energy_j'],summed,places=6)
                if ta or ua:
                    break
            self.assertGreater(summed,0.)
            energy.reset(seed=123)
            self.assertEqual(energy.episode_motor_energy_j,0.)
        finally:
            baseline.close()
            energy.close()


if __name__=='__main__':
    unittest.main()
