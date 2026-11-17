import json
import tempfile
import unittest
from pathlib import Path
from compare_ppo_segb_energy import compare, summarize_evaluation

class EnergyReportTests(unittest.TestCase):
    def test_fast_fall_does_not_inflate_primary_speed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'summary.json'
            data={'episodes':[
                {'episode':0,'seconds':2.,'net_displacement_m':2.,'motor_work_j':20.,'episode_reward':9.,'fell':True},
                {'episode':1,'seconds':20.,'net_displacement_m':4.,'motor_work_j':200.,'episode_reward':399.,'fell':False},
            ]}
            p.write_text(json.dumps(data))
            out=summarize_evaluation(p,'SEGB')
            self.assertAlmostEqual(out['mean_fixed20_speed_mps'],.15)
            self.assertAlmostEqual(out['mean_active_speed_mps'],.6)
            self.assertAlmostEqual(out['survival_rate'],.5)
            self.assertAlmostEqual(out['mean_motor_power_w_while_alive'],10.)
            self.assertEqual(compare(out,out),[out,out])
            with self.assertRaises(ValueError):
                compare(out,{**out,'episodes':3})
