"""NumPy-only safety checks for energy-matched PPO/SEGB (no MuJoCo needed)."""
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np

from energy_math import (absolute_mechanical_power_w, energy_aware_progress,
                         energy_penalty)
from h2k6_tripod_energy_core import (Evidence, PrivatePsi, Gene, PAESArchive,
                                     PARAM_COUNT, q2_parent_probabilities,
                                     random_parameters, propose_relative)
from train_segb_h2k6_energy import (Config, eval_pair, score_gene,
                                      prepare_batch, commit_results, train,
                                      load_checkpoint, MODEL_PATH)
from hashlib import sha256


class FakeEnergyEnv:
    def __init__(self, dying=4, motor_work_step=0.1):
        self.dying = dying
        self.motor_work_step = motor_work_step
        self.n = 0
        self.step_calls = 0
        self.seeds = []

    def reset(self, *, seed):
        self.seeds.append(seed)
        self.n = 0
        return np.zeros(51, dtype=np.float32), {}

    def step(self, action):
        self.n += 1
        self.step_calls += 1
        assert action.shape == (12,)
        dead = self.n >= self.dying
        return np.zeros(51, dtype=np.float32), 0.20, dead, False, {
            'net_distance_m': self.n * 0.01,
            'actuator_energy_j': self.motor_work_step,
            'alive': not dead,
        }

    def close(self):
        pass


class FakeEnergyPool:
    def __init__(self, dying=4):
        self.dying = dying
        self.steps = 0
        self.seeds = []

    def map(self, callback, tasks):
        results = []
        for task in tasks:
            a, b = FakeEnergyEnv(self.dying), FakeEnergyEnv(self.dying)
            result = callback(task, envs=(a, b))
            self.steps += a.step_calls + b.step_calls
            self.seeds.append((a.seeds[-1], b.seeds[-1]))
            results.append(result)
        return results


class EnergyExperimentTests(unittest.TestCase):
    def test_work_and_reward_units(self):
        power = absolute_mechanical_power_w([2., -3.], [4., 2.])
        self.assertEqual(power, 14.0)
        self.assertAlmostEqual(energy_penalty(0.28, 0.02, weight=0.1,
                           reference_power_w=50.), 0.028)
        self.assertAlmostEqual(energy_aware_progress(4.0, 20.0,
                           horizon_s=20., weight=.1, reference_power_w=50.), .198)
        with self.assertRaises(ValueError):
            absolute_mechanical_power_w([float('nan')], [1.])
        with self.assertRaises(ValueError):
            energy_penalty(-1, 20, weight=.1, reference_power_w=50.)

    def test_energy_axis_is_not_alone_a_fall_reward(self):
        # 0m + 0J has third-axis score 0, never a positive energy 'achievement'.
        self.assertEqual(energy_aware_progress(0, 0, horizon_s=20,
                               weight=.1, reference_power_w=50.), 0.)
        self.assertGreater(energy_aware_progress(4, 50, horizon_s=20,
                               weight=.1, reference_power_w=50.), 0.)

    def test_signed_three_axis_history_private_inheritance(self):
        psi = PrivatePsi('parent', 'founder')
        psi.evidence.append(Evidence(111, 2., (0.1,0.,0.08),
                                    700,700,False,True,101))
        psi.evidence.append(Evidence(222, -3., (-0.2,-0.2,-0.1),
                                    500,300,False,False,101))
        new = psi.inherited_by('child','parent')
        self.assertEqual(len(new.evidence),2)
        new.evidence[0].admitted=False
        self.assertTrue(psi.evidence[0].admitted)
        self.assertEqual(psi.evidence[1].useful_axes(), (False,False,False))
        self.assertEqual(psi.evidence[0].useful_axes(), (True,False,True))
        for k in range(8):
            seed,scale,source=propose_relative(psi,np.random.default_rng(k),random_fraction=0.)
            self.assertEqual((seed,source),(111,'relative_guided'))
        with self.assertRaises(ValueError):
            Evidence(99,1.,(.1,.2),100,100,True,False,99)

    def test_three_axis_bank_pareto_and_q2(self):
        bank=PAESArchive()
        rng=np.random.default_rng(13)
        def g(k, movement,life,net):
            x=Gene(str(k),random_parameters(rng),PrivatePsi(str(k),str(k)))
            x.progress=movement
            x.longevity=life
            x.energy_aware_progress=net
            return x
        a,b,c,d=g(0,.2,1.,.14),g(1,.16,1.,.15),g(2,.1,.2,-.1),g(3,.22,1.,.19)
        admitted=bank.commit_batch([a,b,c],rng)
        self.assertEqual(admitted,{'0','1'})
        probs=q2_parent_probabilities(bank.ordered())
        self.assertAlmostEqual(float(probs.sum()),1.)
        bank.commit_batch([d],rng)
        self.assertEqual(list(bank.genes),['3'])

    def test_score_full_horizon_and_early_fall_no_per_second_shortcut(self):
        cfg=Config()
        g=Gene('a',random_parameters(np.random.default_rng(1)),PrivatePsi('a','a'))
        r={'completed':True,'censored':False,'distance':2.,'steps':100,'reward':10.,'energy_j':10.}
        score_gene(g,r,1000,202,cfg)
        self.assertAlmostEqual(g.progress,.1)
        self.assertAlmostEqual(g.longevity,.1)
        self.assertAlmostEqual(g.energy_aware_progress,.099)
        self.assertEqual(g.motor_energy_j,10.)

    def test_distinct_pair_founders_and_matched_seeds(self):
        cfg=Config()
        rng=np.random.default_rng(2)
        tasks,entries=prepare_batch(cfg, PAESArchive(), rng, 0, 8, 16000)
        ids=set()
        for task,entry in zip(tasks,entries):
            self.assertNotEqual(task['parent_id'],task['child_id'])
            self.assertNotIn(task['parent_id'],ids)
            ids.add(task['parent_id'])
            ids.add(task['child_id'])
            self.assertEqual(entry['mode'],'founder')
        t=tasks[0]
        a,b=FakeEnergyEnv(3),FakeEnergyEnv(2)
        rr=eval_pair(t,envs=(a,b))
        self.assertEqual(rr['used'],5)
        self.assertEqual(a.seeds,b.seeds)
        self.assertAlmostEqual(rr['results'][0]['energy_j'],.3)
        self.assertAlmostEqual(rr['results'][1]['energy_j'],.2)

    def test_batch_persists_all_three_signed_axes(self):
        cfg=Config()
        rng=np.random.default_rng(26)
        bank=PAESArchive()
        parent=Gene('g0',random_parameters(rng),PrivatePsi('g0','g0'))
        parent.progress,parent.longevity,parent.energy_aware_progress=.01,.9,0.
        bank.commit_batch([parent],rng)
        child=Gene('g1',random_parameters(rng),PrivatePsi('g1','g0',parent='g0'),
                   parent_id='g0',generation=1)
        entries=[dict(parent=parent,child=child,seed=42,signed_scale=1.,
                      mode='relative_guided',reset_seed=100)]
        results=[dict(pair_seed=100,results=[
            dict(completed=True,censored=False,distance=.2,steps=200,energy_j=8.,fell=True,reward=35.),
            dict(completed=True,censored=False,distance=1.,steps=300,energy_j=3.,fell=True,reward=45.)])]
        history=[]
        commit_results(entries,results,bank=bank,rng=rng,histories=history,
                       steps_used=500,best_reward=None,cfg=cfg)
        self.assertEqual(len(parent.psi.evidence),1)
        ev=parent.psi.evidence[0]
        self.assertEqual(len(ev.delta),3)
        self.assertEqual((ev.parent_steps,ev.child_steps),(200,300))
        self.assertTrue(ev.delta[2]>0)
        self.assertEqual(history[0]['motor_energy_j'],3.)

    def test_budget_and_resume_and_energy_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=replace(Config(),total_steps=101,checkpoint_every=29,
                        output=str(Path(d)/'energy_run'))
            fake=FakeEnergyPool(3)
            steps=train(cfg,pool=fake,verbose=False)
            self.assertEqual((steps,fake.steps),(101,101))
            self.assertEqual((Path(cfg.output)/'summary.json').exists(),True)
            self.assertTrue((Path(cfg.output)/'best_stable_energy_aware.npz').is_file())
            self.assertTrue(all(s==t for s,t in fake.seeds))
            self.assertEqual(train(cfg,pool=fake,verbose=False),101)
            self.assertEqual(fake.steps,101)
            ck=Path(cfg.output)/'checkpoint.npz'
            src_hash=sha256(MODEL_PATH.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                load_checkpoint(ck,cfg=replace(cfg,energy_weight=.2),robot_hash=src_hash)
            with self.assertRaises(ValueError):
                load_checkpoint(ck,cfg=cfg,robot_hash='wrong')
            self.assertEqual(json.loads((Path(cfg.output)/'summary.json').read_text())['interaction_steps'],101)


if __name__ == '__main__':
    unittest.main()
